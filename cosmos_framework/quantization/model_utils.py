"""Fuse sibling linears for calibration, then unfuse for a stock HF checkpoint.

vLLM fuses QKV (and Super/Nano gate/up) at runtime and shares one activation
scale. This module concatenates those linears into one ``FusedLinear`` during
``mtq.quantize`` so ModelOpt calibrates that shared scale, then splits them
back into named ``nn.Linear``s with sliced weights and quantizers.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Generator
from contextlib import ExitStack, contextmanager

import torch
from modelopt.torch.opt.conversion import ModeloptStateManager
from modelopt.torch.quantization.conversion import update_quantize_metadata
from torch import nn

_QUANTIZER_ATTRS = ("input_quantizer", "weight_quantizer", "output_quantizer")
# Per-tensor / per-input scales: never slice even if leading dim happens to match out_features.
_NEVER_SLICE_BUFFERS = frozenset({"_global_amax", "_pre_quant_scale"})


class FusedLinear(nn.Linear):
    """Concatenated QKV (or gate/up). Same forward as nn.Linear so ModelOpt converts it."""

    def __init__(
        self,
        in_features: int,
        out_features_per_key: dict[str, int],
        *,
        bias: bool,
        device=None,
        dtype=None,
    ):
        self.key_to_out_features = dict(out_features_per_key)
        super().__init__(
            in_features,
            sum(out_features_per_key.values()),
            bias=bias,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_linears(cls, /, **linears: nn.Linear) -> FusedLinear:
        first = next(iter(linears.values()))
        fused = cls(
            first.in_features,
            {k: linear.out_features for k, linear in linears.items()},
            bias=first.bias is not None,
            device=first.weight.device,
            dtype=first.weight.dtype,
        )
        with torch.no_grad():
            fused.weight.copy_(torch.cat([linear.weight for linear in linears.values()], dim=0))
            if fused.bias is not None:
                fused.bias.copy_(torch.cat([linear.bias for linear in linears.values()], dim=0))
        return fused

    def ranges(self) -> dict[str, tuple[int, int]]:
        out, start = {}, 0
        for key, n in self.key_to_out_features.items():
            out[key] = (start, start + n)
            start += n
        return out


class LinearView(nn.Module):
    """Drop-in for q_proj/k_proj/v_proj. Not a Linear, so ModelOpt will not convert it."""

    def __init__(self, fused: FusedLinear, key: str):
        super().__init__()
        # Do not assign fused as a submodule (would re-parent / triple-count the weight).
        object.__setattr__(self, "_fused", fused)
        self.key = key
        self.in_features = fused.in_features
        self.out_features = fused.key_to_out_features[key]
        self._start, self._end = fused.ranges()[key]

    def _slice(self, stacked: torch.Tensor) -> torch.Tensor:
        return stacked.narrow(-1, self._start, self._end - self._start)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = self._fused
        cache = getattr(fused, "_view_cache", None)
        if cache is None or cache[0] is not x:
            # One quantized GEMM per unique activation tensor (q/k/v share this).
            fused._view_cache = (x, fused(x))
        return self._slice(fused._view_cache[1])

    def _raw_weight(self) -> torch.Tensor:
        fused = self._fused
        # Prefer the Parameter; after ModelOpt convert, `.weight` can be a fake-quant view.
        weight = fused._parameters.get("weight")
        if weight is None:
            weight = fused.weight
        if type(weight).__name__ == "QTensorWrapper":
            raise RuntimeError(
                "Cannot unfuse packed/real-quant weights. Unfuse after calibration and before mtq.compress()."
            )
        return weight[self._start : self._end]

    @property
    def weight(self) -> torch.Tensor:
        return self._raw_weight()

    @property
    def bias(self) -> torch.Tensor | None:
        fused = self._fused
        bias = fused._parameters.get("bias") if fused.bias is not None else None
        if bias is None:
            return None
        return bias[self._start : self._end]

    def to_linear(self) -> nn.Linear:
        """Materialize a real Linear, copying fused quantizers when present.

        ``input_quantizer`` (and scalar weight scales / NVFP4 ``_global_amax`` /
        SmoothQuant ``_pre_quant_scale``) are broadcast. Per-row and flattened
        per-block weight state is sliced to this view.
        """
        fused = self._fused
        layer = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        with torch.no_grad():
            layer.weight.copy_(self._raw_weight())
            if self.bias is not None:
                layer.bias.copy_(self.bias)

        if not hasattr(fused, "weight_quantizer"):
            return layer

        from modelopt.torch.quantization.nn import QuantModuleRegistry

        layer = QuantModuleRegistry.convert(layer)
        fused_out = fused.out_features
        for name in _QUANTIZER_ATTRS:
            if not hasattr(fused, name):
                continue
            src = getattr(fused, name)
            if name == "input_quantizer":
                setattr(layer, name, copy.deepcopy(src))
            else:
                setattr(
                    layer,
                    name,
                    _clone_quantizer_for_rows(
                        src,
                        self._start,
                        self._end,
                        fused_out,
                        in_features=self.in_features if name == "weight_quantizer" else None,
                    ),
                )
        return layer


def _last_axis_block_size(module: nn.Module) -> int | None:
    """Return a quantizer's concrete last-axis block size, if it has one."""
    block_sizes = getattr(module, "block_sizes", None)
    if not isinstance(block_sizes, dict):
        return None
    block_size = block_sizes.get(-1) or block_sizes.get(1)
    return block_size if isinstance(block_size, int) and block_size > 0 else None


def _slice_output_tensor(
    value: torch.Tensor,
    start: int,
    end: int,
    fused_out: int,
    *,
    in_features: int | None,
    block_size: int | None,
) -> torch.Tensor:
    """Slice row-indexed state in either per-row or flattened per-block layout."""
    if value.ndim == 0 or value.numel() == 1:
        return value.detach().contiguous().clone()

    if value.shape[0] == fused_out:
        return value[start:end].detach().contiguous().clone()

    if in_features is not None and block_size is not None:
        blocks_per_row = (in_features + block_size - 1) // block_size
        flattened_rows = fused_out * blocks_per_row
        if value.shape[0] == flattened_rows:
            trailing_shape = value.shape[1:]
            by_output_row = value.reshape(fused_out, blocks_per_row, *trailing_shape)
            sliced = by_output_row[start:end]
            return sliced.reshape((end - start) * blocks_per_row, *trailing_shape).detach().contiguous().clone()

    return value.detach().contiguous().clone()


def _update_output_shape_metadata(module: nn.Module, out_features: int) -> None:
    """Update static-block metadata cached when the fused weight was calibrated."""
    original_shape = getattr(module, "_original_shape", None)
    if original_shape is not None and len(original_shape) > 0:
        shape = list(original_shape)
        shape[0] = out_features
        module._original_shape = torch.Size(shape)

    amax_shape = getattr(module, "_amax_shape_for_export", None)
    if amax_shape is not None and len(amax_shape) > 0:
        shape = list(amax_shape)
        shape[0] = out_features
        module._amax_shape_for_export = tuple(shape)


def _slice_out_feature_state(
    module: nn.Module,
    start: int,
    end: int,
    fused_out: int,
    *,
    in_features: int | None = None,
) -> None:
    """Keep this view's rows of per-output quantizer state, in place.

    Per-row state uses ``(fused_out, ...)``. ModelOpt static block quantizers
    instead flatten the first two logical dimensions to
    ``(fused_out * blocks_per_row, ...)``. Recover the logical row dimension
    from the quantizer's configured block size before slicing. Per-tensor and
    per-input buffers (``_global_amax``, ``_pre_quant_scale``) remain shared.
    Recurses into children so every member of a ``SequentialQuantizer`` uses
    its own block-size configuration.
    """
    block_size = _last_axis_block_size(module) if in_features is not None else None
    for name, buf in list(module._buffers.items()):
        if name in _NEVER_SLICE_BUFFERS:
            continue
        if buf is not None:
            module._buffers[name] = _slice_output_tensor(
                buf,
                start,
                end,
                fused_out,
                in_features=in_features,
                block_size=block_size,
            )
    for name, param in list(module._parameters.items()):
        if name in _NEVER_SLICE_BUFFERS:
            continue
        if param is not None:
            module._parameters[name] = nn.Parameter(
                _slice_output_tensor(
                    param,
                    start,
                    end,
                    fused_out,
                    in_features=in_features,
                    block_size=block_size,
                ),
                requires_grad=param.requires_grad,
            )

    if in_features is not None:
        _update_output_shape_metadata(module, end - start)
        amax = getattr(module, "_amax", None)
        if block_size is not None and amax is not None and getattr(module, "_is_nvfp4_static_quantizer", False):
            blocks_per_row = (in_features + block_size - 1) // block_size
            expected_numel = (end - start) * blocks_per_row
            if amax.numel() != expected_numel:
                raise ValueError(
                    "Could not partition static NVFP4 state by output row: "
                    f"expected {expected_numel} amax values for {(end - start)}x{in_features} "
                    f"with block size {block_size}, got shape {tuple(amax.shape)}."
                )
    for child in module.children():
        _slice_out_feature_state(child, start, end, fused_out, in_features=in_features)


def _clone_quantizer_for_rows(
    quantizer: nn.Module,
    start: int,
    end: int,
    fused_out: int,
    *,
    in_features: int | None = None,
) -> nn.Module:
    cloned = copy.deepcopy(quantizer)
    _slice_out_feature_state(cloned, start, end, fused_out, in_features=in_features)
    return cloned


def _fuse_named_linears(parent: nn.Module, fused_name: str, keys: list[str]) -> FusedLinear:
    linears = {k: getattr(parent, k) for k in keys}
    fused = FusedLinear.from_linears(**linears)
    parent.register_module(fused_name, fused)  # real submodule → ModelOpt finds it
    for k in keys:
        setattr(parent, k, LinearView(fused, k))
    return fused


@contextmanager
def fuse_named_linears_context(parent: nn.Module, fused_name: str, keys: list[str]) -> Generator[None, None, None]:
    """Replace ``parent``'s named linears with one fused GEMM for the context body.

    Registers ``fused_name`` (e.g. ``qkv_proj``) as a real ``FusedLinear``
    submodule so ModelOpt converts it, and swaps each of ``keys``
    (``q_proj``, ``k_proj``, ...) for a ``LinearView`` that slices that GEMM.
    On exit, deletes the fused module and materializes real linears with
    copied weights and quantizers. Does not rewrite ModelOpt metadata; call
    ``refresh_modelopt_quantizer_state`` on the quantized root after unfuse
    (see ``fuse_cosmos3_linears``).
    """
    _fuse_named_linears(parent, fused_name, keys)
    try:
        yield
    finally:
        # Convert is in-place, but re-bind in case replace_quant_module swapped the object.
        fused = getattr(parent, fused_name)
        views = {}
        for k in keys:
            view = getattr(parent, k)
            object.__setattr__(view, "_fused", fused)
            views[k] = view
        delattr(parent, fused_name)
        for k, view in views.items():
            setattr(parent, k, view.to_linear())


@contextmanager
def temporary_fuse_named_linears(model: nn.Module, parent_regex: str, fused_name: str, keys: list[str]):
    try:
        with ExitStack() as stack:
            for name, module in model.named_modules():
                if re.search(parent_regex, name):
                    stack.enter_context(fuse_named_linears_context(module, fused_name, keys))
            yield
    finally:
        refresh_modelopt_quantizer_state(model)


def refresh_modelopt_quantizer_state(model: nn.Module) -> None:
    """Rewrite every mode's ``quantizer_state`` from the live graph.

    ``mtq.quantize`` snapshots names while fused (``qkv_proj``). Unfuse must
    refresh *all* modes — ``mto.modelopt_state`` only updates the last one.
    Call this on the same module passed to ``mtq.quantize``, after unfuse.
    """

    if not ModeloptStateManager.is_converted(model):
        return
    manager = ModeloptStateManager(model)
    for _, config, metadata in manager.modes_with_states():
        if "quantizer_state" in metadata:
            update_quantize_metadata(model, config, metadata)


__all__ = [
    "fuse_named_linears_context",
    "temporary_fuse_named_linears",
    "refresh_modelopt_quantizer_state",
]
