# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""NATTEN parameter validation and metadata generation."""

from collections.abc import Mapping, Sequence

import torch

from cosmos_framework.model.attention.checks import check_valid_tuple_or_element
from cosmos_framework.model.attention.varlen import generate_multi_dim_varlen_parameters
from cosmos_framework.utils import log


def _validate_single_dim_params(params: Mapping, layer_idx: int, num_dims: int | None) -> dict:
    """
    Helper function to validate NATTEN parameters for a dimensionality profile.

    Args:
        params (Mapping): parameter dict with window_size/window_size_float and other params
        layer_idx (int): layer index for error messages
        num_dims (int | None): 1, 2, 3, or None (for single-profile format)

    Returns:
        dict: validated parameter dict with proper types
    """
    if not isinstance(params, Mapping):
        dim_str = f" ({num_dims}-D)" if num_dims else ""
        raise ValueError(f"Parameters for layer {layer_idx}{dim_str} must be a dict or None, got {params=}.")

    is_causal = False if "is_causal" not in params else params["is_causal"]

    if "window_size_float" in params:
        window_size_float = params["window_size_float"]
        if (
            not isinstance(window_size_float, Sequence)
            or len(window_size_float) not in [1, 2, 3]
            or any(not isinstance(x, float) for x in window_size_float)
        ):
            raise ValueError(f"'window_size_float' must be a float tuple of size 1, 2, or 3, got {window_size_float=}")
        window_size_float = tuple(k for k in window_size_float)

        num_dims = len(window_size_float)

        def check_stride_dilation(x):
            if isinstance(x, float):
                if 0.0 <= x <= 1.0:
                    return tuple(x for _ in range(num_dims))
            elif (
                isinstance(x, Sequence)
                and len(x) == num_dims
                and all(isinstance(y, float) and 0.0 <= y <= 1.0 for y in x)
            ):
                return tuple(y for y in x)
            else:
                raise ValueError(f"Invalid natten float parameter: {x=}")

        stride_float = 0.0 if "stride_float" not in params else params["stride_float"]
        dilation_float = 0.0 if "dilation_float" not in params else params["dilation_float"]

        stride_float = check_stride_dilation(stride_float)
        dilation_float = check_stride_dilation(dilation_float)
        is_causal = check_valid_tuple_or_element(
            is_causal, num_dims=num_dims, typename=bool, raise_error=True, param_name="is_causal"
        )

        if any(x in params for x in ["window_size", "stride", "dilation"]):
            raise ValueError(
                f"Please either use _float parameters, or integer ones, and not mix the two. Got {params=}."
            )

        return {
            "window_size_float": window_size_float,
            "stride_float": stride_float,
            "dilation_float": dilation_float,
            "is_causal": is_causal,
        }

    elif "window_size" in params:
        window_size = params["window_size"]
        num_dims = len(window_size)

        stride = 1 if "stride" not in params else params["stride"]
        dilation = 1 if "dilation" not in params else params["dilation"]

        if any("_float" in x for x in params.keys()):
            raise ValueError(
                f"Please either use _float parameters, or integer ones, and not mix the two. Got {params=}."
            )

        window_size = check_valid_tuple_or_element(
            window_size, num_dims=num_dims, typename=int, raise_error=True, param_name="window_size"
        )
        stride = check_valid_tuple_or_element(
            stride, num_dims=num_dims, typename=int, raise_error=True, param_name="stride"
        )
        dilation = check_valid_tuple_or_element(
            dilation, num_dims=num_dims, typename=int, raise_error=True, param_name="dilation"
        )
        is_causal = check_valid_tuple_or_element(
            is_causal, num_dims=num_dims, typename=bool, raise_error=True, param_name="is_causal"
        )

        return {"window_size": window_size, "stride": stride, "dilation": dilation, "is_causal": is_causal}
    else:
        raise ValueError(
            "Sparse parameters for a layer must have key 'window_size' or 'window_size_float', "
            f"got {params=} in layer index {layer_idx}."
        )


def verify_natten_parameter_list(
    natten_parameter_list: list | None,
    num_layers: int,
) -> list | None:
    """
    Converts list of NATTEN parameters into expected types, and assigns defaults to unset
    parameters.
    This needs to be done separately during model initialization, and not forward pass.
    There are no torch operations in this function.

    Args:
        natten_parameter_list (list | None): list of NATTEN parameters. Must be either None, or a
            list of mappings, one for each layer. Each list element must be either None,
            representing no sparsity / masking (full dense attention), or a mapping of NATTEN
            parameters.

            Parameters can be specified directly with integer or float format:
                - 'window_size_float' (required), 'stride_float', 'dilation_float'
                - 'window_size' (required), 'stride', 'dilation'

            Or, parameters can be specified for multiple dimensionality profiles in case of
            mixed-training (i.e. image and video training) using keys "1d", "2d", "3d":
                - Each key maps to either None (dense attention) or a parameter dict

            Integer and float parameters cannot be used together in the same layer!
            Additionally, you can specify 'is_causal'.

            Examples:
            ```
            # 50 percent sparsity along each dimension in a 2-D token layout
            {'window_size_float': (0.5, 0.5)}  # valid

            # 50 percent sparsity along each dimension in a 2-D token layout
            # Maximum dilation along first dimension, no dilation along second dimension
            {'window_size_float': (0.5, 0.5), 'dilation_float': (1.0, 0.0)}  # valid

            # Fixed window size of 8x8, dilation of 2x1.
            # NOTE: requires ALL inputs to be at least 16x8
            {'window_size': (8, 8), 'dilation': (2, 1)}  # valid

            # Multi-profile: different parameters for 2D (images) and 3D (videos)
            {
                "2d": {"window_size_float": (0.5, 0.5)},
                "3d": {"window_size_float": (1.0, 0.5, 0.5)}
            }  # valid

            # Multi-profile: 2D uses dense attention, 3D uses sparse
            {
                "2d": None,
                "3d": {"window_size_float": (1.0, 0.5, 0.5)}
            }  # valid

            # Invalid:
            {'window_size_float': (0.5, 0.5), 'dilation': (2, 1)}
            ```

        num_layers (int): number of layers in the model. Just used to verify list length.

    Returns:
        output_parameter_list (list | None): verified and type-checked NATTEN parameters, or None if
            no parameters passed.
    """

    if natten_parameter_list is not None:
        parameter_list_out = []
        if not isinstance(natten_parameter_list, Sequence):
            raise ValueError(f"Argument 'natten_parameter_list' must be a list or None, got {natten_parameter_list=}.")

        if len(natten_parameter_list) != num_layers:
            raise ValueError(
                "Number of elements in 'natten_parameter_list' must match number of layers "
                f"in the model, got {num_layers=}, {len(natten_parameter_list)=}."
            )

        for i, layer_parameters in enumerate(natten_parameter_list):
            if layer_parameters is None:
                log.debug(f"Layer {i} will use DENSE attention.")
                parameter_list_out.append(None)
                continue

            if not isinstance(layer_parameters, Mapping):
                raise ValueError(
                    f"Sparse parameters for a layer must be a dict or None, got {layer_parameters=} in layer index {i}."
                )

            # Detect format: multi-profile if has keys "1d", "2d", or "3d"
            dim_keys = {"1d", "2d", "3d"}
            has_dim_keys = any(k in layer_parameters for k in dim_keys)

            if has_dim_keys:
                # Multi-profile format: validate each explicitly defined dimensionality profile
                validated_multi_profile = {}
                for dim_str, dim_int in [("1d", 1), ("2d", 2), ("3d", 3)]:
                    if dim_str in layer_parameters:
                        dim_params = layer_parameters[dim_str]
                        if dim_params is None:
                            validated_multi_profile[dim_int] = None
                        else:
                            validated_multi_profile[dim_int] = _validate_single_dim_params(dim_params, i, dim_int)
            else:
                # Single-profile format: validate and convert to multi-profile format
                # Infer dimensionality from parameter tuple length
                validated_params = _validate_single_dim_params(layer_parameters, i, None)
                if "window_size_float" in validated_params:
                    num_dims = len(validated_params["window_size_float"])
                else:  # "window_size"
                    num_dims = len(validated_params["window_size"])
                validated_multi_profile = {num_dims: validated_params}

            # If all explicitly defined profiles are None, treat as fully dense layer
            if all(v is None for v in validated_multi_profile.values()):
                log.debug(f"Layer {i} will use DENSE attention (all profiles None).")
                parameter_list_out.append(None)
            else:
                parameter_list_out.append(validated_multi_profile)
                log.info(f"Layer {i} NATTEN parameters: {validated_multi_profile}")

        return parameter_list_out

    return None


def generate_natten_metadata(
    token_shapes: list[tuple[int, int, int]],
    head_dim: int,
    num_layers: int,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
    natten_parameter_list: list | None = None,
) -> list | None:
    """
    Generates list of metadata required by Variable-Sized (variable-length) operations in NATTEN.
    Required when training with three_way attention and NATTEN (multi-dimensional / sparse
    attention).

    Args:
        token_shapes (list[tuple]): list of integer tuples corresponding to the
            post-tokenization/patchify token layout shapes in the packed sequence. Must strictly be
            integer tuples with the same profile (all 1D, 2D, or 3D). 1s will be automatically
            stripped (i.e. [(1, 8, 8), (1, 16, 16)] is interpreted as [(8, 8), (16, 16)]).

        head_dim (int): Attention head dimension (used to select NATTEN kernel configurations).

        num_layers (int): number of layers in the model. Just used to verify list length.

        device (torch.device): PyTorch device for offset tensors (should match QKV device).

        dtype (torch.dtype): Expected QKV dtype.

        requires_grad (bool): Determines whether backprop is expected, and sets up metadata for
            backward pass as well.

        natten_parameter_list (list | None): list of NATTEN parameters. Must be either None, or a
            list of mappings, one for each layer. Each list element must be either None,
            representing no sparsity / masking (full dense attention), or a mapping of NATTEN
            parameters in either integer or float format:
                - 'window_size_float' (required), 'stride_float', 'dilation_float'
                - 'window_size' (required), 'stride', 'dilation'

            Integer and float parameters cannot be used together in the same layer!
            Additionally, you can specify 'is_causal'.

            Examples:
            ```
            # 50 percent sparsity along each dimension in a 2-D token layout
            {'window_size_float': (0.5, 0.5)}  # valid

            # 50 percent sparsity along each dimension in a 2-D token layout
            # Maximum dilation along first dimension, no dilation along second dimension
            {'window_size_float': (0.5, 0.5), 'dilation_float': (1.0, 0.0)}  # valid

            # Fixed window size of 8x8, dilation of 2x1.
            # NOTE: requires ALL inputs to be at least 16x8
            {'window_size': (8, 8), 'dilation': (2, 1)}  # valid

            # Invalid:
            {'window_size_float': (0.5, 0.5), 'dilation': (2, 1)}
            ```

    Returns:
        natten_metadata_list (list | None): list of NATTEN varlen metadata, or Nones (dense layers).
            Each non-None element will be a dictionary containing final parameters, and varlen
            metadata (offset and size tensors, max lengths).
            NOTE: to avoid excessive recompilations in torch.compile, we must carefully index into
            this list during model.forward, and ideally using the iteration counter from the loop
            over layers (nn.ModuleList).
    """


    if token_shapes is None or len(token_shapes) < 1:
        raise ValueError("'token_shapes' is required for 'three_way' attention.")

    natten_metadata = None

    if natten_parameter_list is not None:
        natten_metadata = []
        if not isinstance(natten_parameter_list, list):
            raise ValueError(f"Argument 'natten_parameter_list' must be a list or None, got {natten_parameter_list=}.")

        if len(natten_parameter_list) != num_layers:
            raise ValueError(
                "Number of elements in 'natten_parameter_list' must match number of layers "
                f"in the model, got {num_layers=}, {len(natten_parameter_list)=}."
            )

        # We need to filter out 1s from shapes
        def filter_shape(shape: tuple) -> tuple:
            return tuple(x for x in shape if x > 1)

        # Infer token layout rank (dimensionality)
        num_dims = max([len(filter_shape(token_shape)) for token_shape in token_shapes])

        # Single pass: check if all layers support this dimensionality and if any need processing
        needs_processing = False
        for i, layer_parameters in enumerate(natten_parameter_list):
            if layer_parameters is None:
                continue

            # Fail fast if this dimensionality is not defined
            if num_dims not in layer_parameters:
                raise ValueError(
                    f"Layer {i}: batch has {num_dims}D data but parameters are not defined for {num_dims}D. "
                    f"Defined dimensionalities: {sorted(layer_parameters.keys())}"
                )

            # Check if this layer needs processing for this dimensionality
            if layer_parameters[num_dims] is not None:
                needs_processing = True

        # Early exit if all layers are dense for this dimensionality profile
        if not needs_processing:
            log.debug(f"All layers use DENSE attention for {num_dims}D data.")
            return None

        # We actually need to process, so validate and filter all shapes
        token_layout_list = []
        for shape in token_shapes:
            assert isinstance(shape, tuple)
            shape_filtered = filter_shape(shape)
            assert len(shape_filtered) == num_dims, (
                f"All data in batch must have same dimensionality, got {num_dims}D and {len(shape_filtered)}D"
            )
            token_layout_list.append(shape_filtered)

        log.debug(f"Batch dimensionality: {num_dims}D, token_layout_list={token_layout_list}")

        for i, layer_parameters in enumerate(natten_parameter_list):
            if layer_parameters is None:
                natten_metadata.append(None)
                continue

            # Get parameters for this dimensionality (already validated above)
            dim_params = layer_parameters[num_dims]

            if dim_params is None:
                # Dense attention for this dimensionality
                natten_metadata.append(None)
                continue

            # Use dim_params (parameters for this specific dimensionality)
            window_size_list = []
            stride_list = []
            dilation_list = []

            if "window_size_float" in dim_params:
                window_size_float = dim_params["window_size_float"]
                stride_float = dim_params["stride_float"]
                dilation_float = dim_params["dilation_float"]

                for token_layout in token_layout_list:
                    window_size_ = tuple(
                        min(x, max(2, int(k * float(x)))) for k, x in zip(window_size_float, token_layout)
                    )
                    stride_ = tuple(min(k, max(1, int(s * float(k)))) for s, k in zip(stride_float, window_size_))
                    max_dilation = tuple(x // k for k, x in zip(window_size_, token_layout))
                    dilation_ = tuple(min(m, max(1, int(d * float(m)))) for d, m in zip(dilation_float, max_dilation))

                    window_size_list.append(window_size_)
                    stride_list.append(stride_)
                    dilation_list.append(dilation_)

                assert len(window_size_list) == len(stride_list) == len(dilation_list) == len(token_layout_list)

                log.debug(f"Layer {i}: {window_size_list=}")
                log.debug(f"Layer {i}: {stride_list=}")
                log.debug(f"Layer {i}: {dilation_list=}")

            elif "window_size" in dim_params:
                window_size = dim_params["window_size"]
                stride = dim_params["stride"]
                dilation = dim_params["dilation"]

                window_size_list = [window_size for _ in range(len(token_layout_list))]
                stride_list = [stride for _ in range(len(token_layout_list))]
                dilation_list = [dilation for _ in range(len(token_layout_list))]
            else:
                raise ValueError(
                    "Sparse parameters for a layer must have key 'window_size' or 'window_size_float', "
                    f"got {dim_params=} in layer index {i}."
                )

            is_causal = dim_params["is_causal"]

            # Create varlen metadata for natten varlen/varsized ops
            # NOTE: generate_multi_dim_varlen_parameters will automatically map window size -1 to
            # full size, that's why constant window sizes aren't allowed.
            # NOTE: if any of the parameters are constant, natten will simplify them
            natten_metadata.append(
                generate_multi_dim_varlen_parameters(
                    token_layout_list=token_layout_list,
                    head_dim=head_dim,
                    device=device,
                    dtype=dtype,
                    requires_grad=requires_grad,
                    #
                    window_size_list=window_size_list,
                    stride_list=stride_list,
                    dilation_list=dilation_list,
                    #
                    is_causal=is_causal,
                )
            )

    return natten_metadata


def generate_temporal_causal_natten_metadata(
    vision_token_shapes: list[tuple[int, int, int]],
    num_action_tokens_per_supertoken: int,
    num_layers: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
) -> list:
    """Generate per-layer varlen metadata for temporal causal attention on supertokens.

    Each sample's generation tokens are laid out as T_i supertokens of size
    S_i = num_action_tokens_per_supertoken + H_i*W_i. Metadata encodes
    is_causal=(True, False): causal across T, full within S. All layers share
    the same metadata (full window, no spatial sparsity).

    Unlike generate_natten_metadata, this function does not apply filter_shape — (T, S) layouts
    are passed directly even when T=1. NATTEN handles T=1 causal masking correctly (trivially
    full attention within S).

    Args:
        vision_token_shapes: List of (T, H, W) per sample.
        num_action_tokens_per_supertoken: Number of action tokens prefixing each
            supertoken (0 when actions are not packed inline).
        num_layers: Number of transformer layers.
        head_dim: Attention head dimension.
        device: Target device.
        dtype: Target dtype.
        requires_grad: Whether metadata tensors require gradient.

    Returns:
        List of length num_layers, each element the same NATTEN varlen metadata dict.
    """
    # T=1: NATTEN requires kernel_size >= 2 and kernel_size <= token_layout, which are mutually
    # exclusive when T=1. Fall back to full dense attention (None) — a single supertoken trivially
    # attends to only itself, so temporal causality is already satisfied.
    # Mixed T=1/T>1 batches are rejected: NATTEN can't mask T=1 samples, and falling back to dense
    # attention for the whole batch would break temporal causality for the T>1 samples.
    # Ensure min_frames >= 5 in the dataloader so that T_latent = 1 + (N-1)//tcf >= 2 always.
    has_short = any(t < 2 for t, h, w in vision_token_shapes)
    if has_short:
        if not all(t < 2 for t, h, w in vision_token_shapes):
            raise ValueError(
                "Mixed T=1 and T>1 samples in causal training batch: NATTEN cannot apply "
                "causal masking when any sample has T=1 (kernel_size constraint), and falling "
                "back to dense attention would break temporal causality for T>1 samples. "
                "Ensure all samples have T_latent >= 2 (set min_frames >= 5 in the dataloader)."
            )
        return [None] * num_layers
    token_layout_list = [(t, num_action_tokens_per_supertoken + h * w) for t, h, w in vision_token_shapes]
    metadata = generate_multi_dim_varlen_parameters(
        token_layout_list=token_layout_list,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        is_causal=(True, False),
    )
    return [metadata] * num_layers
