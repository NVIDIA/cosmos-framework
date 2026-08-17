# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Nemotron Parakeet encoder and audio projection components."""

from collections.abc import Iterable
from numbers import Integral

import torch
import transformers
from torch import nn
from transformers import ParakeetEncoder, ParakeetEncoderConfig

from cosmos_framework.model.generator.reasoner.audio.projector import AudioProjector
from cosmos_framework.model.generator.reasoner.parakeet.configuration_parakeet import (
    ParakeetAudioConfig,
    get_nemotron_parakeet_config,
)


def _validate_subsampling_factor(subsampling_factor: object) -> int:
    """Return a valid power-of-two factor before Transformers builds its stack."""
    if (
        isinstance(subsampling_factor, bool)
        or not isinstance(subsampling_factor, Integral)
        or subsampling_factor < 2
        or (int(subsampling_factor) & (int(subsampling_factor) - 1)) != 0
    ):
        raise ValueError(
            "config.subsampling_factor must be an integer power of two greater than or equal to 2, "
            f"got {subsampling_factor!r}"
        )
    return int(subsampling_factor)


def get_subsampled_lengths(input_lengths: torch.Tensor, config: ParakeetEncoderConfig) -> torch.Tensor:
    """Compute valid output lengths after Parakeet's strided convolution stack."""
    if input_lengths.ndim != 1:
        raise ValueError(f"input_lengths must have shape [batch], got {tuple(input_lengths.shape)}")
    if input_lengths.dtype == torch.bool or input_lengths.dtype.is_floating_point or input_lengths.dtype.is_complex:
        raise TypeError(f"input_lengths must use an integer dtype, got {input_lengths.dtype}")

    subsampling_factor = _validate_subsampling_factor(config.subsampling_factor)

    kernel_size = config.subsampling_conv_kernel_size
    stride = config.subsampling_conv_stride
    padding = (kernel_size - 1) // 2
    num_layers = subsampling_factor.bit_length() - 1
    output_lengths = input_lengths
    for _ in range(num_layers):
        output_lengths = (
            torch.div(
                output_lengths + 2 * padding - kernel_size,
                stride,
                rounding_mode="floor",
            )
            + 1
        )
    return output_lengths.to(dtype=torch.int)


def _require_transformers_attribute(instance: object, attribute: str, path: str) -> object:
    if not hasattr(instance, attribute):
        raise RuntimeError(
            "NemotronParakeetEncoder is incompatible with the installed Transformers "
            f"{transformers.__version__}: missing required attribute {path!r} while removing "
            "convolution biases for a bias-free checkpoint."
        )
    return getattr(instance, attribute)


class NemotronParakeetEncoder(nn.Module):
    """Thin wrapper around the Transformers Parakeet FastConformer encoder.

    The module accepts normalized 128-bin log-mel features rather than raw
    waveforms. Feature extraction, clip segmentation, and projection into a
    language model's hidden size intentionally remain outside the encoder.
    """

    config: ParakeetEncoderConfig
    encoder: ParakeetEncoder

    def __init__(self, config: ParakeetEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else get_nemotron_parakeet_config()
        _validate_subsampling_factor(self.config.subsampling_factor)
        self.encoder = ParakeetEncoder(self.config)

        if not getattr(self.config, "convolution_bias", True):
            self._remove_convolution_biases()

    def _remove_convolution_biases(self) -> None:
        """Match bias-free Nemotron weights on Transformers 4.57.x.

        Transformers 4.57.x creates three Conv1d biases per Conformer layer
        even when ``convolution_bias=False``. Removing those parameters makes
        the module match the released Nemotron checkpoint and prevents the
        zero-initialized compatibility parameters from drifting in training.
        Every expected module path is validated before mutation so an upstream
        object-graph change fails with an actionable error instead of silently
        changing checkpoint keys.
        """
        layers = _require_transformers_attribute(self.encoder, "layers", "encoder.layers")
        if not isinstance(layers, Iterable):
            raise RuntimeError(
                "NemotronParakeetEncoder is incompatible with the installed Transformers "
                f"{transformers.__version__}: required attribute 'encoder.layers' is not iterable while removing "
                "convolution biases for a bias-free checkpoint."
            )

        convolution_names = ("pointwise_conv1", "depthwise_conv", "pointwise_conv2")
        convolution_modules: list[nn.Module] = []
        for layer_index, layer in enumerate(layers):
            layer_path = f"encoder.layers[{layer_index}]"
            convolution = _require_transformers_attribute(layer, "conv", f"{layer_path}.conv")
            for convolution_name in convolution_names:
                convolution_path = f"{layer_path}.conv.{convolution_name}"
                convolution_module = _require_transformers_attribute(
                    convolution,
                    convolution_name,
                    convolution_path,
                )
                _require_transformers_attribute(convolution_module, "bias", f"{convolution_path}.bias")
                if not isinstance(convolution_module, nn.Module):
                    raise RuntimeError(
                        "NemotronParakeetEncoder is incompatible with the installed Transformers "
                        f"{transformers.__version__}: required attribute {convolution_path!r} is not a torch module "
                        "while removing convolution biases for a bias-free checkpoint."
                    )
                convolution_modules.append(convolution_module)

        for convolution_module in convolution_modules:
            convolution_module.register_parameter("bias", None)

    def get_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Return the valid encoded length for each input feature sequence."""
        return get_subsampled_lengths(input_lengths, self.config)

    def forward(
        self,
        input_features: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode padded log-mel features and return embeddings plus valid lengths.

        Args:
            input_features: Normalized log-mel features with shape
                ``[batch, time, num_mel_bins]``.
            input_lengths: Number of valid feature frames per sample with
                shape ``[batch]``. These are encoder attention lengths for
                already prepared mel tensors; raw-audio frontends must convert
                their own framing convention before calling this module.

        Returns:
            Encoded features with shape ``[batch, encoded_time, hidden_size]``
            and the valid encoded length of each sample.
        """
        if input_features.ndim != 3:
            raise ValueError(
                f"input_features must have shape [batch, time, num_mel_bins], got {tuple(input_features.shape)}"
            )
        if input_features.shape[-1] != self.config.num_mel_bins:
            raise ValueError(
                f"input_features must have {self.config.num_mel_bins} mel bins, got {input_features.shape[-1]}"
            )
        if input_lengths.ndim != 1 or input_lengths.shape[0] != input_features.shape[0]:
            raise ValueError(
                f"input_lengths must have shape [{input_features.shape[0]}], got {tuple(input_lengths.shape)}"
            )
        if input_lengths.dtype == torch.bool or input_lengths.dtype.is_floating_point or input_lengths.dtype.is_complex:
            raise TypeError(f"input_lengths must use an integer dtype, got {input_lengths.dtype}")
        if torch.any(input_lengths < 1) or torch.any(input_lengths > input_features.shape[1]):
            raise ValueError(
                f"Every input length must be between 1 and the padded feature time dimension {input_features.shape[1]}"
            )
        input_lengths = input_lengths.to(device=input_features.device)
        frame_indices = torch.arange(input_features.shape[1], device=input_features.device)
        attention_mask = frame_indices.unsqueeze(0) < input_lengths.unsqueeze(1)
        outputs = self.encoder(input_features=input_features, attention_mask=attention_mask)
        return outputs.last_hidden_state, self.get_output_lengths(input_lengths)


class ParakeetAudioProjector(AudioProjector):
    """Project Parakeet features into a reasoner's hidden space.

    Architecture: LayerNorm -> Linear -> GELU -> Linear, matching the existing
    Qwen-VL vision merger. The input, intermediate, and output dimensions are
    configurable through :class:`ParakeetAudioConfig`.
    """

    def __init__(self, config: ParakeetAudioConfig) -> None:
        super().__init__(
            input_hidden_size=config.encoder_config.hidden_size,
            projection_hidden_size=config.projection_hidden_size,
            out_hidden_size=config.out_hidden_size,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim not in (2, 3):
            raise ValueError(
                "hidden_states must have shape [num_tokens, hidden_size] or "
                f"[batch, time, hidden_size], got {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[-1] != self.input_hidden_size:
            raise ValueError(
                f"hidden_states must have hidden size {self.input_hidden_size}, got {hidden_states.shape[-1]}"
            )
        return super().forward(hidden_states)


class ParakeetAudioModel(nn.Module):
    """Composable Parakeet encoder and reasoner projector."""

    def __init__(
        self,
        config: ParakeetAudioConfig | None = None,
        encoder: NemotronParakeetEncoder | None = None,
        projector: ParakeetAudioProjector | None = None,
    ) -> None:
        super().__init__()
        self.config = config if config is not None else ParakeetAudioConfig()
        self.encoder = encoder if encoder is not None else NemotronParakeetEncoder(self.config.encoder_config)
        self.projector = projector if projector is not None else ParakeetAudioProjector(self.config)

        encoder_hidden_size = self.encoder.config.hidden_size
        if encoder_hidden_size != self.projector.input_hidden_size:
            raise ValueError(
                "encoder and projector hidden sizes must match, got "
                f"{encoder_hidden_size} and {self.projector.input_hidden_size}"
            )

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        """Initialize the fresh projector after meta materialization."""
        self.projector.reset_parameters()

    def forward(
        self,
        audio_features: torch.Tensor,
        audio_feature_lengths: torch.Tensor,
        audio_token_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Consume processor outputs and return projected audio embeddings.

        The argument names intentionally match ``ParakeetAudioProcessor`` so
        callers can pass its tensor dictionary directly with ``model(**batch)``.
        Processor ``audio_token_lengths`` are required because Nemotron keeps
        the encoded position corresponding to the masked center-padding frame;
        those lengths are checked against the encoder's padded output.
        """
        encoded_features, output_lengths = self.encoder(audio_features, audio_feature_lengths)
        if audio_token_lengths.ndim != 1 or audio_token_lengths.shape != output_lengths.shape:
            raise ValueError(
                "audio_token_lengths must match the encoder output-length shape; got "
                f"{tuple(audio_token_lengths.shape)} and {tuple(output_lengths.shape)}"
            )
        if (
            audio_token_lengths.dtype == torch.bool
            or audio_token_lengths.dtype.is_floating_point
            or audio_token_lengths.dtype.is_complex
        ):
            raise TypeError(f"audio_token_lengths must use an integer dtype, got {audio_token_lengths.dtype}")
        expected_lengths = self.encoder.get_output_lengths(audio_feature_lengths.to(output_lengths.device) + 1)
        processor_lengths = audio_token_lengths.to(device=output_lengths.device, dtype=output_lengths.dtype)
        if not torch.equal(processor_lengths, expected_lengths):
            raise ValueError(
                "audio_token_lengths do not match encoder output lengths: "
                f"processor={processor_lengths.tolist()}, encoder={expected_lengths.tolist()}"
            )
        if torch.any(expected_lengths > encoded_features.shape[1]):
            raise ValueError(
                "audio_token_lengths exceed the padded encoder output time dimension: "
                f"lengths={expected_lengths.tolist()}, time={encoded_features.shape[1]}"
            )
        output_lengths = expected_lengths
        return self.projector(encoded_features), output_lengths


__all__ = [
    "NemotronParakeetEncoder",
    "ParakeetAudioModel",
    "ParakeetAudioProjector",
    "get_nemotron_parakeet_config",
    "get_subsampled_lengths",
]
