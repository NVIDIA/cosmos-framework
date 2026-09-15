# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Compatibility for loading serialized ModelOpt graphs with native Diffusers.

Diffusers 0.39's ModelOpt quantizer assigns every checkpoint tensor to
``module._parameters``. That leaves quantizer buffers on meta and discards
QTensorWrapper metadata. Enable this alongside ModelOpt HF checkpointing before
loading a pre-quantized Diffusers model.
"""


def enable_modelopt_diffusers_checkpointing() -> None:
    """Enable ModelOpt restoration, preserving quantizer buffers and FP8 wrappers."""
    import modelopt.torch.opt as mto
    from diffusers.quantizers.modelopt.modelopt_quantizer import NVIDIAModelOptQuantizer
    from diffusers.utils import get_module_from_name
    from modelopt.torch.quantization.qtensor import QTensorWrapper

    mto.enable_huggingface_checkpointing()
    original = NVIDIAModelOptQuantizer.create_quantized_param
    if getattr(original, "_cosmos_preserves_modelopt_state", False):
        return
    original_preprocess = NVIDIAModelOptQuantizer._process_model_before_weight_loading

    def preprocess(self, model, device_map, keep_in_fp32_modules=None, **kwargs):
        if keep_in_fp32_modules is None:
            keep_in_fp32_modules = []
        if self.pre_quantized:
            # Diffusers casts tensors before create_quantized_param is called.
            # Extend the shared loader list so calibrated scales retain FP32.
            for role in ("input_quantizer", "weight_quantizer"):
                if role not in keep_in_fp32_modules:
                    keep_in_fp32_modules.append(role)
        return original_preprocess(self, model, device_map, keep_in_fp32_modules, **kwargs)

    def create_quantized_param(self, model, param_value, param_name, target_device, *args, **kwargs):
        if self.pre_quantized:
            module, tensor_name = get_module_from_name(model, param_name)
            if tensor_name in module._buffers:
                module._buffers[tensor_name] = param_value.to(device=target_device)
                return
            parameter = module._parameters.get(tensor_name)
            if isinstance(parameter, QTensorWrapper):
                module._parameters[tensor_name] = QTensorWrapper(
                    qtensor=param_value.to(device=target_device),
                    metadata=parameter.get_state()["metadata"],
                )
                return
        return original(self, model, param_value, param_name, target_device, *args, **kwargs)

    create_quantized_param._cosmos_preserves_modelopt_state = True
    NVIDIAModelOptQuantizer._process_model_before_weight_loading = preprocess
    NVIDIAModelOptQuantizer.create_quantized_param = create_quantized_param
