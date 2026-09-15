# ModelOpt export: filename check and change summary

Verified on 2026-09-15 against the repository lock and the packages used for the
completed Nano GPU validation. This follow-up changed documentation only; it did
not rerun quantization or inference.

## Filename expected by the pinned version

- The repository permits Transformers `>=4.57.1,<5.0.0`
  ([pyproject.toml](../pyproject.toml)), and `uv.lock` resolves **4.57.6**.
- The validation environment used that exact Transformers version and
  **ModelOpt 0.44.0**. ModelOpt is recorded in the validation environment, not pinned
  in the repository's `uv.lock`.
- ModelOpt owns this sidecar lookup. Its installed Hugging Face plugin defines
  `_MODELOPT_STATE_SAVE_NAME = "modelopt_state.pth"` and joins that name to the
  directory supplied to `from_pretrained`. Its Transformers plugin calls the same
  lookup for graph restoration and FP8 wrapper restoration. Neither uses
  `transformers_modelopt_state.pth`.

Inspected package source is retained locally:
[shared filename lookup](../outputs/nano_fp8_validation_20260915/filename_verification/modelopt_044_huggingface.py)
(lines 36 and 61–69), and
[Transformers integration](../outputs/nano_fp8_validation_20260915/filename_verification/modelopt_044_transformers.py)
(lines 67–110). The successful
[reasoner log](../outputs/nano_fp8_validation_20260915/reasoner_fp8.log)
also explicitly reports restoration from root `modelopt_state.pth` at line 18.

The original code wrote `transformers_modelopt_state.pth` temporarily under the
staged transformer directory, then moved it to root `modelopt_state.pth` during
assembly. **That intermediate filename was not itself a loader bug.** Writing
the reasoner state directly at the final root path removes an unnecessary staging
step. `transformer/modelopt_state.pth` remains a different graph for Diffusers.

## Changes and confirmed bugs

| Area | Finding | Final change |
| --- | --- | --- |
| Root metadata construction | Manually synthesized reasoner quantizer entries from the framework graph; brittle relative to the SOT. Also mutated its input metadata in place. | Build the actual Qwen3-VL graph on meta, capture its quantizer metadata, remap compressed DiT weight metadata, and deep-copy the input. Validate module names/shapes and preserve disabled layers. |
| File assembly | Could write through an output symlink to the source `hf_quant_config.json`. | Exclude generated sidecars from source linking and unlink stale generated-sidecar symlinks before writing. |
| Transformers key loading | Transformers 4.57 stopped after the first regex rename; flat attention keys needed both projection and namespace renames. Its base-model heuristic also rejected the full flat checkpoint's `lm_head`. | Compose renames and return complete task-model names in the Cosmos shim. |
| Transformers config loading | Inherited Qwen config loader selected Nano's nested `vision_config` as the whole config, losing text dimensions/RoPE and crashing on `rope_scaling=None`. | Set the shim's `config_class` to `Cosmos3OmniConfig`. |
| Native Diffusers 0.39 loading | Put quantizer buffers in `_parameters`, leaving actual buffers on meta/invalid; replaced FP8 wrappers; downcast FP32 scales during loading. | Add an opt-in inference compatibility helper that preserves buffer placement, QTensorWrapper metadata, and FP32 quantizer scales. |

Code: [export and assembly](../cosmos_framework/quantization/export.py),
[reasoner metadata builder](../cosmos_framework/quantization/modelopt_state.py),
[Transformers shim](../packages/transformers-cosmos3/transformers_cosmos3/model.py),
[Diffusers compatibility helper](../cosmos_framework/inference/modelopt_diffusers.py).

Validation-script issues were also resolved: container dependencies/user/cache
permissions and a saver that incorrectly treated Diffusers' flat list of frames as
a nested batch. These were environment/harness issues, not checkpoint corruption.

Two investigation corrections: a `softmax_quantizer` entry is legitimate in the
actual ModelOpt Qwen graph; its presence alone was not a bug. Also, the claim that
the earlier Diffusers run necessarily used a dequantization fallback was incorrect:
compression restoration already enables native GEMMs for this state. Explicit
execution audits were added; the final run selected all 504 native FP8 GEMMs.

## Validation and scope

Freshly quantized the supplied BF16 Nano using four recorded calibration captions.
Full Nano quantizer metadata and the complete real-quantize mode match the SOT.
All 1,008 input/weight scale pairs were checked. Native Diffusers generated an image
and video with 504 FP8 GEMMs; Transformers captioned the image with 252 FP8 GEMMs.
Both had zero meta tensors and zero selected-kernel fallbacks. **25 targeted tests
passed**, plus targeted lint/format and diff checks.

The compatibility helper is opt-in and tested with Diffusers 0.39 / ModelOpt 0.44;
no dependency pins were changed. This is a calibration/inference smoke validation,
not a production quality or performance certification. The FP8 image used less
memory but was slower than BF16 in this runtime.

[Saved results and reproduction commands](../outputs/nano_fp8_validation_20260915/README.md)
and [detailed review](quantization_modelopt_state.md).
