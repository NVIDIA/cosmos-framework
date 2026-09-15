# ModelOpt state export review

Reviewed on 2026-09-14 against repository HEAD `5a68de3` and the supplied source of truth:

`/home/wkutak/scratch/dev/cosmos/cosmos-genai-nim/wkutak/c3-quantization/cosmos3/pipeline_checkpoints/build_reasoner_modelopt_state.py`

SOT SHA-256: `c68b2648b5ec95a14270c94c81d595fefafc831f9443fe8f15e1a82b3c78ee27`.

## Assessment and changes

The original export had the correct two-file layout: `transformer/modelopt_state.pth`
for Diffusers and root `modelopt_state.pth` for the Transformers reasoner. Its root
metadata construction differed from the SOT: it renamed framework quantizers and
synthesized attention entries by copying a disabled output quantizer. That relies
on the framework and reasoner graphs having exactly the assumed quantizer roles
and attributes. It also mutated the supplied framework metadata in place.

Assembly now writes the reasoner state directly to root `modelopt_state.pth`; it
reads the DiT state from `transformer/modelopt_state.pth`. No temporary
`transformers_modelopt_state.pth` filename or rename is used.

The root writer now consumes the normalized, compressed Diffusers metadata and
captures mode `quantize` from an actual Qwen3-VL graph on the meta device. It registers
placeholder `_amax` metadata only for enabled static input quantizers. Calibrated
values still come from safetensors; weight-quantizer buffers and QTensorWrapper
metadata come from remapped mode `real_quantize`. Generation-only entries are dropped,
and language-model linears without compressed weights remain disabled.

This follows the SOT's graph and buffer contracts. It applies the quantize mode
directly instead of building the graph through an intentionally failing restore.
On the small FP8 comparison fixture, quantize metadata and the complete real-quantize
mode are equal to the supplied SOT's output. Additional explicit disable rules keep
unquantized reasoner linears disabled; the config dictionaries therefore need not be
byte-identical. Source state is deep-copied. Missing reasoner modules, incompatible
weight shapes, missing compression modes, and unsupported root architectures fail
before writing the root state.

The CPU integration test also exposed a loader failure independent of the metadata:
Transformers 4.57 applies only the first matching `key_mapping` rule. A flat attention
key needs both its projection rename and its language-model prefix. Its automatic
base-model detection also misclassifies the flat full-task checkpoint and rejects
`lm_head` as a corrupted state dictionary. The Cosmos shim now composes the mappings
and returns complete task-model names through its 4.57 key-renaming hook.

Assembly previously linked an existing source `hf_quant_config.json` into the output
and then opened that link for writing, modifying the source file. This generated
sidecar is now excluded from linking, and stale output symlinks are unlinked before
writing. Root and transformer ModelOpt files remain separate.

Code: `cosmos_framework/quantization/modelopt_state.py`,
`cosmos_framework/quantization/export.py`, and
`packages/transformers-cosmos3/transformers_cosmos3/model.py`.

## Initial CPU validation (2026-09-14)

Result: **20 tests passed**. Targeted Ruff lint/format checks and `git diff --check` passed.

Environment: Python 3.12.3, PyTorch 2.14.0+cpu, ModelOpt 0.44.0,
Transformers 4.57.6, Accelerate 1.15.0. The isolated environment is at
`/tmp/cosmos-modelopt-review` (temporary, no longer retained); no repository dependency pins were changed.

```bash
LD_LIBRARY_PATH='' \
COSMOS3_REASONER_STATE_SOT=/home/wkutak/scratch/dev/cosmos/cosmos-genai-nim/wkutak/c3-quantization/cosmos3/pipeline_checkpoints/build_reasoner_modelopt_state.py \
PYTHONPATH=packages/transformers-cosmos3 \
/tmp/cosmos-modelopt-review/bin/python -m pytest --noconftest -c /dev/null \
  tests/quantization/modelopt_state_test.py \
  packages/transformers-cosmos3/transformers_cosmos3/model_test.py -q
```

The tests bypass the repository's GPU/training conftest and eager cookbook package
initializer. Torch, ModelOpt compression/restoration, Transformers loading,
safetensors I/O, and the Cosmos shim are real implementations, not mocked loaders.
The external SOT comparison is optional unless `COSMOS3_REASONER_STATE_SOT` is set.

Coverage includes strict restore after torch serialization, exact SOT metadata
comparison, static activation scales, disabled and dynamic input quantizers,
malformed metadata, source-state immutability, local meta-device architecture
construction, sidecar assembly, and actual Hugging Face loading. The Cosmos shim
is tested with both FP8 and BF16 weights, in both flat and nested namespaces;
every loaded tensor is compared exactly with its saved value, and no meta tensors
remain. Existing shim key-mapping tests run alongside these integration tests.

At that stage, GPU generation and full-model export were unverified. The subsequent
Nano GPU validation below covers both. Super and other Transformers/ModelOpt versions
remain unverified. The full repository suite and type check were not run.

ModelOpt 0.44's optional `output_loading_info=True` path fails in its own wrapper
restoration hook because it receives a tuple instead of a model. Tests use the
normal load return and compare every tensor directly; no third-party patch was made.

## Fresh Nano quantization and GPU validation (2026-09-15)

Artifacts and reproduction commands:
[`outputs/nano_fp8_validation_20260915/README.md`](../outputs/nano_fp8_validation_20260915/README.md).
These large local artifacts are ignored by Git.

Quantized the supplied BF16 Nano snapshot
`411f42a8fdfb8c5b2583cb8786e0938f49796eaa` through the repository's
`quantize_fp8_checkpoint` workflow on one H100 NVL. Calibration used four recorded
VideoUFO captions, 480x832, 29 frames, 20 sampling steps, guidance 6, seed 0.
This is a smoke calibration, not a production quality calibration sweep.

The new checkpoint contains 504 FP8 transformer weights. Comparing the **full Nano**
root state to the supplied SOT gave exact equality for mode-0 quantizer metadata
and the complete mode-1 compressed-weight metadata/configuration. All 1,008 stored
input/weight scale pairs equal their corresponding FP32 `_amax / 448` values.

The GPU run exposed two additional loader issues:

1. The Transformers shim inherited `Qwen3VLConfig`. For a `cosmos3_omni` root whose
   nested vision config is tagged `qwen3_vl`, Transformers 4.57 selects that nested
   dictionary as the entire config, losing text dimensions and RoPE settings.
   The shim now sets `config_class = Cosmos3OmniConfig`. The integration matrix
   covers both Cosmos and Qwen root tags, flat/nested weights, and BF16/FP8.
2. Diffusers 0.39's native ModelOpt tensor loader assigned quantizer buffers into
   `_parameters`, leaving the actual `_buffers` on meta or invalid, and replaced
   QTensorWrapper objects. It also cast calibrated FP32 scales to the requested
   BF16 dtype before assignment. The opt-in helper in
   `cosmos_framework/inference/modelopt_diffusers.py` preserves buffer placement,
   wrapper metadata, and FP32 quantizer scales. Its real Diffusers reload test
   checks every saved tensor exactly and verifies no meta tensors remain.

Use the helper before loading with native Diffusers 0.39:

```python
from cosmos_framework.inference.modelopt_diffusers import enable_modelopt_diffusers_checkpointing
from diffusers import Cosmos3OmniPipeline, Cosmos3OmniTransformer
from modelopt.torch.quantization.backends.gemm_registry import enable_real_quant_gemm
import modelopt.torch.quantization.backends.fp8_per_tensor_gemm  # register FP8 GEMM
import torch

enable_modelopt_diffusers_checkpointing()
transformer = Cosmos3OmniTransformer.from_pretrained(
    f"{checkpoint}/transformer", torch_dtype=torch.bfloat16, local_files_only=True
)
pipeline = Cosmos3OmniPipeline.from_pretrained(
    checkpoint, transformer=transformer, torch_dtype=torch.bfloat16, local_files_only=True
).to("cuda")
enable_real_quant_gemm(transformer)
```

The explicit component path makes ModelOpt restore the component state. The helper
is optional and imports heavy dependencies only when called. It changes native
Diffusers loader methods process-wide after opt-in; compatibility is tested with
Diffusers 0.39 / ModelOpt 0.44, not other versions. The framework's separate TorchAO
inference path does not use this helper.

Final GPU evidence: Diffusers generated a 512x512 image and a 29-frame 832x480 video
with 504 FP8 GEMMs selected and no selected-kernel fallbacks. Transformers restored
the root state, loaded 252 FP8 weights, selected 252 FP8 GEMMs, and captioned the
generated image coherently. Both loaders left zero meta tensors. A same-prompt,
same-seed BF16 image is saved alongside the FP8 outputs. The MP4 decodes correctly
and has changing frames; the short clip shows reaching, not a completed pick/place.

Final targeted suite: **25 passed**, including external-SOT comparison and real
Diffusers reload. Targeted Ruff lint/format checks and `git diff --check` passed.
Runtime versions: Python 3.12.3, NGC PyTorch 2.10 development build, ModelOpt 0.44.0,
Transformers 4.57.6, Diffusers 0.39.0. Exact package versions, image ID, prompts,
settings, source snapshot, logs, and scripts are retained with the artifacts.
