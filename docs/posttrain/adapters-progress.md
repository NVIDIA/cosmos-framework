# Posttrain adapter implementation qualification

Implementation branch: `feat/posttrain-vlm-adapters`, based on main
`7738e9e7969e30160163b9d7941782fefd6eaa5c`. The audited baseline was
`1a2f03ab6111b457633827736f94f67d55bd5202`; intervening main changes affect
quantization/docs, not the SFT builder or native adapters. Existing LoRA/rsLoRA
and pure LoRA merge mathematics are reused from `utils/generator/lora.py`.
Previous WTS/AutoML results used unchanged framework code and do not qualify this implementation.

| Requirement | Code / tests | Status |
|---|---|---|
| Component selectors, independent settings, aliases/ties | `posttrain/config/adapters.py`, `posttrain/model/adapters.py`, `tests/test_adapters.py` | Implemented and CPU validated (selectors, independent settings, aliases, counts and optimizer membership) |
| LoRA / rsLoRA | Native `LoraInjectedLinear` reused; posttrain builder integration | Implemented; CPU validated; all 35 real-model training/continuation cases passed, FP32 export/reload matrix running |
| DoRA magnitude, optimization and pure merge | `PosttrainAdapterLinear`, `optimizer_groups` | Implemented; CPU reference gradients/reload/merge validated; all 35 real-model training/continuation cases passed, FP32 export/reload matrix running |
| PiSSA residual contract and saved initial factors | `PosttrainAdapterLinear.initial_A/initial_B`; portable format v1 | Implemented; CPU principal components/recovery/merge validated; all 35 real-model training/continuation cases passed, FP32 export/reload matrix running |
| Fresh, warm start and exact resume lifecycle | `Qwen3VLBuildHooks.after_load_before_parallelize`, SFT `build_trainer` | Implemented and CPU validated: bitwise controlled continuation through trainer and full SFT recipe |
| Portable export / strict reload | `posttrain/checkpoint/adapters.py` | Implemented and CPU validated in fresh processes |
| Merged HF export | Existing HF materializer reused; bundled processor and single-shard handling extended | Implemented and CPU validated; FP32 export implemented; real-model BF16 export failed and is excluded (see below) |
| Vision gradients, media and activation checkpointing | Native tiny Qwen3-VL plus real PNG/MP4 through SFT packer, with/without AC | Implemented and CPU validated; two-GPU first/final vision cases now pass FP32 export and fresh reload with AC on/off; broader matrix running |
| FSDP2 and mixed modality ranks | FP32 adapter storage/accumulation policy added | All 35 training/continuation cases passed, including two-rank mixed modalities and AC on/off; FP32 export/reload matrix running |
| Wheel/container revision provenance | Sealed wheel and byte-by-byte installed-package attestation | Implemented and validated: all 1,838 package files match; 184 tests passed from installed f29da1c outside the checkout |
| Full WTS epoch and held-out evaluation | Existing experiments preserved | Pending feature qualification; no new HPO |
| Recipes, documentation and PR | Runnable native adapter recipe, WTS converter, adapter guide; [draft PR #342](https://github.com/NVIDIA/cosmos-framework/pull/342) | Implemented; GPU evidence pending |

Initial execution profile: dense native Qwen3-VL / staged Cosmos3-Nano-VLM,
BF16 frozen base, FP32 adapters, AdamW, eager/SDPA, non-reentrant full activation
checkpointing, single GPU then pure FSDP2. Generic HF, native MoE, HSDP, tensor/
expert/context parallelism, compilation, quantization and adapter stacking are
excluded until separately qualified. WTS has 5,555 training rows; the full epoch
will use one GPU so the existing exact DP cursor needs no duplicated/dropped rows.

CPU validation: 265 tests passed, 1 optional test skipped in the focused regression
suite; 2 additional DoRA-reference/PiSSA-principal-component tests passed. Environment:
Python 3.12.3, torch 2.10.0+cpu, Transformers 5.8.1, TorchCodec 0.10.0,
safetensors 0.8.0; PEFT 0.21.2 is a test reference only, not the implementation.
The SFT test root path was corrected after the posttrain directory migration.
The final adapter-disabled regression run on `f29da1c` passed 205 tests, with two
optional tests skipped, in 22.47 seconds. This covers SFT configuration, entrypoint,
objective, packer, trainer validation, native Qwen3-VL loading and HF export.

Additional correctness coverage checks separately trained FP32 normalization with
BF16 activations, and native/portable/HF preservation of declared embedding ties.
The latter exposed two compatibility defects: the native Transformers 4 tie-key
list needed the Transformers 5 mapping form, and HF's `assign=True` overlay needed
to restore tying before serialization. The converter format advances to version 5.
The installed-wheel check runs outside the checkout: `.gitignore` previously
caused Hatch to omit the tracked `posttrain/roles/env` runtime package.

Installed revisions and recovery evidence:

- `3208d7b`: installed-wheel job `37627771` passed 44 tests. Its first GPU
  attempt stopped before an update because the harness omitted validation reasons;
  the harness regression test now covers the complete path.
- `15800a9`: installed-wheel job `37628309` passed 45 tests. Its 20 GPU
  cases reached portable export after training/continuation checks, then hit
  Lustre `EOPNOTSUPP` for `RENAME_NOREPLACE`.
- `3a58c98e4cafe24c73c4392571953c855b69833c`: installed job `37629424`
  passed 51 tests and the real Lustre immutable-publication probe. Wheel SHA-256
  `88eeb16b59156577ade24b60f5a324bd489f3692f66a05445d3c1813521265fb`;
  container SHA-256 `64269714b74a6ee720b081cb24be2e3dbb3911fee0acef675dcc03d88b66838e`.
  Training gates `37629559` (20 method/world/AC cases), `37629560` (12 isolated
  component cases and the full-WTS media/accumulation profile), and `37629971`
  (two first/final-vision cases) exercise actual text, PNG, MP4 and WTS inputs.
  Completed training cases have zero controlled next-update parameter error;
  qualification remains incomplete until every required export/reload gate passes.
- `c8885824e82dfe9df924eb3b8738536aff13ba96`: installed job `37631928`
  passed 53 tests in 79.27 seconds, with all 1,838 package files byte-attested.
  Wheel SHA-256 `c87180bf3f989ac30d6d767f68d2e8bd72a1a64575a176ccc303e8a355216755`;
  container SHA-256 `c882fe7defd72a1974fbc2fdfc0a979db1b1ba584330525f09ea1336beb5ee2b`.
  This revision changed verification, not the training/export runtime.

GPU portable reload is bitwise exact in completed cases. BF16 merged deployment
failed its declared output bounds. Diagnostic `37630400` isolated BF16 adapter
versus dense rounding; `37630661` verified all 750 serialized tensors and FP32
merge algebra (maximum HF error `3.16e-4`, 32/32 argmax agreement). Native and HF
also use different vision GEMM batch shapes for combined image/video inputs.
Protocol v2 (`37632021`, `37632023`, `37632027`) retained exact tensor and FP32
controls, but BF16 probability-drift gates still failed: worst-position total
variation `.1115` for all-component LoRA, `.1356` for first/final vision, and
`.0475` original-probability regret for a changed rsLoRA greedy token. The bounds
were not widened to count these as passes.

The implementation now exports FP32 merged models, preserving the already-rounded
training-base values before promotion. The existing materializer records precision
in its identity, verifies every floating shard and writes matching Transformers
4/5 config precision metadata. Strict FP32 native/HF equivalence and a fresh
`dtype="auto"` load are required. BF16 merged inference is excluded; portable
reload remains the exact training-precision path. Installed-package job `37633097` passed all 184 adapter/export tests in 123.99
seconds for `f29da1ccf4f201e1ed73c2c26800908c392256d1`. Wheel SHA-256
`66b66fad810a2b998b5c8347bb6c8528c5842a43aae5c10205a758776390f302`;
container SHA-256 `6861377c581a16585d6c5510d6e2064cb9709589c90d6d7efe130ef1691ab574`.
FP32 export jobs `37633099` and `37633101` preserved 15 completed cases before
their unfinished work was redistributed across `37636040`, `37636045`,
`37636047` and `37636052`; the two interrupted attempts were archived. First/final vision
job `37633104` completed both two-rank AC-on/off cases, with exact portable reload,
750 bitwise-correct FP32 serialized tensors, 64/64 argmax agreement and maximum
HF logit error `1.74e-4`. All-component single-GPU LoRA and language-only LoRA
also passed the new export checks. Single-GPU all-component rsLoRA, DoRA and
PiSSA have also passed; the complete matrix is still pending. Full WTS training
remains gated; no new HPO.

All 35 training/continuation cases now pass: 53 rank reports contain 386 finite,
applied update reports. Every intended trainable tensor changed over multiple
updates; initialization and controlled resumed parameter values have zero maximum
absolute error. Export/reload qualification remains pending for the full matrix.

The final artifact audit found that merged exports verified the portable identity
at conversion time but did not retain it in the committed marker. The exporter
now records the full adapter/processor/media identity and adapter weight digest
inside the materializer's transaction identity. Missing or conflicting metadata
refuses reuse. The expanded adapter/export suite passes 189 tests locally; the
installed build and real-model metadata-contract gate are pending. The full-WTS
launcher is paused until this additional artifact check passes. Numerical merge
arithmetic and training/portable runtime are unchanged.
