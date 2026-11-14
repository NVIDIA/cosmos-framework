# Experimental stock Qwen3-VL projection profile

`qwen3_vl_fp32_fixed_mma_v1` is an opt-in inference candidate for the existing
`Qwen/Qwen3-VL-2B-Instruct` binding at revision
`89644892e4d85e24eaac8bacfd4f463576704203`. It has no real-model numerical or task
qualification yet. The original stock FP32 cache and split-prefix failures
remain failures under their original source and profile identities.

The preserved three-loader, four-video trace localized the first disagreements
to language projections operating on identical selected inputs and weights.
The independently checked 24 operator pairs reproduce native shape-dependent
FP32 reductions. This candidate reuses the existing fixed A100 FP64 MMA kernel
inside each language projection and the tied output head, then rounds each
projection separately to FP32. It preserves the intermediate A projection,
LoRA/rsLoRA scaling, DoRA direction norm and unscaled bias, and the separate
trained and saved-initial PiSSA branches. It creates no merged-weight cache.

Native normalization, attention, MRoPE, vision, processor and modality assembly
remain unchanged. Portable loading retains the exact recorded ordered modality
forward and extra-normalization wrappers. Base and merged loading retain their
stock HF assembly. Consequently, an isolated projection pass would not establish
whole-model cache consistency or portable/merged equivalence.

The public boundaries are:

| Route | Required identity |
| --- | --- |
| Portable | Existing v3 `qwen3_vl_hf_stock_v1` artifact, every component `factorized_v1`, recorded BF16 base and FP32 factors/extras; all six method/scaling settings remain distinct. |
| Merged | Complete FP32 export tagged with this profile, including its original adapter identity, exact tensor inventory/digests and unchanged processor assets. |
| Unadapted base | The exact immutable base and processor, with a distinct `base_deployment` identity; no adapter metadata is fabricated. |

All routes require Transformers 5.8.1, PyTorch 2.10.0, Triton 3.6.0, one A100
SM80, batch one, eager evaluation with gradients disabled, native SDPA, and no
TF32, autocast, graph capture, compilation or offload. Construct the runtime
outside `torch.inference_mode()` so tensor version counters remain available.
Recorded training dropout remains in the artifact and is inactive only because
every module is in evaluation mode. The separate stable FP64 branch profile
and its zero-dropout admission are unchanged.

Use the [portable example](../../examples/reasoner/portable_fixed_projections.toml)
with the existing public generation or evaluation entry point. For a tagged
merged export, run:

```bash
python -m cosmos_framework.posttrain.checkpoint.tools.adapters \
  --merged \
  --adapter /adapters/stock-factorized/portable \
  --base /models/Qwen3-VL-2B-Instruct/89644892e4d85e24eaac8bacfd4f463576704203 \
  --out /exports/qwen3-vl-fixed-projections \
  --output-dtype float32 --execution-profile qwen3_vl_fp32_fixed_mma_v1
```

Then omit `portable_adapter` and point `model_name` at that export. For the
unadapted control, use the original base as `model_name`, omit
`portable_adapter`, and set
`model.base_execution_profile = "qwen3_vl_fp32_fixed_mma_v1"`. This route freezes
all base parameters and reports `base_deployment`; portable loading preserves
the artifact's original trainability flags. The public generator and evaluator
record the actual deployment identity and generation/preprocessing settings.

Ordinary `DynamicCache` prefixes belong to one profile installation. Forks
retain that ownership; a foreign or earlier-installation prefix is rejected
before mutation. Tensor, callable, norm/attention settings and configuration
changes invalidate the installed execution. Call `runtime.clear_adapter_caches()`
after untracked `.data` writes; this invalidates the profile until an immutable
reload, including directly held external prefixes. `restore_inference_execution()`
restores the original FP32 forwards while preserving tensors and the original
modality wrappers, and discards runtime-owned caches.

The next qualification is fixed before candidate outputs: the 12 saved operator
witnesses first, then all three public loaders on the original four WTS videos
and 32 common-history decisions, repeated prefixes, and all 17 own-artifact
text/image/video reference inputs for each adapted loader. Original numerical
and greedy gates remain unchanged. The bounded allocation is one 80-GB A100,
96 GiB host memory and four hours; startup/compilation, each workload and peak
memory must be measured. Full 2,676-question evaluation and the 576-question
scene-disjoint subset remain separate requirements for each final scheduled
checkpoint. No training schedule is repeated to test this inference candidate.
