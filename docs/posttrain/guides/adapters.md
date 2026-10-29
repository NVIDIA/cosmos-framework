# Native Qwen3-VL adapter SFT

The posttrain SFT workflow can adapt dense native Qwen3-VL, including
Cosmos3-Nano-VLM, with language, final/deepstack merger and selected vision-block
adapters. It reuses `LLMTrainer`, the conversation packer, data cursor and checkpoint
transactions. The native TOML workflow already supported LoRA/rsLoRA before this
contribution; this is an extension of the separate posttrain workflow.

See the [qualification record](../adapters-progress.md) for what has actually
been tested. CPU correctness is not real-model GPU qualification. Four-bit
training, generic HF wrappers, native MoE/expert tensors, HSDP, TP/EP/CP/PP,
compilation, stacked adapters and arbitrary linear/embedding `modules_to_save`
are outside this execution profile. Plain native `nn.Linear` targets and optional
native normalization modules are supported. Fused vision QKV is one whole
linear target, not independently selectable Q/K/V slices.

## Recipe and component selection

Use the posttrain SFT environment from [the SFT guide](sft.md), including
Transformers 5.8.1. Set `SFT_MODEL_PATH` to a complete immutable local snapshot
(config, safetensors, tokenizer, processor and template). The initial recipe
uses Cosmos3-Nano-VLM's last vision block, index 26; other dense Qwen3-VL
checkpoints must select indices that actually exist.

```bash
export SFT_MODEL_PATH=/models/Cosmos3-Nano-VLM
export SFT_DATA_ROOT=/data/adapter-example
python -m cosmos_framework.posttrain.scripts.prepare_sft_example --output "$SFT_DATA_ROOT"
export SFT_ADAPTER_METHOD=lora   # lora, dora, pissa
export SFT_ADAPTER_RSLORA=0      # 1 uses alpha/sqrt(rank), including with DoRA/PiSSA
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  -m cosmos_framework.posttrain.entrypoints.sft \
  --config=cosmos_framework/posttrain/configs/sft_cosmos3_nano_adapters.py
```

The recipe sets `rl.adapters` using the existing dataclass/OmegaConf conventions.
An independent-settings example in a Python recipe is:

```python
from cosmos_framework.posttrain.config.adapters import AdapterConfig, AdapterComponentConfig, AdapterExtraModuleConfig

config.rl.adapters = AdapterConfig(components=[
    AdapterComponentConfig(component="language", rank=8, alpha=16, dropout=0.05,
        targets=["model.language_model.layers.*.self_attn.q_proj",
                 "model.language_model.layers.*.self_attn.v_proj"]),
    AdapterComponentConfig(component="projector", method="dora", rank=4, alpha=8,
        lr_multiplier=0.5, targets=["model.visual.merger.linear_fc2",
                                  "model.visual.deepstack_merger_list.*.linear_fc2"]),
    AdapterComponentConfig(component="vision", method="pissa", rank=4, alpha=8,
        initialization="svd_lowrank", pissa_niter=4, dropout=0, layers=[25, 26],
        lr_multiplier=0.1, targets=["model.visual.blocks.*.attn.qkv"]),
], extra_modules=[
    AdapterExtraModuleConfig(component="language", targets=["model.language_model.norm"],
                             lr_multiplier=0.1, weight_decay=0),
])
```

Every full-path glob must match after layer restrictions. Cross-component matches,
unsupported module types, missing layer indices, conflicting overlaps and partial
selection of tied aliases fail before injection. Shared modules/parameters are
resolved by identity; compatible aliases share adapters and one optimizer entry.
Cross-block ties are refused for FSDP2. Architecture-declared embedding ties remain
frozen; embedding/head adaptation is not supported. Arbitrary custom ties are not a
claim of compatibility with a stock HF architecture.
Portable reload preserves declared custom adapter aliases; stock HF merged export
rejects them because its config cannot reconstruct those ties. Architecture-declared
frozen embedding/head ties are preserved through both export types.

The resolved manifest records component, path, type, shape, owner alias, rank,
effective scaling, method/init/dropout/dtype, LR/decay and expected trainable count.
It is checked against actual trainability and final optimizer parameter identities.
Clear `optimizer.train_keys`, `decay_groups` and `lr_multipliers`: adapter component
settings own those policies. `lr_multiplier` multiplies the global optimizer LR;
`weight_decay` is absolute and defaults to zero. The initial profile uses AdamW,
FP32 adapter storage/accumulation, BF16 or FP32 frozen base, eager/SDPA and full
non-reentrant activation checkpointing or none. BF16 adapter storage is refused.
Separately trained normalization parameters remain FP32 while their output keeps
the input activation dtype. Merged BF16 exports round these parameters once.

## Methods and lifecycle

LoRA and rsLoRA use the existing `LoraInjectedLinear`: Kaiming-uniform A, zero B,
scale `alpha/r` or `alpha/sqrt(r)`. Zero B makes the initial A gradient zero; tests
check connectivity and actual changes over several updates. Autocast is disabled
around adapter math; the final update is cast to the base output dtype. Dense merge
uses the existing FP32 accumulation policy with autocast and TF32 disabled.

DoRA keeps a trainable FP32 output-row magnitude, initialized to the base row norm.
The direction norm is detached during training, following the
[DoRA reference](https://github.com/NVlabs/DoRA). Bias is unscaled. Magnitudes enter
the optimizer, full checkpoint and portable state. Export normalizes the completed
direction and applies the trained magnitude. Only one adapter is merged into its
base; combining multiple adapters is not implemented.

PiSSA initializes A/B using `svd` or randomized `svd_lowrank` with an explicit
iteration count (default 4). It stores immutable initial factors A0/B0 and computes
`W0 x + s(BA - B0A0)x`. This represents the residual base `W0 - s B0A0` while
leaving the original parameter bytes intact. The artifact contains both initial
factors; it must never be loaded as raw BA on W0. Merge uses the
[PiSSA rank-2r conversion](https://github.com/MuLabPKU/PiSSA), concatenating
`[A; A0]` and `[B, -B0]` with the original effective scale. PiSSA requires dropout
zero; DoRA plus PiSSA initialization on the same target is rejected.

Adapter-enabled construction follows this order:

1. Construct the native base on meta, prepare its existing multimodal forward,
   materialize, tie architecture-declared embeddings, and strictly load the base.
2. Resolve all targets; install/initialize adapters, or allocate initialized
   placeholders for warm start/exact resume. Resume never decomposes the base.
3. Freeze the base, enable adapters/extra normalization parameters, and warm-load
   a portable artifact if requested.
4. Apply activation checkpointing and FSDP2 after every adapter exists. FSDP retains
   mixed BF16 frozen/FP32 trainable storage, with FP32 gradient reduction.
5. Construct AdamW from the final parameter objects; assert complete, unique
   membership. Exact resume then strictly restores trainer/model/optimizer,
   scheduler, clocks, rank RNG and the data cursor through the existing boundary.

No adapter tensor remains on meta. The adapter-disabled builder retains its
existing sharding-before-loading order. Adapter initialization currently loads the
full base on each GPU before sharding, so the base must fit on each device; this
is intentional for the initial 80 GB profile. Native Qwen3-VL's single visual
forward and graph-connected dummy media path are preserved for differing modality
presence across ranks. Vision features are not detached or cached by this path.

## Three artifact types

**Training checkpoints** remain the existing full consolidated or DCP transaction.
They include frozen weights as well as adapter tensors, PiSSA initial factors,
optimizer, scheduler, update state, RNG and data cursor. No adapter-only checkpoint
optimization is introduced. Resume uses `rl.resume`; missing adapter keys or changed
objective/base/processor/media identity fail. Resume across a changed DP world is
not an exact-continuation claim. `warm_start` must be cleared for exact resume.

**Portable adapters** are immutable directories with `adapter.safetensors` and a
commit-last `adapter_manifest.json`. They record SHA-256 over all original weight
shards and config/processor/tokenizer/template sidecars, the resolved target map,
extra modules, method configuration, effective processor arguments and relevant
dependency/format versions. A same-size different base is rejected. Export is an
atomic no-replace publication using checkpoint IO primitives. These artifacts
contain no optimizer or data state and cannot resume training.

```bash
python -m cosmos_framework.posttrain.checkpoint.tools.adapters \
  --checkpoint /runs/example/checkpoints/iter_000006 --out /exports/adapter
# In a recipe, for a new training run:
# config.rl.adapters.warm_start = "/exports/adapter"
```

**Merged HF models** use pure merge preparation, then the existing locked/staged
HF materializer and sidecar verification. Export leaves live parameters, optimizer
membership and tensor versions unchanged. Stock HF reload is tested separately
from native portable reload.

```bash
python -m cosmos_framework.posttrain.checkpoint.tools.adapters \
  --adapter /exports/adapter --base "$SFT_MODEL_PATH" \
  --merged --out /exports/merged
```

Small FP32 fixtures require output agreement within `atol=3e-5, rtol=3e-5` and
bitwise controlled next-update recovery. GPU qualification records measured errors;
BF16 merge permits `atol=.05, rtol=.02`, mean absolute logit error at most `.005`,
and identical argmax on controlled probes. Portable GPU reload uses `2e-5` and a
`2e-6` mean bound; controlled GPU continuation uses `atol=1e-7, rtol=1e-5` on
trainable parameters and optimizer state. Passing a backward call alone is not a
quality or resource qualification.

WTS annotations can be converted without resampling using
`posttrain.scripts.prepare_wts_sft`. Full-epoch jobs must account for every original
row under the existing DP divisibility rule. Report training/validation loss,
held-out multiple-choice accuracy/invalid answers, samples and supervised tokens
per second, setup time, peak CUDA memory, checkpoint/export size and latency,
checkpoint SHA, wheel SHA and source revision. Full WTS runs begin only after the
focused installed-package and GPU gates pass; AutoML is not a substitute.
