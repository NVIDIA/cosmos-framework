# Physical-action adapter SFT

The π0.5 offline recipe trains language K/V, the multimodal projector, selected
SigLIP vision layers and the action expert with LoRA, rsLoRA, DoRA or PiSSA.
It uses `entrypoints.sft`, `LLMTrainer`, the existing flow-matching objective,
rank data cursors and checkpoint transactions. The PPO recipe keeps its existing
frozen cached prefix. Offline SFT computes a fresh differentiable prefix for each
batch; its action loss can therefore update the language, projector and vision
adapters. This is supervised action prediction, not a claim of simulator success.

The initial checkpoint is `RLinf/RLinf-Pi05-LIBERO-SFT` at
`45ccfcc4e28634f1576ebf78cab0fbe2fd82432d`. The execution profile is Torch 2.10,
Transformers **4.57.6**, native mixed BF16/FP32 base storage, FP32 adapters, SDPA,
AdamW, pure DP/FSDP2, microbatch one, and activation checkpointing `none` or `full`.
Transformers 5 changes the SigLIP checkpoint namespace and is explicitly refused.
See the [evidence table](../adapter-family-progress.md) for installed GPU results;
small-model CPU tests alone do not qualify this checkpoint or world size.

## Data and training

Stage the checkpoint, its `physical-intelligence/libero/norm_stats.json`, the
PaliGemma SentencePiece tokenizer, and a public LIBERO LeRobot parquet snapshot.
The preparation tool reads embedded images, preserves demonstration action and
image conventions, and creates episode-disjoint windows with two real camera
views, an eight-dimensional state, a task prompt and ten seven-dimensional
actions. It rejects an existing output directory and writes input hashes and
split provenance. Select episode IDs actually present in the staged parquet:

```bash
"$SFT_PYTHON" -m cosmos_framework.posttrain.scripts.prepare_libero_flow_sft \
  --source /data/libero-snapshot --source-revision <immutable-dataset-revision> \
  --out /data/libero-flow --training-episodes 0,1 --validation-episodes 2,3 \
  --stride 5 --horizon 10
export SFT_MODEL_PATH=/models/pi05-libero
export SFT_TOKENIZER_PATH=/models/pi05-processor/tokenizer.model
export SFT_NORM_STATS_PATH=/models/pi05-processor/norm_stats.json
export SFT_DATA_ROOT=/data/libero-flow
export SFT_ADAPTER_METHOD=lora   # lora, dora, pissa
export SFT_ADAPTER_RSLORA=0      # set 1 for rsLoRA
"$SFT_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 \
  -m cosmos_framework.posttrain.entrypoints.sft \
  --config=cosmos_framework/posttrain/configs/sft_pi05_adapters.py
```

The recipe explicitly selects real native paths:

| Component | Targets |
| --- | --- |
| Language | `paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn.{k_proj,v_proj}` |
| Projector | `paligemma_with_expert.paligemma.model.multi_modal_projector.linear` |
| Vision | `paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.26.self_attn.{q_proj,v_proj}` |
| Action | `paligemma_with_expert.gemma_expert.model.layers.*.self_attn.{q_proj,v_proj}`, `action_in_proj`, `action_out_proj` |

Braces in this table are shorthand; configuration lists each full path separately.
Final-prefix Q/output/MLP targets are refused: the expert consumes per-layer K/V,
so those final-prefix outputs cannot influence its loss. Earlier prefix blocks may
use other ordinary linear projections. Each component supports independent method,
rank, alpha, dropout, initialization and LR/decay under the shared
[adapter rules](adapters.md). No adapter math or optimizer is duplicated here.

`rl.task="flow_bc"` requires the registered native packer and staged JSONL data.
Token packing/loss-mask knobs and gradient accumulation are refused for this
initial action profile. Validation masks padding at the action-element level and
reports `valid_action_count`; it never calls action elements language tokens.
Episode, row and media overlap between training and validation fails explicitly.
The third π0.5 view remains masked as in the native LIBERO processor.

## Checkpoint and export contracts

Training resume uses the same full checkpoint transaction as conversational SFT.
Use `rl.resume` for exact continuation; `rl.adapters.warm_start` starts a new run
from a portable adapter and must be cleared for resume. Adapter identity includes
the original checkpoint hash, tokenizer and normalization hashes, resolved targets,
effective media processing and native parameter precision map. PiSSA initial
factors and DoRA magnitudes are included in both checkpoint and portable state.

Portable export uses the existing checkpoint adapter tool. A merged physical-action
model has a distinct, committed format, `cosmos-pi05-merged-fp32-v1`:

```bash
"$SFT_PYTHON" -m cosmos_framework.posttrain.checkpoint.tools.adapters \
  --adapter /exports/pi05-adapter --base "$SFT_MODEL_PATH" \
  --processor /models/pi05-processor --merged --out /exports/pi05-merged
```

The processor directory must contain the exact recorded `tokenizer.model` and
`norm_stats.json`. Export reconstructs the original rounded base privately, applies
the shared FP32 method-specific merge, preserves embedding tying, and publishes
weights plus processor files before the hash manifest. Load with
`cosmos_framework.posttrain.model.pi05.adapter_export.load_merged`.
This FP32 inference profile is not a stock HF, PPO or Omni serving artifact. The
original mixed-precision source loader refuses it to prevent silent re-rounding.

Qualification compares fixed-noise, fixed-time physical velocity predictions and
losses across fresh portable reload and fresh merged reload. Portable tolerance is
`atol=rtol=2e-5`, mean error at most `2e-6`; FP32 merged tolerance is `atol=5e-4`,
`rtol=2e-4`, mean error at most `5e-5`. Controlled checkpoint continuation uses the
shared trainable/optimizer/RNG/data-state gate. Held-out flow loss is an offline
fit metric; rollout task success requires a separately qualified simulator run.
