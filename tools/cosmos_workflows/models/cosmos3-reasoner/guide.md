# Cosmos Framework reasoner workflows

This guide supports the existing `cosmos3-post-training` and
`cosmos3-inference` skills. Helpers plan Framework reasoner training,
checkpoint preparation, and evaluation. Native generator recipes do not require
this managed workflow layer.

## Read on demand

- [Data contracts](references/cosmos-data-specs.md): conversation and task-aware annotations.
- [Training parameters](references/cosmos-reason-parameters.md): native TOML and dense/PEFT settings.
- [Runtime operations](references/operations.md): image, checkpoint, decoder, and distributed contracts.
- [Evaluation](references/cosmos-reason-evaluate.md): sealed-plan inheritance, export, metrics, and completion.
- [Reproducibility gates](references/cosmos-reproducibility-gates.md): provenance and failure checks.
- [Single-GPU workflow](references/cosmos-reason-single-gpu-video.md): the same planner with one GPU.
- [Structured contract](references/skill_info.yaml) and [Framework runtime](references/training-contract.yaml).

## Intake and planning

Resolve the model before selecting an image or constructing a spec:

```bash
python scripts/cosmos_workflow.py resolve --model nvidia/Cosmos3-Nano --action train
```

For Nano, ask for the explicit source checkpoint format: `qwen3_vl` for a
compatible complete HF checkpoint, or `cosmos3_omni` for native exact-key
conversion. Do not infer that choice from history. The planner owns the Nano
architecture mapping and resolves immutable Hub revisions automatically.
For Edge, use its public checkpoint or a complete compatible local snapshot;
never substitute a workload-specific donor. Preserve source checkpoints.

Collect train/validation annotation paths and media roots, user-owned result
and checkpoint directories, dense/PEFT settings, frame/pixel budget, effective
batch, epochs, seed, and compute shape. Infer dataset family from the actual
records, not names or paths. Inspect every selected record/media reference;
reject empty splits, missing media, overlap, and insufficient records per GPU.
Multiple task-aware annotation files remain an explicit native input list.

The reasoner training planner supports Docker and SLURM. When both fit and
the user has not selected one, ask once. The shared execution layer also
retains Kubernetes, Brev, and virtualenv for workflows whose contracts support
them. Read the chosen platform's preflight and the shared launch guide before
submission; do not assume a platform from the helper's parser default.

Use `scripts/cosmos_workflow.py plan --help` for the input flags. The resulting
plan contains native Framework TOML, environment bindings, resolved model and
dataset fingerprints, image identity, target paths, and container preflight.
`--write-spec` identifies the staged TOML. Persist the sealed plan, then use
`--plan-artifact` for preflight, materialization, and rendering so input
inspection is not silently repeated. Verify the config checksum before launch.
Planning itself does not authorize downloads, image builds, or jobs.

## Checkpoint and image preparation

Omni preparation runs the native converter in the selected Framework image,
writing only into the user-owned checkpoint directory on the compute frame.
On SLURM, inspect shared paths through SSH and verify them in the allocation;
controller-local paths are not substitutes. A supplied SQSH is authoritative.
Source builds require an explicit build mode, clean full commit/tree identity,
and image provenance. Local image tags are build targets, not published images.

## Training and monitoring

Review the exact inputs, compute, image, costs, output paths, and commands with
the user. After approval, open the job record before submission and dispatch
through the platform's `submit/status/logs/cancel` contract. Bind the record ID
into the sealed render; never rerun an old submission script with a new record.

Training uses native torchrun, structured TOML, CUDA TorchCodec decoding,
resume-safe batching, and the Framework lifecycle callback. Select caches from
the inspected working set; preserve sampling and exact validation batch sizes.
Do not enable a diagnostic subset unless the user requests it. Full runs reject
sample limits. SLURM uses one task/container per node, no requeue, and native
HSDP shard/replicate degrees. Preserve child failure through cleanup.

Poll the actual platform. Scheduler completion alone is insufficient: require
child exit zero and structured terminal success. Report token-weighted loss
using numerator/count events and task-aware accuracy only where defined.
Infrastructure retries create a fresh record and checksum-verified retry plan;
never turn model/config failures into automatic retries.

## Evaluation and inference

Run `scripts/evaluation_workflow.py` with the sealed training plan and status.
It inherits the exact corpus, prompt, processor profile, seed, and model identity;
ask only for unresolved fields. Follow its automated Framework checkpoint
pre-action, verify the terminal export manifest on the compute frame, then
rerun resolution with the action model and manifest. DCP is not directly
evaluator-loadable; PEFT exports are merged by the native exporter.

Submit only a checksum-valid `ready=true` plan. Pass its `spec_bundle.execution`
unchanged to the platform. Native reasoner inference and serving use the same
verified HF checkpoint handoff. Invoke `cosmos-reasoner-inference` with native
CLI arguments, not a separate inference TOML:

```bash
cosmos-reasoner-inference --model_path /checkpoints/verified-hf \
  --media /data/example.mp4 --type video --prompt "Describe this video." \
  --num_frames 8 --max_new_tokens 128 --results_dir /results/inference
```

`--media` accepts an extracted file or directory, not a tar archive. For direct
native DCP inference outside the managed handoff, the same CLI exposes
`--config_file`, `--export_dir`, and `--vit_checkpoint_path`. PEFT checkpoints
are merged during export; do not request an inference-time LoRA merge.

The Framework planner does not advertise a
validated checkpoint quantization route; do not infer one from the standalone
quantization command's presence.
