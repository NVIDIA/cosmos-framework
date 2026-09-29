# Framework runtime operations

The helpers prepare images, checkpoints, and distributed execution for the
Framework reasoner recipes.

## Image identity

Build the root `Dockerfile` from a clean Framework checkout. Source-build mode
records full commit/tree IDs, base image, build timestamp, and
`/opt/cosmos/image-provenance.json`. Do not reuse a stale SQSH after rebuilding.
The packaged local image tag is not a public registry release. Remote
platforms require a registry image/digest or a verified shared SQSH.

A supplied SLURM SQSH is the runtime authority. Inspect the actual image and
allocated-node mounts, not a similarly named controller checkout. Preflight
must run the native Python as a non-root UID, import the converter, trainer,
status callback and TorchCodec, decode representative media on the assigned
CUDA device, and prove NCCL initialization and available output space.

## Checkpoints

Nano intake requires the explicit `qwen3_vl` or `cosmos3_omni` choice. Direct
HF input must contain complete config, processor/tokenizer and weight shards.
Omni conversion uses `cosmos_framework.scripts.convert_model_to_vlm_safetensors`
inside the same Framework runtime and writes into the selected compute-frame
checkpoint directory without mutating the source. Hub references resolve to
immutable revisions; private access requires only credential-presence checks.
The converter requires a full Hub commit for remote inputs, or hashes local
checkpoint contents. A missing filesystem path is not a Hub model ID. Its
output must be separate from both inputs and the cache; even `--force` cannot
replace those directories or follow an output symlink.

Framework training emits DCP. Before evaluate/inference/serving, use
`framework_checkpoint_action.py` to plan, prepare, and verify exact-key HF
export. Its manifest binds DCP metadata, saved config, base-model identity,
revision, and exported weights. Native PEFT is reconstructed and merged during
export. Reuse requires matching provenance, not just an existing directory.

## Distributed execution and decoding

Use one container/task per node, native torchrun, shard degree equal to GPUs
per node and replicate degree equal to nodes. Preserve actual child exit
status. Disable asynchronous DCP on multi-node shared SLURM runs. Follow
`execution/platforms/slurm/references/cosmos-slurm-guardrails.md` for exclusions,
timeouts, no-requeue policy, and fresh-record retries.

Training and evaluation use the native CUDA TorchCodec path with explicit
local-rank device binding. Caches are on-demand and sized from inspected unique
media. Preserve exact frame/pixel settings, validation batch sizes, padded
sample multiset, and ordering. The evaluation image must attest its baked
preprocessor; missing capabilities require a rebuilt image, not mounted source.

Structured terminal success and child exit zero are required in addition to
scheduler completion. Persist weighted metric numerators/counts, checkpoints,
configuration, provenance, logs, and status in the job record.
