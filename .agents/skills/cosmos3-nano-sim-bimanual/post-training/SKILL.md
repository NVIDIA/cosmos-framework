---
name: cosmos3-nano-sim-bimanual-post-training
description: Train Cosmos3 Nano Sim Bimanual from Cosmos3-Nano with Cosmos Framework's four native stage TOMLs and a local training bundle. Covers data preparation, initialization, training, checkpoint handoffs, resume and student export.
---

# Post-train Cosmos3-Nano-Sim-Bimanual

Train an action-conditioned world model that predicts the visual consequences
of bimanual actions. It consumes a task description, RGB observations and an
aligned action trajectory, and produces video rather than robot actions.
The worked chain starts from Cosmos3-Nano and runs Stage 0 through Stage 3.

## Choose your starting point

| Goal                                | Start with                                               | Follow                                          |
| ----------------------------------- | -------------------------------------------------------- | ----------------------------------------------- |
| Start a new four-stage training run | Cosmos3-Nano and a completed local bundle                | [Stage 0 Quickstart](#quickstart-train-stage-0) |
| Continue an existing chain          | Your completed stage checkpoint and saved configuration  | [Continue training](#continue-training)         |
| Generate video                      | Released Bimanual model or your exported Stage 3 student | [Inference](../inference/SKILL.md)              |

Use the [data contract](references/data-contract.md) for action meaning,
normalization, timing and captions, and the [parameter guide](references/parameter-guide.md)
before changing the recipe. The commands below use Cosmos Framework.

Use the [native Bimanual recipes](../../../../cosmos_framework/configs/simulation/common/bimanual_profiles.py)
and [Bimanual data preflight](../../../../docs/bimanual_data_preflight.md)
in this Framework checkout.

## Set up the environment

Follow the [Framework setup guide](../../../../docs/setup.md),
then check the additional [Bimanual requirements](../shared/release-workflow.md#cosmos-framework).
Reuse a working Framework environment.

## Quickstart: train Stage 0

### Activate the Framework environment

Use a dedicated Framework checkout, Python 3.13, the documented CUDA/FFmpeg
system libraries, and a GPU allocation. Set `FRAMEWORK_DIR` to that checkout.
From its root, activate the prepared training environment:

```bash
cd "$FRAMEWORK_DIR"
source .venv/bin/activate
export COSMOS_INTERNAL=0 COSMOS_TRAINING=1
python -m cosmos_framework.scripts.train --help
```

For a CUDA 12.8 host, use the matching group from the setup guide.
Do not clear `LD_LIBRARY_PATH`; keep the CUDA compatibility and FFmpeg library
paths required by your installation. The training extra includes
`mujoco==3.10.0` for calibrated action/FK preprocessing.
The registered layout uses FSDP shard degree 8 and context-parallel degree 4.
The eight-process command below must run inside an allocation that supports
that layout and the sample's memory requirements. Multi-node placement follows
[Framework training](../../../../docs/training.md#option-b-raw-torchrun).

### 1. Prepare and validate training data

Set `BIMANUAL_BUNDLE` to a completed user-owned local bundle. It contains
`manifest.json`, `samples.json`, local media/action resources, and
`resources/tokenizer/`. An inference JSONL or action sidecar alone is not a
training bundle. The Framework loader consumes this format; it does not provide
a raw-robot-data-to-bundle conversion command.

Download the versioned single-source robot example from the
[Sim-Bimanual checkpoint](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual/tree/main/examples/training).
Its README records coverage, checksum verification and extraction commands.
Use it to understand the bundle layout and native dataloader handoff; it is a
small integration example, not a convergence dataset. The duplicated AgiBot
inputs and paired GT videos show the relationship to inference, while the
training bundle retains raw robot actions rather than already-normalized
inference sidecars.

Prepare data using the [local bundle layout](references/data-contract.md#user-owned-local-bundle),
[action schema](references/unified-v1-action-schema.md) and
[caption guide](references/caption-guide.md):

1. Split source episodes before creating windows, keeping all windows from an
   episode in one split.
2. Choose robot-window lengths for the task and available memory, then keep RGB
   and actions aligned: a window with `T+1` observations has `T` transitions.
   Window length is selected by the user and is not fixed by the loader.
3. Preserve coordinate frames, rot6d convention, calibrated gripper width,
   domain IDs, validity masks and the pinned global-asinh statistics.
   Training materializes raw captures through the native transform; an
   inference sidecar's already-normalized values are not raw training actions.
4. Start with one supported robot source. Advanced users may provide multiple
   supported sources; Framework normalizes their
   declared relative weights. Framework also applies stage-specific rendering
   internally, so do not create separate source-resolution strata.
5. Inspect representative decoded samples and the actual packed batch, including
   59 semantic action values padded to 64, their masks, units, and semantics.

Bind the prepared bundle, check its files, then read one packed Stage 0 sample
before loading model weights:

```bash
export COSMOS_SIM_BIMANUAL_LOCAL_BUNDLE="$BIMANUAL_BUNDLE"
python -m cosmos_framework.scripts.validate_bimanual_data \
  --bundle "$COSMOS_SIM_BIMANUAL_LOCAL_BUNDLE"
python -m cosmos_framework.scripts.validate_bimanual_data \
  --bundle "$COSMOS_SIM_BIMANUAL_LOCAL_BUNDLE" --materialize --stage 0
```

Both commands must report `passed: true`. The second must also report
`materialized: true`, batch sample/token counts and tensor shapes/dtypes.
See [what these checks cover](references/data-contract.md#validate-with-the-native-loader).

The native `make_bimanual_local_dataloader` handles stage-specific stream
selection and packing, and installs loader-state resume support. Use the
[Framework data interface](references/data-contract.md#cosmos-framework).

### 2. Prepare the Cosmos3-Nano initializer and video VAE

Set `NANO_REVISION` to the chosen immutable `nvidia/Cosmos3-Nano` revision.
The following paths are used by conversion and training later in this guide:

```bash
export NANO_HF_DIR="$PWD/checkpoints/Cosmos3-Nano-$NANO_REVISION"
export NANO_DCP_ROOT="$PWD/checkpoints/Cosmos3-Nano-dcp-$NANO_REVISION"
export VIDEO_VAE_DIR="$PWD/checkpoints/wan22_vae"

hf download nvidia/Cosmos3-Nano \
  --revision "$NANO_REVISION" --local-dir "$NANO_HF_DIR"
hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
  --local-dir "$VIDEO_VAE_DIR"

python -m cosmos_framework.scripts.convert_model_to_dcp \
  --checkpoint-path "$NANO_HF_DIR" -o "$NANO_DCP_ROOT"

export COSMOS_SIM_BIMANUAL_GA_CHECKPOINT="$NANO_DCP_ROOT"
export VIDEO_VAE_PATH="$VIDEO_VAE_DIR/Wan2.2_VAE.pth"
test -f "$NANO_DCP_ROOT/model/.metadata"
test -f "$VIDEO_VAE_PATH"
```

Record the model/VAE identities and conversion revision. This produces model
weights, not optimizer or trainer state. The native Stage 0 recipe skips
`net_ema.` and the three new action modules during this fresh load, keeps
regular-weight loading strict, and copies the loaded/fresh regular weights
into EMA through its built-in callback. Do not add a second EMA callback.
Same-run resume restores saved EMA instead.

### 3. Configure Stage 0 from Cosmos3-Nano

Set `IMAGINAIRE_OUTPUT_ROOT` to writable storage mounted at the same absolute
path on every worker, and choose a unique `RUN_NAME` for the chain.

```bash
export IMAGINAIRE_OUTPUT_ROOT="$BIMANUAL_OUTPUT_ROOT"
RUN_NAME=bimanual-my-data
TOML=examples/toml/sft_config/sim_bimanual_stage0_bidir.toml

python -m cosmos_framework.scripts.train \
  --sft-toml="$TOML" --dryrun -- \
  "job.name=$RUN_NAME-stage0" \
  "model.config.tokenizer.vae_path=$VIDEO_VAE_PATH" \
  model_parallel.context_parallel_size=4
```

The TOML selects `sim_local_bimanual_stage0_bidir`. Inspect the resolved config
under `$IMAGINAIRE_OUTPUT_ROOT/cosmos3_simulation/bimanual/$RUN_NAME-stage0/`.
Confirm the local bundle/tokenizer, Cosmos3-Nano load path, Stage 0 EMA callback,
parallelism and `job.wandb_mode=disabled`. Training does not require a release
inference checkpoint or a W&B key. A dry run resolves config without loading
weights, decoding a batch or taking an optimizer step.

### 4. Launch Stage 0

Use the same shell, paths and run name:

```bash
torchrun --standalone --nproc-per-node=8 \
  -m cosmos_framework.scripts.train --sft-toml="$TOML" -- \
  "job.name=$RUN_NAME-stage0" \
  "model.config.tokenizer.vae_path=$VIDEO_VAE_PATH" \
  model_parallel.context_parallel_size=4
```

#### Optional: run a 12-iteration smoke test

To check data loading, forward/backward updates, and checkpoint writing before a
reference-length run, use a dedicated run name and add the two bounded-run
overrides:

```bash
RUN_NAME=bimanual-my-data-smoke
torchrun --standalone --nproc-per-node=8 \
  -m cosmos_framework.scripts.train --sft-toml="$TOML" -- \
  "job.name=$RUN_NAME-stage0" \
  "model.config.tokenizer.vae_path=$VIDEO_VAE_PATH" \
  model_parallel.context_parallel_size=4 \
  trainer.max_iter=12 \
  checkpoint.save_iter=12
```

Confirm finite losses and gradients and a completed checkpoint at iteration 12.
The same overrides can be used with another stage's TOML. Omit them for the
reference training budgets.

Monitor losses and gradients, inspect the emitted batch semantics, and keep the
job running through completed checkpoint saves. To resume an interrupted Stage
0 run, rerun the same command with the same run name, output root, bundle,
topology, and parent checkpoint; Framework restores the saved iteration,
optimizer, scheduler, EMA, and loader state.

## Continue training

The native reference budgets are 35,000, 8,000, 4,000 and 1,500 trainer
iterations for Stages 0–3, respectively. All four TOMLs live under
`examples/toml/sft_config/`:

| Stage                        | TOML                             | Required initialization                                                                                           |
| ---------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| 0: representation adaptation | `sim_bimanual_stage0_bidir.toml` | `COSMOS_SIM_BIMANUAL_GA_CHECKPOINT`                                                                               |
| 1: teacher forcing           | `sim_bimanual_stage1_tf.toml`    | `COSMOS_SIM_BIMANUAL_STAGE0_CHECKPOINT`                                                                           |
| 2: TF-dCM                    | `sim_bimanual_stage2_tfdcm.toml` | `COSMOS_SIM_BIMANUAL_STAGE1_CHECKPOINT` for student and frozen teacher                                            |
| 3: SF-DMD                    | `sim_bimanual_stage3_sfdmd.toml` | `COSMOS_SIM_BIMANUAL_STAGE2_CHECKPOINT` for student; `COSMOS_SIM_BIMANUAL_STAGE0_CHECKPOINT` for reference/critic |

After accepting Stage 0, bind its exact completed iteration root, not `model/`:

```bash
export COSMOS_SIM_BIMANUAL_STAGE0_CHECKPOINT="$STAGE0_CHECKPOINT_ROOT"
TOML=examples/toml/sft_config/sim_bimanual_stage1_tf.toml
```

Repeat the config check and launch with this TOML and `job.name=$RUN_NAME-stage1`.
After Stage 1, export `COSMOS_SIM_BIMANUAL_STAGE1_CHECKPOINT` and select Stage 2;
after Stage 2, export `COSMOS_SIM_BIMANUAL_STAGE2_CHECKPOINT` and select Stage 3.
Use distinct stage run names. The recipe adds `/model` where a nested load needs
it. Every new stage starts fresh optimization state; same-stage resume restores
the existing run. See [Framework checkpoint handoffs](references/checkpoint-and-resume.md#cosmos-framework).

Keep the four-step Stage 3 schedule. It updates the student once per six trainer
iterations and the critic on the other five, so 12 iterations exercise two
student and ten critic updates. Inspect both optimizers, not just the iteration
counter. Frozen teacher/reference and critic weights are not inference artifacts.

## Export and check inference

Accept each stage only after finite updates, completed save and fresh-process
readback. Keep its resolved config and immutable checkpoint identity. Use
held-out generation to judge model quality separately from short-run execution.

For a portable Stage 3 student, set `STAGE3_CHECKPOINT_ROOT` to its completed
iteration root, `STAGE3_CONFIG` to that run's saved `config.yaml`, and
`STUDENT_EXPORT` to a new export directory:

```bash
python -m cosmos_framework.scripts.export_model \
  --checkpoint-path "$STAGE3_CHECKPOINT_ROOT" \
  --config-file "$STAGE3_CONFIG" \
  --experiment sim_local_bimanual_stage3_sfdmd \
  --use-ema-weights --no-vit --student-only-checkpoint-metadata \
  -o "$STUDENT_EXPORT"
```

This command selects student EMA explicitly. Preserve the selected variant and
saved four-step schedule, and inspect the export's action metadata and local-path
sanitization. Follow [inference](../inference/SKILL.md#use-your-own-stage-3-checkpoint)
with that export. The released schema-5 example also uses four steps; preserve
each export's exact saved ladder rather than assuming the schedules are identical.

## References

- [Training data contract and local bundle layout](references/data-contract.md)
- [Unified action schema](references/unified-v1-action-schema.md)
- [Caption preparation](references/caption-guide.md)
- [Parameter guide](references/parameter-guide.md)
- [Framework environment and workflow](../shared/release-workflow.md#cosmos-framework)
- [Framework checkpoint handoffs and resume](references/checkpoint-and-resume.md#cosmos-framework)
