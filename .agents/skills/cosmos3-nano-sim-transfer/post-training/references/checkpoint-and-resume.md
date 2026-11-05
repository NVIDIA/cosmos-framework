# Save, resume and hand off Cosmos3-Nano-Sim-Transfer weights

Choose a checkpoint for the operation: initialize a new stage, resume training
or export for inference. Preserve the parent lineage in the
[method reference](canonical-recipe.md). Framework Stages 0–3 correspond to
method steps 1–4.

## Choose the operation

| Operation                              | Weights                                                                                           | Optimizer, scheduler and trainer                                              | Data state                                             |
| -------------------------------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------ |
| Start a new training step              | Load the selected parent's specified weight variant                                               | Start fresh                                                                   | Start fresh                                            |
| Resume the same training run           | Restore regular trainable weights and EMA. Include the score model for generated-history learning | Restore all saved state, including both optimizers/schedulers when applicable | Restore the local-loader cursor state                  |
| Continue from a chosen full checkpoint | Load the complete, explicitly selected checkpoint                                                 | Restore the matching state and iteration                                      | Keep the same bundle, seeds, topology and accumulation |
| Export for inference                   | Select Stage 3 student EMA                                                                        | Exclude training-only state                                                   | Not needed                                             |

**EMA** means an exponential moving average of trainable weights. Use it
for stage transitions and final inference. A new stage copies the selected EMA
into trainable weights and starts fresh optimization state. Stage 0 starts from
base regular weights. Same-run resume restores regular weights, matching
optimizer state and EMA separately.

Create a fresh run name and output namespace for each new step or deliberate
rewind. When resuming, retain the existing identity. Check the training logs for
the source actually loaded and the restored iteration. The native
checkpointer gives an existing same-run latest checkpoint precedence over the
configured initializer. Preserve the format and directory structure of every
model-only or full-state artifact.

## Bind every parent explicitly

| Stage              | Initial weights                                                      | Fresh-state handoff                                                                                   |
| ------------------ | -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| 0, bidirectional   | Compatible base regular weights                                      | Fresh optimizer/scheduler/trainer; select your completed EMA                                          |
| 1, teacher forcing | Your Stage 0 EMA                                                     | Copy EMA into regular weights; select your Stage 1 EMA                                                |
| 2, TF-dCM          | Your Stage 1 EMA for student and frozen teacher                      | Bind both consumers to the same completed parent; select your Stage 2 EMA                             |
| 3, SF-DMD          | Your Stage 2 EMA for student; your Stage 0 EMA for teacher and score | Load score model independently with FP32 parameters; initialize student EMA; select final student EMA |

To start at Stage 1, use a completed Stage 0 checkpoint. Select parent
checkpoints using held-out outputs.

The [local Framework walkthrough](local-training.md) supplies executable commands
for these bindings:

| Consumer                         | Path                                  | Weight/state selection                                                                            |
| -------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Stage 0 initializer              | `BASE_CKPT_ROOT`, containing `model/` | `load_training_state=False`, `load_ema_to_reg=False`: regular weights, fresh state                |
| Stage 1 initializer              | `STAGE0_CKPT_ROOT`                    | `load_training_state=False`, `load_ema_to_reg=True`: EMA copied into regular weights, fresh state |
| Stage 2 student                  | `STAGE1_CKPT_ROOT`                    | Same EMA warm start as Stage 1                                                                    |
| Stage 2 frozen teacher           | `STAGE1_CKPT_ROOT/model`              | Model-specific loader reads the selected EMA                                                      |
| Stage 3 student                  | `STAGE2_CKPT_ROOT/model`              | Model-specific loader reads EMA and initializes student EMA from it                               |
| Stage 3 teacher and score model  | `STAGE0_CKPT_ROOT/model`              | Independent loads of the same EMA source. Score model retains FP32 training parameters            |
| Explicit full-state continuation | Completed child-run iteration root    | `load_training_state=True`, `load_ema_to_reg=False`                                               |

Stage 3 starts with an empty `checkpoint.load_path` and fresh training state.
Its model-specific paths supply the initial weights. For a same-run restart,
the latest completed checkpoint restores all saved state. Keep the existing
run name and output root for that operation.

Frozen teachers do not receive optimizer updates. Retain every parent required
by the training implementation for the entire child run, including restarts.
The Framework Stage 2/3 models construct their parents before restoring the
child’s saved state, so restart still requires those parent artifacts.

Use a model-only teacher artifact to initialize a later stage. Retain the
teacher’s completed full-state checkpoint when its own training may continue.

## What a complete restart must preserve

A full checkpoint for generated-history learning contains student weights,
student EMA, score-model weights, both optimizer and scheduler states, trainer
iteration, available per-rank random-number-generator state and the trainer's
gradient-scaler state. BF16 disables gradient scaling, but its state field can
still be present. Recover the frozen teacher from its identified parent.

The [native local configuration](local-training.md) installs the
`local_simulation_resume` callback for every stage. It saves the joint packer
and child-loader cursor state, including Stage 0's mixed-resolution streams.
Keep bundle contents, seeds, rank topology and accumulation unchanged to resume
the same data sequence. Confirm restored iteration and local-loader state in
the resumed log.

The asynchronous saver snapshots training state into pinned host
memory before writing. See [checkpoint memory](#checkpoint-memory) before
enabling this mode. Timed-save retention must keep the previous completed
checkpoint until its replacement finishes. The latest completed timed save and
regular iteration milestones are separate. Treat a partly written checkpoint
as incomplete even if its directory exists.

## Checkpoint memory

Asynchronous checkpoint staging uses shared, pinned CPU tensors. Size
`/dev/shm` for the combined model and optimizer staging buffers of all ranks
sharing that mount, with headroom for dataloaders and other processes. Total
host RAM must cover these buffers as well as the live training workload;
increasing `/dev/shm` alone does not add physical memory. GPU memory and free
checkpoint-disk space are separate requirements.

Check the mount inside the training container with `df -h /dev/shm`.
Framework checks the aggregate tensor staging requirement before allocation
and reports required and available capacity if the mount is too small. Increase
the container's `--shm-size` (or use the supported host IPC setup described in
[Framework setup](../../../../../docs/setup.md#docker-container)), or select
`checkpoint.dcp_async_mode_enabled=False` as in the
[short training walkthrough](local-training.md#run-and-resume-each-stage).
Synchronous saving retains the full-state checkpoint and resume format while
avoiding this extra asynchronous staging tree; ordinary host-memory and
dataloader shared-memory requirements still apply.

## Resume from a checkpoint

Rerun the same command with the same `IMAGINAIRE_OUTPUT_ROOT` and job name.
The checkpointer reads `checkpoints/latest_checkpoint.txt` and restores the
latest completed checkpoint. Increase `trainer.max_iter` to the desired total
iteration count. The [training walkthrough](local-training.md) shows this
operation for every stage.

For a deliberate rewind, use a new job name and set `checkpoint.load_path` to
the selected iteration root, `checkpoint.load_training_state=True` and
`checkpoint.load_ema_to_reg=False`. Clear initialization filters with
`checkpoint.keys_to_skip_loading=[]` and `checkpoint.keys_not_to_resume=[]` so
all saved weights and state are restored. Retain the same bundle and parent
bindings. The new job name prevents an existing latest-checkpoint pointer from
overriding the selected initializer.

Keep full-state checkpoints, including FP32 master weights and optimizer
accumulators, for further training. Model-only inference exports omit optimizer
and loader state. Before a long continuation, check the restored iteration and
run an update with the resumed model. Stage 3 updates the score model and
student on separate iterations.

## Pass the selected model to inference

Pass the selected model together with its model definition, tokenizer/VAE
identity, preprocessing, conditioning and prompt policy, sampling schedule and
cache policy. Include a small control video and prompt so the recipient can
verify a fresh load. Preserve the original training configuration separately
from any inference-only projection.

The selected final model uses student EMA after generated-history
learning. Its training cache window is 51 latent times with a sink of 1.
The default short inference recipe uses full history with no sink and generates
121 RGB frames, while training allows up to 201. For longer outputs, follow the
inference Skill’s long-rollout settings.

Follow the [inference Skill](../../inference/SKILL.md)
to load Stage 3 student EMA directly from its DCP checkpoint. Use the
[output comparison guide](../../inference/references/output.md)
to compare control adherence, appearance and motion on held-out examples.
