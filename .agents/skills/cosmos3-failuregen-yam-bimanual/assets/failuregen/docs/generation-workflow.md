# FailureGen generation workflow

FailureGen transfers motion from a recorded failure into a compatible successful
scene. Cosmos predicts the visual continuation from the host's initial camera
observation, the transferred actions, and neutral task text.

The README workflow uses three supplied donor/host pairs and a matching inference
bundle. Preparing new recordings and selecting new pairs are separate workflows.

## Workflow components

Paths below are relative to this repository.

| Stage                 | Component                                                                       | Purpose                                                                              |
| --------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Label recordings      | [Annotation guide](annotation-format.md)                                        | Define task phases, failure intervals, and episode outcomes for automatic selection. |
| Prepare robot actions | `src/failuregen/yam_training/prepare_eef.py`                                    | Prepare windows and normalization statistics from compatible EEF exports.            |
| Adapt Cosmos          | `src/failuregen/yam_training/launch.py` and `configs/yam_eef_gb200.toml`        | Train the dynamics model for new or different data.                                  |
| Select new pairs      | `src/failuregen/workflow.py` and `src/failuregen/yam_training/failure_types.py` | Identify failure donors and match clean host windows by task phase and robot pose.   |
| Prepare the examples  | `src/failuregen/showcase.py` and `configs/showcase.json`                        | Use the three supplied pairs and timestamps without reading annotation documents.    |
| Generate video        | `src/failuregen/adapters/cosmos.py`                                             | Run Cosmos with prepared 20-D actions and the adapted checkpoint.                    |

[Code provenance](code-provenance.json) records source revisions and file hashes.

## Prepare robot actions

For new recordings, preserve episode IDs, camera timing, and robot telemetry.
Use a conversion appropriate to the robot's kinematics to obtain end-effector
poses and gripper states. The YAM preparation code expects this compatible EEF
export; it does not infer another robot's kinematics from videos.

1. Sample observations and poses from 30 Hz to 5 Hz before computing actions.
2. For each arm, compute the relative transform between successive poses:
   `delta_T = inverse(T_previous) @ T_current`.
3. Pack translation in meters, the first two columns of the relative rotation
   matrix, and the absolute recorded gripper state at the destination time.
   Each arm contributes ten values, for a total of 20.
4. Split by episode so that windows from one recording stay in the same split.
5. Compute normalization statistics from training transitions only and retain
   the prepared manifest, state arrays, and statistics with the training run.

Normalize each action dimension using
`2 * (a - q01) / max(q99 - q01, 1e-8) - 1`, without clipping. Generation must
reuse the statistics associated with its checkpoint.

FD training is needed only when using new or different data. See the
[FD training guide](fd-training.md) for the required export contents, preparation,
and optional adaptation commands. The README workflow downloads an already adapted
checkpoint and matching inputs from Hugging Face.

## Select donors and hosts

The README workflow uses these fixed windows, all with seed 0:

| Failure type  | Donor episode · start | Host episode · start |
| ------------- | --------------------- | -------------------- |
| Handover drop | 42 · 10.6 s           | 79 · 10.8 s          |
| Pickup miss   | 40 · 7.2 s            | 43 · 4.4 s           |
| Pickup slip   | 61 · 8.6 s            | 46 · 7.6 s           |

For automatic selection, reviewed failure intervals identify donor events, and
task phases locate the motion preceding each event. A donor window must contain
enough lead-in to complete the blend before the failure begins.

Hosts must be distinct, clean successful episodes with a compatible task phase
and enough recorded actions for a complete window. The EEF planner compares
end-effector position, orientation, and gripper state at the transfer point to
rank compatible windows. This geometric score is not a prediction of generation
success. The README workflow validates its supplied windows instead of rerunning this
selection process.

## Keep inputs and checkpoint compatible

The inference bundle supplies the source videos, prepared actions, normalization
statistics, model configuration, and checkpoint. See the
[expected file structure](../../../README.md#dataset-and-checkpoint-contents).

`showcase.load_context` checks the prepared-manifest and normalizer hashes, the
20-D EEF contract, and the 32-domain model configuration. Preparation also checks
the selected state files and checkpoint metadata. Use the checkpoint's matching
prepared bundle; do not rewrite its contract to accept different inputs.

Use Cosmos framework revision `cf5d68c00d97ccd2480a2320ed652b92dec63102`, recorded
in the training contract. A different revision may have an incompatible model
configuration schema.

## Transfer motion and generate video

Transfer retains four host actions (0.8 seconds), blends into donor motion over
two actions (0.4 seconds), and projects blended rotation matrices onto SO(3).
The selected arm channels follow the donor segment; other channels retain the
host motion. Normalize the resulting actions using the checkpoint's statistics.

Cosmos receives the host observation, transferred actions, and neutral task
context. The profile uses `xdof_yam`, domain 16, action dimension 20, precomputed
actions, 30 sampling steps, guidance 1.0, shift 10.0, and throughput parallelism.

The camera canvas is 960×960, with the overhead view above the two wrist views;
the framework resolution tier is `image_size=720`. Each example has 16 actions
and 17 observation frames at 5 Hz, covering a 3.2-second continuation. Running
the README workflow requires a machine with four GB200 GPUs.

## Inspect the results

Compare the generated continuation with its donor and host. Check scene
consistency, visual validity, and the event that actually occurs. A generated
clip can show the intended failure, a different failure, or no failure.
Use the [annotation guide](annotation-format.md) when labeling those outcomes.

The three example inputs were checked against their reference action sequences,
and the notebook author confirmed successful generation of all three examples
on October 14, 2026. [Source notebook validation](notebook-validation.md) records
additional execution and playback checks for the upstream presentation changes.
Regenerated videos need not be bit-identical across hardware or framework versions.
