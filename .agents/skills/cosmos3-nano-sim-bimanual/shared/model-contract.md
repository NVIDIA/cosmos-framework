# Cosmos3 Nano Sim Bimanual model contract

## Product task

Cosmos3 Nano Sim Bimanual is an action-conditioned world model for bimanual
manipulation. It consumes a task description, one RGB observation stream, and a
time-aligned action trajectory, then predicts the visual future implied by
those actions. It does not choose robot actions.

Training starts from the released `Cosmos3-Nano` model, converted to a local
DCP root and bound as `COSMOS_SIM_BIMANUAL_GA_CHECKPOINT`. Every later training
checkpoint is produced by the user's own ordered run. Training does not require
an inference release checkpoint. Inference uses either the current released HF
model or the user's exported Stage 3 student. Keep their identities and
saved schedules separate: the worked schema-5 release and canonical user-trained
Stage 3 both use four steps, but each export owns its exact saved ladder.

Each sample contains:

- a non-empty task or initial-scene description.
- one RGB video or initial RGB condition.
- one action transition for every adjacent pair of RGB frames.
- an embodiment identifier and action-validity mask.
- explicit frame rate, resolution, frame count, and seed.
- the preprocessing definition used by the Cosmos3-Nano model or by the user's own
  post-training output.

The output is one generated RGB video plus the resolved generation metadata.
For `T+1` RGB frames, provide exactly `T` action rows.

## Bimanual action schema

The model-facing `unified_v1` action is a 59-value vector:

```text
0:9    ego pose: xyz + rot6d
9:18   right wrist pose: xyz + rot6d
18:33  right fingertips: five xyz points
33     right gripper openness
34:43  left wrist pose: xyz + rot6d
43:58  left fingertips: five xyz points
58     left gripper openness
```

Map unsupported semantics to zero and mark them invalid in a 59-value boolean
mask. Pad model input from 59 to `max_action_dim=64`. Padded values and mask
entries remain zero. Calibrate hardware measurements before mapping, convert
poses to xyz plus rot6d, and apply the normalization defined by the training
config exactly once.

Reject non-finite actions, an incorrect width, a mask that disagrees with the
mapping, or timestamps that do not align with the RGB stream.

## Training stages

The complete product recipe has exactly four ordered stages, numbered Stage 0
through Stage 3. Stage 0 is the first stage:

1. **Stage 0 — representation adaptation:** adapts the Cosmos3-Nano weights to the RGB,
   language, and action distribution.
2. **Stage 1 — causal prediction:** learns chunked future prediction from clean
   history.
3. **Stage 2 — fixed-step distillation:** converts the Stage 1 result to the
   fixed-step generation schedule.
4. **Stage 3 — rollout distillation:** adapts the Stage 2 result on its own
   generated history. Its output checkpoint is used for inference. Frozen
   reference weights and the critic exist only inside training.

Stage 1, Stage 2, and Stage 3 use a two-latent-frame prediction chunk. For `T`
actions, temporal compression by four maps `T+1` RGB observations to
`1 + T/4` latent frames; `T` must be divisible by four. Rollout length is
selected by the input trajectory, independently of the prediction chunk size.
`state_t=300` is latent capacity, not an RGB-frame request.

Framework provides four `sim_bimanual_stage*.toml` files under
`examples/toml/sft_config/`. They select native local-bundle registrations from
`cosmos_framework.configs.simulation.experiment.bimanual.stages`. Follow the
[post-training tutorial](../post-training/SKILL.md) for exact names, resource
bindings and commands. The native builder uses the Cosmos3-Nano initializer and the
user's own checkpoint chain, starts fresh optimization state at each new stage,
and restores complete state for same-run resume. Do not substitute inference
examples for the local data bundle.
