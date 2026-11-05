# Cosmos3-Nano-Sim-Transfer training method

Framework **Stages 0–3** train a bidirectional teacher, teach causal generation,
distill to four denoising steps and learn from generated history. Start at
Stage 0 and pass your selected EMA outputs between stages.

Use the [Framework walkthrough](local-training.md) to run these stages, the
[parameter guide](parameter-guide.md) to plan changes, and the
[checkpoint guide](checkpoint-and-resume.md) for handoffs and resume. Preserve
the [shared input contract](../../shared/model-contract.md). The reference
settings below are starting values for task-specific tuning.

## Four steps, with different roles for the teacher

```text
Base model, regular weights
  └─ Stage 0 / step 1: Learn control-to-video generation over a complete window
       └─ Stage 1 / step 2: Learn causal generation using real past frames
            └─ Stage 2 / step 3: Distill generation to four denoising steps
                 └─ Stage 3 / step 4: Train on the model's own generated history
                      └─ Cosmos3-Nano-Sim-Transfer, selected EMA weights
```

A **teacher** is a frozen model providing a training target. **Teacher forcing**
is different: it supplies real previous video frames as history. Step 2 uses
teacher forcing without a separate frozen teacher. Step 3 freezes the model
from step 2. Step 4 uses the bidirectional model from step 1 as its teacher.

| Stage / method step                                      | What the model learns                                                                                          | Initialization and selected output                                                                                                                         |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Stage 0 / step 1. Whole-window control-to-video learning | Predict target RGB from aligned controls and text, attending across the complete window                        | Start from compatible base regular weights. Select your completed Stage 0 EMA                                                                              |
| Stage 1 / step 2. Causal learning from real history      | Predict the current noisy RGB latent using controls through the current time and real RGB strictly in the past | Start from your selected Stage 0 EMA. Select your completed Stage 1 EMA                                                                                    |
| Stage 2 / step 3. Four-step consistency distillation     | Match a teacher-guided denoising transition with a student requiring fewer sampling steps                      | Initialize student and frozen teacher from the same selected Stage 1 EMA. Select your completed Stage 2 EMA                                                |
| Stage 3 / step 4. Learning from generated history        | Improve the student's generated rollout using distribution-matching supervision                                | Student starts from your selected Stage 2 EMA. Frozen teacher and trainable score model start from your selected Stage 0 EMA. Select the final student EMA |

Each new stage starts fresh optimizer, scheduler, trainer and data state.
Select parent checkpoints using held-out outputs and preserve initialized
modules across stage transitions.

The native local config preserves Stage 0's mixed 480/720 streams and overrides
its dropout to 0.1. Stages 1–2 also set dropout 0.1; Stage 3 uses the required
480/zero-dropout bundle policy. Local bundles use their prepared sample
inventory, synchronized latent-length groups and saved local-loader state.
The local adapter emits one pair per rank in Stages 1–3 with zero workers and
prefetch 1.

## Prepare the data and conditioning

Each example contains one control video, the corresponding RGB video and a
caption describing their shared time interval. Select edge, blurred RGB, depth
or segmentation with equal weights. The target is RGB.

Use the same source-frame indices and camera geometry for target and persisted
controls. Directly resize RGB with bicubic antialiasing, depth with bilinear
interpolation, and segmentation with nearest-neighbor interpolation. Derive
edge and blur controls from the resized RGB. Reject examples requiring more
than 3% aspect-ratio distortion, and do not reflection-pad the target. Step 1
resizes after decoding. Steps 2–4 resize during bounded decoding to reduce
memory use while preserving the same geometry.

Retained captions are sampled as 93% structured captions and 7% dense prose
captions from the same window. The optional real-RGB prefix has 0, 1 or 2 latent
frames with probabilities 0.7, 0.2 and 0.1. A prefix is known RGB conditioning,
not an extra control modality. Mask its positions out of the generation loss.

Step 1 uses 480- and 720-resolution streams with weights 4 and 2, respectively,
so the expected stream-selection frequency is 2:1. Steps 2–4 use the 480 tier.
The local bundle builder selects each clip's longest eligible caption window
and groups records by latent length. The loader keeps length groups aligned
across ranks, shuffles within each group and pads groups to distribute samples
across the available ranks. Each draw applies the stage's resolution and caption
policy. The [bundle preparation guide](../../shared/local-dataset.md) describes
the input layout and alignment checks.

## Stage 0 / step 1. Learn whole-window control-to-video generation

The bidirectional model learns a rectified-flow denoising objective over the
whole target window. Rectified flow trains a network to predict a direction
from noisy inputs toward real data. Its time distribution, loss weighting and
noise shift are part of the recipe.

| Setting                            | Reference value and meaning                                                                                                             |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Architecture                       | Bidirectional attention. Cosmos3 Nano Reasoner, Qwen3-VL-8B tokenizer and Wan2.2 video VAE                                              |
| Video lengths                      | 480 tier: up to 201 RGB frames. 720 tier: up to 149. Minimum 5 RGB frames                                                               |
| Loader draw size                   | 2 examples from the 480 stream or 1 from the 720 stream per child-loader fetch                                                          |
| Packing                            | At most 74,000 tokens. No fixed maximum number of examples per packed batch                                                             |
| Caption / control dropout          | Caption probability 0.1. Control probability 0                                                                                          |
| Flow objective                     | Transformed-uniform video-noise distribution defined below, uniform time weight, loss multiplier 10. Noise shift 5 at 480 and 10 at 720 |
| Optimizer                          | Fused Adam. Base learning rate `1e-4`, betas `(0.9, 0.99)`, weight decay 0.05, epsilon `1e-8`                                           |
| Learning-rate schedule             | Linear schedule with 0 warmup steps, start multiplier `1e-6`, maximum multiplier 0.4, minimum 0, cycle 100,000 iterations               |
| Precision                          | BF16 computation, FP32 master weights, BF16 gradient reduction                                                                          |
| EMA                                | Power exponential moving average enabled. Rate 0.1, iteration shift 0                                                                   |
| Memory settings                    | Selective activation checkpointing. Dynamic compilation of the language component. CUDA graphs disabled                                 |
| VAE                                | Temporal compression 4, spatial compression 16, 48 latent channels. Causal video encoding                                               |
| VAE encoding                       | Chunk sizes 24 RGB frames at 480 and 8 at 720. Exact encoding durations 17, 61 and 73 frames                                            |
| VAE warmup                         | 256, 480 and 720 tiers, after 3 iterations                                                                                              |
| Parallelism                        | Parameter-shard degree 8. Context-parallel degree 1. Classifier-free-guidance parallel degree 1                                         |
| Randomness / accumulation          | Trainer seed 0, loader seed 42. Gradient accumulation 1                                                                                 |
| Training limit / logging           | Maximum 150,000 iterations. Reference logging every 50 iterations                                                                       |
| Checkpoints                        | Strict loading. Asynchronous distributed checkpoints every 250 iterations, plus effective 30-minute saves. Timed-save retention 3       |
| Gradient clipping / memory cleanup | Gradient norm limit 1. Explicit garbage collection every 200 iterations                                                                 |

The optimizer selects ten parameter groups: the generation expert, diffusion
time embedding, video-latent input and output projections, audio input and
output projections, audio modality embedding, action input and output
projections, and action modality embedding. Action projection and embedding
groups have a 5× learning-rate multiplier. The other selected groups use 1×.
All other parameters are frozen. Only groups participating in the video/control
forward pass receive gradients.

For the transformed-uniform training-noise distribution, draw `u` uniformly
from `[0, 1]`, then compute
`t = 1 - u - 1.29 × (cos²(πu/2) - 1 + u)`. Apply the resolution's noise shift
`a` as `sigma = a × t / (1 + (a - 1) × t)`. This determines which noise levels
the model sees during training. It is separate from the fixed four-step
inference schedule. The uniform alternative starts with uniform `t` before
the same shift transform.

The learning-rate schedule multiplies the base learning rate. 0.4 is a
multiplier, not a learning rate. Packing emits a variable number of examples.
Measure emitted examples and tokens per optimizer update; the child-loader
fetch sizes are 2 and 1, while the packer can combine multiple fetches.

## Stages 1–3 / steps 2–4. Learn causal generation and distill it

These steps optimize only the generation expert, diffusion time embedding,
video-latent input projection and video-latent output projection. Other model
parameters stay frozen. They retain the reasoner, tokenizer and VAE above.

Shared settings are BF16 computation, FP32 master weights and gradient
reduction, power EMA rate 0.1/shift 0, trainer seed 0, loader seed 42, gradient
accumulation 1, gradient norm limit 1 and garbage collection every 200
iterations. Use full activation checkpointing with model compilation disabled.
VAE latent geometry stays 48 channels, spatial compression 16, temporal
compression 4 and spatial patch size 2. Encode in 24-frame chunks, with exact
encoding durations 17, 61 and 73. Warm up the 480 tier.

| Setting                                      | 2. Real-history causal learning                           | 3. Four-step consistency distillation                              | 4. Generated-history learning                                                                                                            |
| -------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Stream / emitted batch                       | 480 tier. Emit one pair per data-parallel rank            | Same                                                               | 480 tier. One pair per data-parallel rank                                                                                                |
| RGB / latent lengths                         | 9–201 RGB frames aligned as `1+4n`. 3–51 latents          | Same                                                               | Same. Equal rollout length across parameter-shard group                                                                                  |
| Latent chunk / noise shift                   | 1 / 5                                                     | 1 / 5                                                              | 1 / 5                                                                                                                                    |
| Caption / control dropout                    | 0.1 / 0                                                   | 0.1 / 0                                                            | 0 / 0                                                                                                                                    |
| Retained caption mixture                     | 93% structured, 7% same-window dense                      | Same                                                               | Same                                                                                                                                     |
| History                                      | Real past RGB, with differentiable cached keys and values | Same                                                               | Generated past RGB after any real prefix                                                                                                 |
| Optimizer                                    | Fused Adam                                                | Fused Adam                                                         | Separate Fused Adam optimizers for student and score model                                                                               |
| Learning rate                                | `1e-5`                                                    | `2e-6`                                                             | Student `1e-6`. Score model `2e-7`                                                                                                       |
| Adam betas                                   | `(0.9, 0.999)`                                            | `(0, 0.999)`                                                       | `(0, 0.999)`                                                                                                                             |
| Weight decay / epsilon                       | 0.01 / `1e-8`                                             | Same                                                               | Same                                                                                                                                     |
| Schedule                                     | Constant after 0 warmup. Cycle 100,000                    | Constant after 0 warmup. Cycle 10,000                              | Constant linear schedule. Cycle `10^13`. 0 warmup for each optimizer                                                                     |
| Flow distribution / multiplier               | Transformed uniform / 10                                  | Transformed uniform / 100                                          | `uniform` / 10. Student distribution-matching and score-model loss multipliers each 1                                                    |
| Frozen teacher                               | None                                                      | Step 2 model. Guidance 3. 48 noise levels, one-interval transition | Step 1 bidirectional model. Guidance 3                                                                                                   |
| Update cadence                               | One optimizer                                             | One optimizer                                                      | One student update per five score-model updates: a six-update cycle                                                                      |
| Student sampling                             | No fixed four-step schedule                               | Four-step stochastic schedule `[1, 15/16, 5/6, 5/8]`, then zero    | Same schedule for generated-history rollout                                                                                              |
| Replay and parameter sharding                | Recompute target only on the second pass                  | Same. Defer parameter resharding                                   | Replay generated history one latent at a time. Exclude conditioned frames. Keep student parameters gathered and defer teacher resharding |
| Training history cache                       | Real teacher-forced history                               | Same                                                               | Window 51 latent times with the first 1 pinned as a sink. Unquantized cache                                                              |
| Maximum iterations                           | 100,000                                                   | 10,000                                                             | 10,000                                                                                                                                   |
| Default logging interval                     | 200                                                       | 10                                                                 | 10                                                                                                                                       |
| Iteration checkpoint interval                | 500                                                       | 250                                                                | 250, with asynchronous saving                                                                                                            |
| Timed checkpoint interval                    | 30 minutes. Retain latest completed timed save            | Same                                                               | Same                                                                                                                                     |
| Parameter / context / guidance shard degrees | 8 / 1 / 1                                                 | Same                                                               | Same                                                                                                                                     |

Step 2 applies supervised rectified-flow loss to target RGB, using real RGB as
clean history. Control queries see controls through the current time and RGB
strictly before it. Target queries also see their own current noisy target.
Future controls and future RGB remain masked.

Step 3 first uses a frozen teacher to move one interval toward lower noise.
The student matches its clean-endpoint prediction against its own stopped-gradient
prediction at that lower-noise point. Apply loss only to generated RGB
positions. The teacher performs one transition on the 48-level grid per example.

Step 4 trains a separate **score model** on generated samples with the
rectified-flow objective. Its estimate of the student's generated distribution
is compared with the frozen teacher's estimate to update the student. This is
why the score model and student need separate optimizers and update counts.
The initialized student is copied into student EMA. Load the score model
independently from the teacher checkpoint to retain FP32 training parameters.
The frozen teacher itself receives no optimizer updates.

For teacher guidance in steps 3 and 4, positive and negative branches share
controls, history, noise, and timesteps. Use the JSON-serialized default negative
caption from `cosmos_framework/inference/defaults/neg_prompts.json`. Append the
actual clip duration, FPS, and dimensions to the negative branch, even
when dropout removes the positive caption and its metadata. The stored four-step
time list is already shifted. Changing an ordinary sampler's step count or
shift does not change this distilled schedule.

## Batch arithmetic and stopping

For steps 2–4 at the reference context-parallel degree 1:

```text
examples per update = one emitted pair × data-parallel world size × accumulation
```

Data-parallel world size counts both parameter-shard and replication ranks.
Keep the world size divisible by the parameter-shard degree. Stage 3 requires
context-parallel degree 1, one pair per rank, equal rollout lengths and
synchronized prefix lengths. These keep forward/backward schedules aligned
across ranks. Count student and score-model updates separately.

Stage 0 uses a 74,000-token packing limit and a variable number of examples.
Measure examples and tokens per update when changing data, topology or
accumulation. The local adapter uses zero workers and prefetch factor 1.
Select a checkpoint using held-out outputs. Follow the
[checkpoint guide](checkpoint-and-resume.md) for restart and inference handoffs.

## Check intermediate training models

Use the sampling profile corresponding to the checkpoint's training stage.
Keep control clips, captions and output geometry fixed when comparing
checkpoints.

| Training output                        | Inference profile                                          | Text / control guidance                                               | History and negative text                                                                                        |
| -------------------------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Whole-window bidirectional model       | Rectified flow with UniPC, 50 steps. Shift 5 at 480        | 3 / 1.5 for edge, blur and depth. Control guidance 2 for segmentation | Joint window generation, no causal cache. Matched serialized negative caption. No negative metadata              |
| Causal model trained with real history | Rectified flow with UniPC, 50 steps. Shift 5 at 480        | 3 / 1                                                                 | Generated history, one latent per chunk, full history/no sink, zero real RGB prefix. Empty unconditional caption |
| Four-step consistency model            | Saved stochastic schedule `[1, 15/16, 5/6, 5/8]` then zero | 1 / 1                                                                 | Generated history, one latent per chunk, full history/no sink, zero real RGB prefix. No negative branch          |
| Stage 3 generated-history model        | Same saved four-step schedule                              | 1 / 1                                                                 | Same causal input and history policy. Final training used window 51/sink 1                                       |

Compare 121-frame, native 480-tier outputs using the same aligned controls,
window-matched text and effective geometry. The bidirectional reference starts
its first chunk without RGB prefix and uses one conditional overlap frame for
subsequent video chunks. Causal reference generation starts with zero real RGB
prefix. Request native-size output when comparing checkpoints.
