<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: OpenMDW-1.1 -->

# Sim-Bimanual — Run the Cosmos3 Two-Arm Neural Simulator on Held-Out AgiBot Data

> **Authors:** [Saeed Babamohamadi](https://www.linkedin.com/in/saeedbabamohamadi)
> **Organization:** [NVIDIA](https://nvidia.com/)

Run the released `Cosmos3-Nano-Sim-Bimanual` checkpoint as an action-conditioned neural simulator on public
AgiBot episodes it was not trained on, measure how fast it runs, and measure how closely it follows the real
recording when it is given the real actions, against two controls.

![Held-out episode 815362 of task 600 (hang the key on the hook and place the bag in the storage box) at frames 30, 60 and 900. Columns: real video, matched-action rollout, frozen first frame](assets/sim-bimanual/hero_task600_ep815362_f30_f60_f900.png)
*Held-out AgiBotWorld-Beta episode 815362, task 600. Rows: frames 30, 60, 900. Columns: the real head camera; the
simulator driven by the episode's recorded actions from its real first frame; the frozen-first-frame control. By frame
900 the rollout has moved both arms as the real episode does and a tag hangs on the hook, but the bag at the bottom
right is gone. [GIF](assets/sim-bimanual/hero_task600_ep815362_strip.gif), [captions](assets/sim-bimanual/captions.md).*

**Who this is for.** You have a two-arm AgiBot G1 policy and a real first frame from an AgiBot episode or your own
AgiBot workcell. The policy proposes actions, the simulator shows what the head camera would see, and you watch where
it goes wrong. That is a preview for debugging and demos. It is not a scoreboard: nobody has shown that this
simulator ranks policies the way a physics simulator does, so do not use it to choose between checkpoints.

| Model                              | Workload       | Use case                                                                                                                                     |
| ---------------------------------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `nvidia/Cosmos3-Nano-Sim-Bimanual` | Inference only | Given a start frame and a stream of two-arm actions (schema 5, `normalized_unified_v1`), generate what the head camera would see, 901 frames |

## What You'll Build

- A **converter** from a raw AgiBotWorld-Beta episode to the checkpoint's schema-5 action file, checked against the
  framework's own data loader before any rollout
- A **fidelity check**: 20 held-out episodes rolled out from their real first frame with their real actions, scored
  against the real video and against two controls (a frozen first frame, and the same episodes driven by another
  task's actions)
- A **cost card**: seconds per 901-frame rollout, frames per second, peak memory, on one GB300

## Pipeline

| Stage | What happens                                                                                   | GPU    |
| ----- | ---------------------------------------------------------------------------------------------- | ------ |
| 1     | Install the framework, download the checkpoint, run its own input check and AgiBot example     | 80 s   |
| 2     | Download three AgiBotWorld-Beta tars, extract 20 held-out episodes, convert them, run the gate | no     |
| 3     | 20 matched-action rollouts and 20 action-swap rollouts; build the frozen-frame control         | 35 min |
| 4     | Score the three against the real video; tables, plot and review page                           | 5 min  |

## Prerequisites

| Requirement                                                                                                                                                                                               | Why                                                       |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| `cosmos-framework` at `86e3528d791446ac9928bfff8432951b23d8bf47` (`main`, 2026-11-06), set up as [docs/inference_bimanual.md](../../../docs/inference_bimanual.md) describes                              | the checkpoint loads on `main`; the measurements ran here |
| Read access to [`nvidia/Cosmos3-Nano-Sim-Bimanual`](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual), revision `a9e3adf6235f56f0082be7a96e497348c286c472`                                         | the checkpoint, 30 GB on disk                             |
| Access to [`agibot-world/AgiBotWorld-Beta`](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta) (gated; accept the terms on Hugging Face), revision `2252b3fdc88dc67fa5d5e4b771e1b86f5e71c278` | the held-out episodes, three tars of about 48.5 GB each   |
| `ffmpeg` and `ffprobe` on `PATH`                                                                                                                                                                          | the converter cuts the real clips                         |

### GPU you need

One GPU with about 50 GB free: the rollouts peaked at 39.95 GB allocated and 47.94 GB reserved by torch, 49,973 MiB on
the whole GPU (one GB300). An 80 GB GPU should fit; that was not measured. Disk: about 180 GB (checkpoint and the three
dataset tars; the tars can be deleted after extraction).

## Dataset

| Name                                   | Source                                                                                                                                                                                                            | Licence                                                                                                                                                                                                              |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| AgiBotWorld-Beta, G1 bimanual episodes | [`agibot-world/AgiBotWorld-Beta`](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta) on Hugging Face, downloaded by each user from the publisher; revision `2252b3fdc88dc67fa5d5e4b771e1b86f5e71c278` | CC BY-NC-SA 4.0. This recipe does not bundle or rehost any file from it. Each user is responsible for checking the content of datasets and the applicable licenses and determining if suitable for the intended use. |

The checkpoint takes schema-5 `normalized_unified_v1` actions: 59 values per frame (padded to 64), already normalised
and masked by the source, because inference does not normalise again. For an AgiBot episode the valid slots are 0–17
(head and right wrist), 33 (right gripper), 34–42 (left wrist) and 58 (left gripper); every other slot is exactly zero.
[docs/inference_bimanual.md](../../../docs/inference_bimanual.md) is the authority for the input contract.

## Splits

| Split            | Content                                                                                                                                                                                                                                                       |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| bundled examples | the examples shipped with the checkpoint (smoke test only)                                                                                                                                                                                                    |
| held-out         | 20 AgiBotWorld-Beta episodes (10 of task 600, 10 of task 596) from tasks the bundled examples do not come from, listed with their tars and selection rule in [`assets/sim-bimanual/manifests/heldout_20.json`](assets/sim-bimanual/manifests/heldout_20.json) |

Fidelity on bundled examples is not a result. Only held-out episodes count. The 10 per task were taken by even spread
over the sorted eligible episode ids; the swap pairing is in
[`assets/sim-bimanual/manifests/heldout_swap20_pairs.json`](assets/sim-bimanual/manifests/heldout_swap20_pairs.json).

## Commands, one GB300

Every command runs from the framework root with the venv active. The scripts ship in this folder; `S` points at them.

```bash
# 1. Framework, checkpoint, the release's own checks (docs/inference_bimanual.md; the train extra adds mujoco 3.10.0 for the converter)
git checkout 86e3528d791446ac9928bfff8432951b23d8bf47 && git lfs install --local
uvx --from 'uv==0.11.29' uv lock --check
uvx --from 'uv==0.11.29' uv sync --locked --group cu130 --extra guardrail --extra train
source .venv/bin/activate && export COSMOS_TRAINING=0
export S=$PWD/.agents/skills/cosmos3-sim-bimanual-agibot/assets/sim-bimanual
export SIM_BIMANUAL_CHECKPOINT=$PWD/checkpoints/Cosmos3-Nano-Sim-Bimanual COSMOS_FRAMEWORK_ROOT=$PWD
hf download nvidia/Cosmos3-Nano-Sim-Bimanual --revision a9e3adf6235f56f0082be7a96e497348c286c472 --local-dir $SIM_BIMANUAL_CHECKPOINT
python tools/check_bimanual_mixed_inputs.py --checkpoint-path $SIM_BIMANUAL_CHECKPOINT   # prints passed: true
# then run the AgiBot 001 example exactly as docs/inference_bimanual.md "Run on one GPU" shows: 901 frames, 640x352, 30 fps

# 2. Held-out episodes (accept the dataset terms first). The 20 ids and the three tars are in $S/manifests/heldout_20.json.
R="--repo-type dataset --revision 2252b3fdc88dc67fa5d5e4b771e1b86f5e71c278 --local-dir heldout/raw"
hf download agibot-world/AgiBotWorld-Beta proprio_stats/804068-856282.tar    $R
hf download agibot-world/AgiBotWorld-Beta observations/600/815362-842380.tar $R
hf download agibot-world/AgiBotWorld-Beta observations/596/810449-822244.tar $R
mkdir -p heldout/extracted   # one video and one state file per episode, for example:
tar -xf heldout/raw/observations/600/815362-842380.tar -C heldout/extracted 815362/videos/head_color.mp4
tar -xf heldout/raw/proprio_stats/804068-856282.tar    -C heldout/extracted 600/815362/proprio_stats.h5
python $S/tools/convert_agibot_episode.py --task-id 600 --name task600_ep815362 --start 0 --num-actions 900 \
  --h5 heldout/extracted/600/815362/proprio_stats.h5 --video heldout/extracted/815362/videos/head_color.mp4 \
  --out heldout/inputs/task600
# ... the same for the other 19 episodes (task 596 rows go to heldout/inputs/task596), then the fixed prompts:
python $S/tools/make_manifests.py prompts --inputs heldout/inputs --prompts $S/manifests/prompts_heldout_20.json
# Gate, before any rollout: the converter against the framework's own data loader on the checkpoint's training bundle
mkdir -p heldout/bundle && tar --zstd -xf $SIM_BIMANUAL_CHECKPOINT/examples/training/sim_bimanual_robot_example_v1.tar.zst -C heldout/bundle
B=$(dirname "$(find heldout/bundle -name samples.json | head -1)")   # the bundle directory holds samples.json
python $S/tools/check_conversion_gate.py --bundle $B --out heldout/conversion_gate.json   # exit 0, "pass": true

# 3. Rollouts: the documented command with -i/-o changed and seed 42 (each row carries guidance 3.0 and shift 5.0)
FLAGS="--checkpoint-path $SIM_BIMANUAL_CHECKPOINT --model-mode=FORWARD_DYNAMICS --resolution=480 --image-size=480 \
  --num-steps=4 --seed=42 --kv-cache-inference-size=64 --attention-sink-size=8 --autoregressive=True \
  --parallelism-preset=latency --max-num-seqs=1 --no-use-torch-compile --no-diffusion-cache --no-guardrails --video-save-quality=4"
for t in 600 596; do
  torchrun --standalone --nproc-per-node=1 -m cosmos_framework.scripts.inference \
    -i heldout/inputs/task$t/task$t.jsonl -o outputs/cosmos3_nano_simulation_humanoid_heldout $FLAGS
done
python $S/tools/make_manifests.py swap   --inputs heldout/inputs --out heldout/inputs/swap20/heldout_swap20.jsonl
torchrun --standalone --nproc-per-node=1 -m cosmos_framework.scripts.inference \
  -i heldout/inputs/swap20/heldout_swap20.jsonl -o outputs/cosmos3_nano_simulation_humanoid_heldout_swap20 $FLAGS
python $S/tools/make_manifests.py frozen --inputs heldout/inputs --out heldout/frozen   # no rollout: real frame 0 held for 901 frames

# 4. Score all three against the real video, then the review page
python $S/tools/make_manifests.py eval --inputs heldout/inputs --outputs outputs --frozen heldout/frozen --out heldout/eval
for g in heldout:matched frozen:frozen swap20_vs_real:swap20_vs_real; do
  python $S/tools/eval_fidelity.py --batch heldout/eval/entries_${g%%:*}.jsonl --out heldout/eval/${g##*:} --align auto --device cuda
done
python $S/tools/make_review_page.py --entries heldout/eval/review_entries.jsonl --out heldout/eval/review/index.html \
  --aggregate heldout/eval/matched/aggregate_summary.json --aggregate-plot heldout/eval/matched/aggregate_drift.png
```

What each script reads and writes is in [`assets/sim-bimanual/tools/README.md`](assets/sim-bimanual/tools/README.md).
Inspect every rollout receipt (`sample_outputs.json`): all 40 must say `status: success`.

## Settings

| Setting         | Value                                                                                                            |
| --------------- | ---------------------------------------------------------------------------------------------------------------- |
| Model mode      | forward dynamics, autoregressive, KV cache 64, attention sink 8                                                  |
| Denoising steps | 4 (the release's four-step schedule)                                                                             |
| Guidance, shift | 3.0, 5.0 in every row, as in the published example rows                                                          |
| Resolution      | 640×360 first frame in, 640×352 out (the generator keeps the top-left 22×40 latents of the padded canvas)        |
| Length          | 900 actions, 901 frames at 30 fps                                                                                |
| Seed            | 42 on every rollout in the tables                                                                                |
| Actions         | schema 5, `normalized_unified_v1`, global-asinh statistics pinned by the checkpoint, scalar domain `agibotworld` |
| Prompt          | fixed per episode, [`manifests/prompts_heldout_20.json`](assets/sim-bimanual/manifests/prompts_heldout_20.json)  |

## Results

Measured on one GB300 with the checkpoint at `a9e3adf6235f56f0082be7a96e497348c286c472`: 20 held-out AgiBotWorld-Beta
episodes, 10 each from two tasks, mean ± std over episodes, real video cropped to its top 352 rows. Every number below
is in [`assets/sim-bimanual/results/metrics.json`](assets/sim-bimanual/results/metrics.json); the run record is
[`assets/sim-bimanual/manifests/RECORD.md`](assets/sim-bimanual/manifests/RECORD.md).

### What the numbers look like

| ![Task 600, episode 815362: real, matched rollout and frozen control at frames 30 and 900](assets/sim-bimanual/stills_task600_ep815362_f30_f900.png) | ![Task 596, episode 810449: real, matched rollout and frozen control at frames 30 and 900](assets/sim-bimanual/stills_task596_ep810449_f30_f900.png) |
| ---------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| task 600, episode 815362: real, matched rollout, frozen control at frame 30 and frame 900                                                            | task 596, episode 810449 (place the inner pot with rice into the rice cooker): the same three at frame 30 and frame 900                              |

The "top 352 rows" crop keeps rows 0 to 351 of the 360-row real frame, because the generator outputs 352 rows and keeps
the top of its input. Frame 0 of every generated video scores 32.9 to 36.7 dB against that crop (the autoencoder
round-trip of the conditioning frame), while a centre or bottom crop gives 16.2 to 23.4 dB. If your alignment check gives
the lower numbers, the crop is wrong.

### Speed and memory

| Run                                     | Rollouts | Wall incl. model load | Seconds per 901-frame rollout | Frames/s | Peak GPU (whole device) |
| --------------------------------------- | -------- | --------------------- | ----------------------------- | -------- | ----------------------- |
| bundled AgiBot 001 example (seed 2026)  | 1        | 80 s                  | —                             | —        | 43,234 MiB              |
| held-out, task 600, matched             | 10       | 509 s                 | 48–62 (all 40: mean 51.1)     | 17.6     | 49,967 MiB              |
| held-out, task 596, matched             | 10       | 506 s                 |                               |          |                         |
| held-out, cross-task action swap        | 20       | 1070 s                |                               |          | 49,973 MiB              |

Peak torch memory over the 40 rollouts: 39.95 GB allocated, 47.94 GB reserved. At these settings one GB300 generates
about 17.6 frames per second, not real time: a 901-frame rollout plays back in 30 s and takes about 51 s to generate.

### Fidelity over the rollout (matched real actions against two controls)

| Comparison                                | Metric  | frame 30     | frame 60     | frame 120    | frame 240    | frame 480    | frame 900    |
| ----------------------------------------- | ------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| matched actions                           | PSNR dB | 24.88 ± 2.17 | 18.65 ± 1.98 | 19.42 ± 1.89 | 20.16 ± 1.94 | 18.72 ± 2.08 | 17.57 ± 2.06 |
|                                           | SSIM    | 0.893        | 0.803        | 0.791        | 0.776        | 0.723        | 0.680        |
|                                           | LPIPS   | 0.076        | 0.165        | 0.166        | 0.158        | 0.202        | 0.261        |
| frozen first frame (control)              | PSNR dB | 34.60 ± 8.20 | 23.18 ± 9.25 | 16.75 ± 2.37 | 15.46 ± 2.27 | 13.73 ± 2.18 | 12.86 ± 2.00 |
|                                           | SSIM    | 0.945        | 0.845        | 0.792        | 0.741        | 0.652        | 0.604        |
|                                           | LPIPS   | 0.042        | 0.136        | 0.201        | 0.251        | 0.359        | 0.426        |
| wrong actions, same 20 episodes (control) | PSNR dB | 22.83 ± 3.57 | 15.91 ± 2.10 | 14.55 ± 1.67 | 14.88 ± 2.16 | 13.37 ± 1.47 | 12.73 ± 1.45 |
|                                           | SSIM    | 0.868        | 0.732        | 0.708        | 0.679        | 0.602        | 0.556        |
|                                           | LPIPS   | 0.102        | 0.256        | 0.295        | 0.318        | 0.413        | 0.480        |

Window means (frames 0–299 | 300–599 | 600–900 | overall):

| Comparison                                                                                                      | PSNR dB                                 | SSIM                             | LPIPS                            |
| --------------------------------------------------------------------------------------------------------------- | --------------------------------------- | -------------------------------- | -------------------------------- |
| matched, 20 episodes                                                                                            | 21.15 \| 19.07 \| 17.67 \| 19.30 ± 1.58 | 0.804 \| 0.730 \| 0.689 \| 0.741 | 0.148 \| 0.196 \| 0.249 \| 0.198 |
| matched, task 600 (10)                                                                                          | 21.55 \| 20.27 \| 19.01 \| 20.28        | 0.824 \| 0.799 \| 0.767 \| 0.797 | 0.149 \| 0.186 \| 0.229 \| 0.188 |
| matched, task 596 (10)                                                                                          | 20.76 \| 17.87 \| 16.32 \| 18.32        | 0.784 \| 0.661 \| 0.610 \| 0.685 | 0.148 \| 0.206 \| 0.269 \| 0.208 |
| frozen first frame (20)                                                                                         | 20.03 \| 13.92 \| 12.91 \| 15.62 ± 2.29 | 0.799 \| 0.662 \| 0.610 \| 0.690 | 0.194 \| 0.347 \| 0.417 \| 0.320 |
| wrong actions, same 20 episodes (each task-600 episode driven by a task-596 episode's actions, and the reverse) | 16.81 \| 13.67 \| 12.55 \| 14.34 ± 1.20 | 0.728 \| 0.613 \| 0.548 \| 0.630 | 0.267 \| 0.398 \| 0.494 \| 0.387 |

Full tables with standard deviations on every cell and the per-task rows:
[`results_tables_heldout.md`](assets/sim-bimanual/results_tables_heldout.md) (matched and frozen) and
[`results_tables_swap20.md`](assets/sim-bimanual/results_tables_swap20.md) (action swap, with its drift plot
[`aggregate_drift_swap20.png`](assets/sim-bimanual/aggregate_drift_swap20.png)).

Reading. All three rows are the same 20 episodes. Overall, the matched rollout beats the frozen first frame by 3.68 dB
(paired, 0.83 to 5.77 dB, better on 20 of 20 episodes) and the action swap by 4.96 dB (3.46 to 7.88 dB, 20 of 20), so
the output follows the actions it is given and not only the first frame. Early in the rollout the frozen frame is ahead,
because little has moved yet: the per-frame mean of the matched rollout passes the frozen frame at frame 88 and stays
above it to frame 900 (+4.7 dB at frame 900). Fidelity decays from 24.9 dB at frame 30 to 17.6 dB at frame 900, and the
two tasks differ by 2.0 dB overall. These are fidelity numbers only; whether the simulator ranks policies like a
physics simulator has not been measured.

### How you know it worked

- `check_bimanual_mixed_inputs.py` prints `passed: true`, the AgiBot 001 example gives 901 frames at 640×352, and
  `check_conversion_gate.py` exits 0.
- Frame 0 of every rollout scores above 30 dB against the top-352-row crop of the real clip.
- The matched rollout beats the action-swap control overall and the frozen first frame overall, and stays above the
  frozen frame from about frame 90 to the end.

## Prompt and agent guide

**Prompts for the simulator.** Each rollout takes the first frame, the action file and one text prompt. The 20 prompts
in [`manifests/prompts_heldout_20.json`](assets/sim-bimanual/manifests/prompts_heldout_20.json) were written from each
episode's first frame and the dataset's task text before any generation. Each describes the starting scene (objects,
surfaces, camera), the two arms and the task, and never narrates what happens later. The action-swap rollouts keep
the frame episode's prompt, so the two controls isolate the actions. Whether a different prompt changes fidelity was not
measured. The model's general prompt guidance is in the
[Sim-Bimanual prompt guide](../cosmos3-nano-sim-bimanual/shared/prompt-guide.md).

**Running this recipe with a coding agent.**

- Pin everything the agent cannot guess: framework commit, model and dataset revisions, the command block, the output
  layout. An agent that has to choose a version will choose a different one each run.
- Give it a stop rule, a budget in GPU-hours and the checks under "How you know it worked".
- Make it write a record per run (data, split, GPU, peak memory, GPU-hours, result, controls, leakage check) before it
  reports. A number without a record is not a result.
- For this recipe specifically: the action file must already be normalised and masked (inference normalises nothing);
  run the conversion gate before any rollout; check frame-0 alignment above 30 dB before scoring anything; keep the
  bundled examples out of the held-out set. Each of these caught a real mistake during this work.

## Known limitations

- **Embodiment.** The converter handles AgiBot G1 episodes with the `robotiq` gripper calibration (every episode of
  tasks 600 and 596); other AgiBot grippers need the matching audited calibration (`--gripper-hardware`). Do not feed
  it LIBERO or SO-101 actions.
- **The literal sidecar gate is not possible.** The release's training bundle holds three different episodes from the
  ones behind its published `agibot_{001,006,034}` action sidecars (different scenes and camera-mask spans), so the
  converter was checked against the framework's data loader on the bundle (max difference 0.0) and the published
  sidecars were checked for structure only. See [`manifests/RECORD.md`](assets/sim-bimanual/manifests/RECORD.md).
- **Not a policy evaluator yet.** Whether this simulator ranks policy checkpoints the way a physics simulator does has
  not been measured. Use it for previewing and debugging a policy, not for deciding between checkpoints.
- **Not a data generator yet.** No policy has been post-trained on this simulator's output here, so no claim is made
  about its value as training data.
- **Not real time at these settings:** about 17.6 generated frames per second on one GB300.
- **Dataset licence.** AgiBotWorld-Beta is CC BY-NC-SA 4.0 (see Dataset). This recipe points at the publisher's
  download and ships no data; the media in this folder are derivative works under the same licence (see
  [captions](assets/sim-bimanual/captions.md)).

## Document Information

|             |                                                                                                                                                                              |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Framework   | `cosmos-framework` `86e3528d791446ac9928bfff8432951b23d8bf47` (`main`, 2026-11-06); inference code unchanged, the recipe's scripts are in this folder                        |
| Model       | `nvidia/Cosmos3-Nano-Sim-Bimanual`, revision `a9e3adf6235f56f0082be7a96e497348c286c472` (schema 5)                                                                           |
| Data        | `agibot-world/AgiBotWorld-Beta`, revision `2252b3fdc88dc67fa5d5e4b771e1b86f5e71c278`, CC BY-NC-SA 4.0, user-downloaded, not bundled; episodes in `manifests/heldout_20.json` |
| Tools       | sha256 and line counts in [`assets/sim-bimanual/tools/README.md`](assets/sim-bimanual/tools/README.md)                                                                       |
| Measured on | 1× GB300, 2026-11-09                                                                                                                                                         |
