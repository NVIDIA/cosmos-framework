<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: OpenMDW-1.1 -->

# Sim-Bimanual — Run the Cosmos3 Real-Time Two-Arm Neural Simulator on AgiBot Data

> **Authors:** [Saeed Babamohamadi](https://www.linkedin.com/in/saeedbabamohamadi)
> **Organization:** [NVIDIA](https://nvidia.com/)
> **Status:** draft v0.3 (skill folder), 2026-10-15; numbers unchanged from v0.2 of 2026-10-05. All numbers measured on 1× GB300; records committed. What each input is backed by is in [CHECKLIST.md](CHECKLIST.md).

Run the `Cosmos3-Nano-Sim-Bimanual` checkpoint (early access) as an action-conditioned neural simulator on
public AgiBot bimanual data, measure how fast it runs and how closely it follows real episodes, and learn
what it can and cannot be used for today.

![Held-out episode 815362 of task 600 (hang the key on the hook and place the bag in the storage box) at frames 30, 60 and 900. Columns: real video, matched-action rollout, frozen first frame](assets/sim-bimanual/hero_task600_ep815362_f30_f60_f900.png)
*Held-out AgiBotWorld-Beta episode 815362, task 600. Rows: frames 30, 60, 900. Columns: real head camera; the simulator driven by the episode's real actions from its real first frame; the frozen-first-frame control. By frame 60 the rollout has moved with the actions and the frozen frame has not. [Strip video](assets/sim-bimanual/hero_task600_ep815362_strip.mp4), [gif](assets/sim-bimanual/hero_task600_ep815362_strip.gif), [captions](assets/sim-bimanual/captions.md).*

**Who this is for, today.** You have a two-arm AgiBot policy that outputs the 29-D AgiBot action
schema, and a real first frame from an AgiBot episode or your own AgiBot workcell. The policy
proposes actions, the simulator shows what the cameras would see, the policy reacts, and you watch
where it goes wrong. That is a preview for debugging and demos. It
is not a scoreboard: nobody has shown that this simulator ranks policies the way a physics simulator
does, so do not use it to choose between checkpoints. It is also not a data generator yet: no policy
has been post-trained on its output in this program, so that use is not claimed here. When the
post-training recipe for the simulator is released and a post-training result on its generated data
exists, both will be added.

| Model                                             | Workload       | Use case                                                                                      |
| ------------------------------------------------- | -------------- | --------------------------------------------------------------------------------------------- |
| `nvidia/Cosmos3-Nano-Sim-Bimanual` (early access) | Inference only | Given a start frame and a stream of 29-D two-arm actions, generate what the cameras would see |

## What You'll Build

- A **running neural simulator** for the AgiBot G1 two-arm embodiment on one GB300
- A **cost card**: frames per second, memory, time per action chunk
- A **fidelity check**: generated video against the real episode for the same actions, on held-out episodes
- A clear statement of **what the checkpoint does not accept**: it is trained for the AgiBot action
  schema only. It does not accept single-arm robots or other bimanual action spaces.

## Pipeline

| Stage | What happens                                                                                | GPU                       |
| ----- | ------------------------------------------------------------------------------------------- | ------------------------- |
| 1     | Download the checkpoint and an AgiBot episode set; inspect the action schema                | no                        |
| 2     | Run the bundled examples; confirm output resolution and frame rate                          | minutes                   |
| 3     | Roll out held-out episodes from their first frame with their real actions; measure fidelity | about 1 h per 20 episodes |

## Prerequisites

The checkpoint does not load on the public framework at `c23e51f2f157ae3e51cfcd86ebfb5464850894f2` (its model alias is missing there). The numbers
below were measured on framework revision `573d46507dcd91155d7f371b366da760a7a5f718` and cookbook revision `190b6565726fea4695ad168ee6bf197d30ca7c88` (early access)
(`cookbooks/cosmos3/simulation/bimanual/bimanual.md`), both of which ship publicly with the checkpoint at GTC Berlin.
Until then, access to those two repositories is required. The commands are the cookbook's, unchanged except where noted.

Revision reconciliation. Every run record of this recipe (run01 to run10, 2026-09-30 to 2026-10-07) names the same two revisions, framework `573d46507dcd91155d7f371b366da760a7a5f718` and cookbook `190b6565726fea4695ad168ee6bf197d30ca7c88`; no number here was measured on the public `c23e51f2f157ae3e51cfcd86ebfb5464850894f2` (the LIBERO recipes' pin) or on any other revision. The public revision of the bimanual documentation that will ship with the checkpoint did not exist when this was written, so it cannot be named; when it does, run step 2 on it and confirm 640×352, 901 frames and `status: success` before trusting the numbers below on that revision.

| Requirement                                                                                                                                                                                               | Why                                    |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| 1× GB300 (measured); 1× H100 80 GB is a fit estimate from the 58.5 GB peak, not measured                                                                                                                  | the checkpoint is 62 GB on disk        |
| Hugging Face read access to `nvidia/Cosmos3-Nano-Sim-Bimanual`, revision `1d95a0b5b19d49a24aceebf578cf5e85db310ac0` (early access)                                                                        | checkpoint is private until GTC Berlin |
| Access to [`agibot-world/AgiBotWorld-Beta`](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta) (gated; accept the terms on Hugging Face), revision `2252b3fdc88dc67fa5d5e4b771e1b86f5e71c278` | the held-out episodes                  |

### GPU you need

One GPU with at least 64 GB (peak 58.5 GB measured on a GB300; an 80 GB card should fit, not measured). The 20
held-out rollouts take under an hour; the checkpoint download is 58 GB.

## Dataset

| Name                                   | Source                                                                                                                                                                                                            | Licence                                                                                                                                                                                                              |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| AgiBotWorld-Beta, G1 bimanual episodes | [`agibot-world/AgiBotWorld-Beta`](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta) on Hugging Face, downloaded by each user from the publisher; revision `2252b3fdc88dc67fa5d5e4b771e1b86f5e71c278` | CC BY-NC-SA 4.0. This recipe does not bundle or rehost any file from it. Each user is responsible for checking the content of datasets and the applicable licenses and determining if suitable for the intended use. |

Action schema the checkpoint expects, from its configuration: embodiments `agibotworld` and
`agibot_gear_gripper` (29-D actions), plus `camera_pose` and `hand_pose` streams. The checkpoint's model
card was a placeholder at the time of testing; the configuration file is the authority.

## Splits

| Split            | Content                                                                                                                                                                                                                                                                                  |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| bundled examples | the clips shipped with the checkpoint (in-distribution; useful for a smoke test only)                                                                                                                                                                                                    |
| held-out         | AgiBot episodes not used in the checkpoint's post-training, selected from the public set: the 20 episodes (10 of task 600, 10 of task 596) listed with their tars and selection rule in [`assets/sim-bimanual/manifests/heldout_20.json`](assets/sim-bimanual/manifests/heldout_20.json) |

Fidelity on bundled examples is not a result. Only held-out episodes count.

## Commands, one GB300

The five `tools/` scripts live in [`assets/sim-bimanual/tools/`](assets/sim-bimanual/tools/) (README of inputs, outputs and dependencies there). Copy that folder to `tools/`, and [`assets/sim-bimanual/manifests/`](assets/sim-bimanual/manifests/) to `manifests/`, next to the cookbook checkout before running the block below; every command runs from the framework root with the venv active.

```bash
# 1. environment and checkpoint (early access: framework revision 573d46507dcd91155d7f371b366da760a7a5f718; see Prerequisites)
uv sync --all-extras --group=cu130-train && source .venv/bin/activate
uv pip install mujoco   # 3.14.0 used; the AgiBot forward kinematics in step 3 need it
hf download nvidia/Cosmos3-Nano-Sim-Bimanual --revision 1d95a0b5b19d49a24aceebf578cf5e85db310ac0 --local-dir ./Cosmos3-Nano-Sim-Bimanual   # 58 GB, about 45 s
# The cookbook says `export LD_LIBRARY_PATH=`; torchcodec then fails on libnppicc.so.13. Expose the venv's CUDA 13 runtime instead:
export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.13/site-packages/nvidia/cu13/lib

# 2. bundled 901-frame rollout, the cookbook's command (these flags are "the step-2 flags" below)
torchrun --standalone --nproc-per-node=1 -m cosmos_framework.scripts.inference   -i inputs/omni/agibot.jsonl -o outputs/cosmos3_nano_simulation_humanoid   --checkpoint-path ./Cosmos3-Nano-Sim-Bimanual --model-mode=FORWARD_DYNAMICS   --resolution=480 --image-size=480 --action-chunk-size=900 --num-frames=901 --num-steps=4   --autoregressive=True --parallelism-preset=latency --max-num-seqs=1   --no-use-torch-compile --no-diffusion-cache --no-guardrails --video-save-quality=4
# 61-frame variant: truncate each action array to 60 rows; --action-chunk-size=60 --num-frames=61

# 3. held-out episodes: accept the dataset terms on Hugging Face first (gated)
hf download agibot-world/AgiBotWorld-Beta --repo-type dataset --revision 2252b3fdc88dc67fa5d5e4b771e1b86f5e71c278 --include "task_info/*" --local-dir heldout
# To reproduce the measured set, skip the selector: the 20 episode ids, the two task ids and the three tars are in
# assets/sim-bimanual/manifests/heldout_20.json. To select a fresh set, this is the command the record shows
# (its exclusion file, 31 strings = the bundled examples' captions and task names, ships as assets/sim-bimanual/manifests/exclusions.json):
python tools/select_episodes.py --task-info-dir heldout/task_info --n 20 --min-frames 901   --exclude-text manifests/exclusions.json --fetch-index --max-obs-tars 2 --min-tasks 2 --out heldout/selection.json
# The three tars of the measured set, one per call, 48.4 to 48.9 GB each:
hf download agibot-world/AgiBotWorld-Beta proprio_stats/804068-856282.tar     --repo-type dataset --revision 2252b3fdc88dc67fa5d5e4b771e1b86f5e71c278 --local-dir heldout/raw
hf download agibot-world/AgiBotWorld-Beta observations/600/815362-842380.tar  --repo-type dataset --revision 2252b3fdc88dc67fa5d5e4b771e1b86f5e71c278 --local-dir heldout/raw
hf download agibot-world/AgiBotWorld-Beta observations/596/810449-822244.tar  --repo-type dataset --revision 2252b3fdc88dc67fa5d5e4b771e1b86f5e71c278 --local-dir heldout/raw
# Extract one member per selected episode (members: <episode>/videos/head_color.mp4 and <task>/<episode>/proprio_stats.h5), e.g.:
tar -xf heldout/raw/observations/600/815362-842380.tar -C heldout/extracted 815362/videos/head_color.mp4
tar -xf heldout/raw/proprio_stats/804068-856282.tar    -C heldout/extracted 600/815362/proprio_stats.h5
# Convert each episode to the checkpoint's 29-value normalised action layout (framework forward kinematics plus the schema normaliser).
# Prompt = task name + the episode's first action text from task_info/task_<task>.json; one --out directory per task.
export COSMOS_FRAMEWORK_ROOT=$PWD SIM_BIMANUAL_CHECKPOINT=$PWD/Cosmos3-Nano-Sim-Bimanual
python tools/convert_agibot_episode.py --h5 heldout/extracted/600/815362/proprio_stats.h5 --video heldout/extracted/815362/videos/head_color.mp4   --name task600_ep815362 --prompt "Hang the key on the hook and place the bag in the storage box. Pick up the key on the cabinet with left arm."   --start 0 --num-actions 900 --out heldout/inputs/task600
# ... the same for the other 19 episodes in heldout_20.json (task 596 rows go to --out heldout/inputs/task596); then one JSONL per task:
cat heldout/inputs/task600/task600_ep*.jsonl > heldout/inputs/task600/task600.jsonl
cat heldout/inputs/task596/task596_ep*.jsonl > heldout/inputs/task596/task596.jsonl

# 4a. matched-action rollouts (the measurement): the step-2 flags with -i and -o changed, one process per task
torchrun --standalone --nproc-per-node=1 -m cosmos_framework.scripts.inference   -i heldout/inputs/task600/task600.jsonl -o outputs/cosmos3_nano_simulation_humanoid_heldout   --checkpoint-path ./Cosmos3-Nano-Sim-Bimanual --model-mode=FORWARD_DYNAMICS   --resolution=480 --image-size=480 --action-chunk-size=900 --num-frames=901 --num-steps=4   --autoregressive=True --parallelism-preset=latency --max-num-seqs=1   --no-use-torch-compile --no-diffusion-cache --no-guardrails --video-save-quality=4
# then the same with -i heldout/inputs/task596/task596.jsonl (same -o)

# 4b. frozen-first-frame control: no rollout; the real frame 0 held for 901 frames at the generator's 640x352 geometry
python tools/make_manifests.py frozen --inputs heldout/inputs --out heldout/frozen

# 4c. cross-task action-swap control on the same 20 episodes: i-th task-600 episode paired with the i-th task-596 episode
#     (ids ascending), both directions; frame and prompt from the frame episode, 900 actions from the partner
python tools/make_manifests.py swap --inputs heldout/inputs --out heldout/inputs/swap20/heldout_swap20.jsonl
torchrun --standalone --nproc-per-node=1 -m cosmos_framework.scripts.inference   -i heldout/inputs/swap20/heldout_swap20.jsonl -o outputs/cosmos3_nano_simulation_humanoid_heldout_swap20   --checkpoint-path ./Cosmos3-Nano-Sim-Bimanual --model-mode=FORWARD_DYNAMICS   --resolution=480 --image-size=480 --action-chunk-size=900 --num-frames=901 --num-steps=4   --autoregressive=True --parallelism-preset=latency --max-num-seqs=1   --no-use-torch-compile --no-diffusion-cache --no-guardrails --video-save-quality=4

# 5. score all three against the real video (entries_*.jsonl are {"name","real","gen","prompt"} rows), then the review page
python tools/make_manifests.py eval --inputs heldout/inputs --outputs outputs --frozen heldout/frozen --out heldout/eval
python tools/eval_fidelity.py --batch heldout/eval/entries_heldout.jsonl        --out heldout/eval/matched        --align auto --device cuda
python tools/eval_fidelity.py --batch heldout/eval/entries_frozen.jsonl         --out heldout/eval/frozen         --align auto --device cuda
python tools/eval_fidelity.py --batch heldout/eval/entries_swap20_vs_real.jsonl --out heldout/eval/swap20_vs_real --align auto --device cuda
python tools/make_review_page.py --entries heldout/eval/review_entries.jsonl --out heldout/eval/review/index.html   --aggregate heldout/eval/matched/aggregate_summary.json --aggregate-plot heldout/eval/matched/aggregate_drift.png
```

The four published `tools/` scripts (`select_episodes.py`, `convert_agibot_episode.py`, `eval_fidelity.py`, `make_review_page.py`)
are not part of the cookbook; they ship with this recipe in `assets/sim-bimanual/tools/`, with a README of inputs,
outputs and dependencies. The fifth, `make_manifests.py`, is a reconstruction (its header says so): the records state the
pairing rule, the frozen-frame definition and the manifest formats, not the commands that produced them.
One trap: the forward-dynamics path applies no normaliser to the action file, so the file must already hold
checkpoint-normalised values; the conversion script does that and reproduces the bundled example sidecars bit-exactly.
Held-out episodes: 20 from tasks 600 and 596, chosen because the bundled examples come from other tasks (first-frame
PSNR against every bundled clip at most 13.2 dB; the same episode would score above 30 dB). The 10 per task were taken by
even spread over the sorted eligible episode ids, not by the selector's seeded shuffle (its `--seed` default is 0); ids,
tars and the rule are in `assets/sim-bimanual/manifests/heldout_20.json`, the swap pairing in
`assets/sim-bimanual/manifests/heldout_swap20_pairs.json`. Dataset revision `2252b3fdc88dc67fa5d5e4b771e1b86f5e71c278`.

## Settings

| Setting                       | Value                                                                                                                                                                    |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Denoising steps               | 4                                                                                                                                                                        |
| Guidance                      | 1.0                                                                                                                                                                      |
| Decoding                      | autoregressive, causal                                                                                                                                                   |
| Resolution                    | model config "480"; 640×360 input, 640×352 output                                                                                                                        |
| Frame rate                    | model base 24 fps with fps modulation; rollouts resolved at 30 fps                                                                                                       |
| Action chunk                  | the model's internal latent chunk is 2 frames per autoregressive step; each request carries the whole action file (900 actions for 901-frame runs, 60 for 61-frame runs) |
| Seed                          | 42 on every rollout in the tables (same seed, same GPU: byte-identical repeat)                                                                                           |
| Checkpoint revision           | `1d95a0b5b19d49a24aceebf578cf5e85db310ac0` (checkpoint id `demo_stage3_sfdmd_chunk2_mecka_agibot_480p_step42@iter_000002000`)                                            |
| Framework / cookbook revision | `573d46507dcd91155d7f371b366da760a7a5f718` / `190b6565726fea4695ad168ee6bf197d30ca7c88` (early access)                                                                   |
| Dataset revision              | `agibot-world/AgiBotWorld-Beta` `2252b3fdc88dc67fa5d5e4b771e1b86f5e71c278`                                                                                               |

## Results

Measured on 1× GB300, 4 denoising steps, guidance 1.0, autoregressive rollouts from the real first frame with the real
29-D actions. Fidelity against the real video on 20 held-out AgiBotWorld-Beta episodes (10 each from two tasks), top
352 rows, mean ± std over episodes. Every number in the tables below is in
[`assets/sim-bimanual/results/metrics.json`](assets/sim-bimanual/results/metrics.json) with the run record it came from.

### What the numbers look like

| ![Task 600, episode 815362: real, matched rollout and frozen control at frames 30 and 900](assets/sim-bimanual/stills_task600_ep815362_f30_f900.png) | ![Task 596, episode 810449: real, matched rollout and frozen control at frames 30 and 900](assets/sim-bimanual/stills_task596_ep810449_f30_f900.png) |
| ---------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| task 600, episode 815362: real, matched rollout, frozen control at frame 30 and frame 900                                                            | task 596, episode 810449 (place the inner pot with rice into the rice cooker): the same three at frame 30 and frame 900                              |

The two tasks differ by 3.6 dB overall; episode ids in `assets/sim-bimanual/captions.md`. The "top 352 rows" crop keeps
rows 0 to 351 of the 360-row real frame, because the generator outputs 352 rows and keeps the top of the input: frame 0
of every generated video scores 33 to 36 dB against that crop (the autoencoder round-trip of the conditioning frame),
while a centre or bottom crop gives 17 to 24 dB. If your alignment check gives the lower numbers, the crop is wrong.

### Speed and memory

| Run                                                      | Frames | Rollouts | Wall incl. load | Seconds per rollout | Frames/s | Peak GPU |
| -------------------------------------------------------- | ------ | -------- | --------------- | ------------------- | -------- | -------- |
| bundled example, 901 frames                              | 901    | 4        | 9 min 04 s      | 96–156 (mean 125)   | 7.2      | 56.2 GB  |
| bundled example, 61 frames                               | 61     | 4        | 1 min 36 s      | 7–19 (mean 10.8)    | 5.7      | 36.1 GB  |
| held-out, task 600                                       | 901    | 10       | 19 min 36 s     | 98–130 (mean 111)   | 8.1      | 56.2 GB  |
| held-out, task 596                                       | 901    | 10       | 16 min 46 s     | 72–119 (mean 92)    | 9.8      | 56.2 GB  |
| cross-task action swap, early four-episode probe (run09) | 901    | 4        | 5 min 21 s      | 69–73 (mean 70.5)   | 12.8     | 56.2 GB  |
| cross-task action swap, all 20 episodes (control; run10) | 901    | 20       | 38 min 12 s     | 96–122 (mean 108)   | 8.3      | 56.2 GB  |
| all 901-frame rollouts before run10                      | 901    | 39       | —               | 69–156 (mean 103)   | 8.7      | 58.5 GB  |

At these settings the simulator produces 7 to 13 frames per second of wall-clock throughput on one GB300, not real
time. The 30 fps in Settings is the clock of the saved video, not the generation speed: a 901-frame rollout plays back
in 30 s and takes 70 to 156 s to generate.

### Fidelity over the rollout (matched real actions vs two controls)

| Comparison                                | Metric  | frame 30     | frame 60     | frame 120    | frame 240    | frame 480    | frame 900    |
| ----------------------------------------- | ------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| matched actions                           | PSNR dB | 31.08 ± 4.72 | 24.64 ± 5.93 | 19.89 ± 2.04 | 17.03 ± 2.13 | 15.63 ± 3.07 | 14.74 ± 2.18 |
|                                           | SSIM    | 0.927        | 0.853        | 0.801        | 0.743        | 0.667        | 0.621        |
|                                           | LPIPS   | 0.062        | 0.112        | 0.171        | 0.232        | 0.309        | 0.362        |
| frozen first frame (control)              | PSNR dB | 33.78 ± 7.14 | 23.32 ± 9.17 | 16.51 ± 2.37 | 15.20 ± 2.44 | 13.46 ± 2.00 | 12.59 ± 1.98 |
|                                           | SSIM    | 0.945        | 0.848        | 0.791        | 0.734        | 0.643        | 0.597        |
|                                           | LPIPS   | 0.045        | 0.143        | 0.210        | 0.264        | 0.375        | 0.438        |
| wrong actions, same 20 episodes (control) | PSNR dB | 27.61 ± 6.28 | 17.80 ± 4.40 | 14.72 ± 2.13 | 14.43 ± 2.07 | 12.49 ± 1.50 | 12.07 ± 1.41 |
|                                           | SSIM    | 0.902        | 0.769        | 0.716        | 0.674        | 0.570        | 0.520        |
|                                           | LPIPS   | 0.086        | 0.226        | 0.293        | 0.332        | 0.464        | 0.539        |

Window means (frames 0–299 | 300–599 | 600–900 | overall):

| Comparison                                                                                                               | PSNR dB                                 | SSIM                             | LPIPS                            |
| ------------------------------------------------------------------------------------------------------------------------ | --------------------------------------- | -------------------------------- | -------------------------------- |
| matched, 20 episodes                                                                                                     | 21.08 \| 15.90 \| 14.93 \| 17.30 ± 2.45 | 0.801 \| 0.677 \| 0.630 \| 0.703 | 0.173 \| 0.295 \| 0.348 \| 0.272 |
| matched, task 600 (10)                                                                                                   | 22.46 \| 17.96 \| 16.89 \| 19.10        | 0.832 \| 0.767 \| 0.728 \| 0.776 | 0.144 \| 0.231 \| 0.285 \| 0.220 |
| matched, task 596 (10)                                                                                                   | 19.69 \| 13.83 \| 12.96 \| 15.49        | 0.770 \| 0.587 \| 0.533 \| 0.630 | 0.203 \| 0.359 \| 0.410 \| 0.324 |
| frozen first frame (20)                                                                                                  | 19.63 \| 13.67 \| 12.60 \| 15.30 ± 2.23 | 0.795 \| 0.655 \| 0.602 \| 0.684 | 0.203 \| 0.361 \| 0.430 \| 0.331 |
| wrong actions, same 20 episodes (control: each task-600 episode driven by a task-596 episode's actions, and the reverse) | 17.18 \| 12.81 \| 11.90 \| 13.96 ± 1.32 | 0.734 \| 0.584 \| 0.516 \| 0.611 | 0.270 \| 0.446 \| 0.551 \| 0.422 |

Full tables with standard deviations on every cell and the per-task rows: [`results_tables_heldout.md`](assets/sim-bimanual/results_tables_heldout.md) (matched and frozen; also the early four-episode swap probe, superseded) and [`results_tables_swap20.md`](assets/sim-bimanual/results_tables_swap20.md) (the twenty-episode swap control, with its drift plot [`aggregate_drift_swap20.png`](assets/sim-bimanual/aggregate_drift_swap20.png)).

Reading. All three rows are the same 20 episodes. The generated video tracks the real episode better than a frozen
first frame from about frame 60 onward (+3.4 dB at frame 120, +2.2 dB at frame 900). Driven by the wrong episode's
actions it scores below both the matched rollout (14.0 against 17.3 dB overall) and the frozen frame (15.3 dB), so the
output depends on the actions it is given and not only on the first frame. Fidelity decays steadily over a 900-frame rollout,
from 31 dB to about 15 dB, and differs by task by 3.6 dB. These are fidelity numbers only; whether the simulator ranks
policies like a physics simulator has not been measured.

### How you know it worked

- The bundled example renders with two arms visible and the gripper state following the action stream.
- A held-out rollout beats the frozen-first-frame control from about frame 60 onward and beats the action-swap control overall.

## Prompt and agent guide

**Prompts for the simulator.** Each rollout takes the first frame, the 29-D action file and one text prompt. The
prompt used for every held-out rollout here was the task name followed by the first action description from the
dataset's own annotation ("Hang the key on the hook and place the bag in the storage box. Pick up the key on the
cabinet with left arm."). Keep it short, factual and in the dataset's words. The prompt sets the scene the model
imagines; the actions drive the motion. Whether a richer prompt changes fidelity was not measured; the prompt was held
fixed so that the action and frozen-frame controls isolate the actions.

**Running this recipe with a coding agent.** What to give an agent that runs this recipe from the document alone:

- Pin everything the agent cannot guess: framework commit, model and dataset revisions, the exact command block,
  the output directory layout. An agent that has to choose a version will choose a different one each run.
- Give it a stop rule and a budget in GPU-hours, and the three checks under "How you know it worked". Without a
  stop rule it keeps going past a finished result; without the checks it reports a broken run as done.
- Make it write a record per run (data, split, GPU type and count, peak memory, GPU-hours, result with seeds,
  control, leakage check) before it reports. A number without a record is not a result.
- Tell it what it may not do: delete checkpoints, change a pinned revision, write a secret to disk, enter
  generated data into a test set. Everything else it should fix itself and note in the record.
- Expect it to hit the deviations listed under Known limitations; they are the ones a careful reader hits too.
- For this recipe specifically: the agent must apply the checkpoint's own normaliser to the action file (the inference
  path applies none), check frame-0 alignment at 33 to 36 dB before scoring anything, and keep the bundled examples out
  of the held-out set. All three caught real mistakes during this work.

## Known limitations

- **Embodiment.** AgiBot 29-D actions only. No single-arm support, no YAM, no other bimanual schemas. Do
  not feed it LIBERO or SO-101 actions; there is no adapter.
- **Not a policy evaluator yet.** Whether this simulator ranks policy checkpoints the same way a physics
  simulator does has not been measured, because no public bimanual policy with a physics evaluator was
  available. Until that agreement is shown, use it for previewing and debugging a policy, not for
  deciding between checkpoints.
- **Not a data generator yet.** No policy or detector has been post-trained on this simulator's output
  in this program, so no claim is made about its value as training data.
- **(early access)** Checkpoint, cookbook and licence can change before release; replace every line marked (early access) at the GTC Berlin release.
- **Not real time at these settings.** 7 to 13 generated frames per second on one GB300 for 900-frame rollouts.
- **Dataset licence.** AgiBotWorld-Beta is CC BY-NC-SA 4.0 (see Dataset). This recipe is a guide: it points at the publisher's download and ships no data.

## Related measurement, different model (not part of this recipe's results)

A Cosmos3-Nano forward-dynamics model post-trained on AgiBot data (8× GB200, 1000 iterations, a different checkpoint
from `Cosmos3-Nano-Sim-Bimanual`, scored with an action-conditioned prediction error rather than PSNR) reached a held-out
error of 0.0012 against 0.0197 for a baseline that predicts the mean action chunk. It shows that the post-training path
works on this data; it says nothing about the released simulator checkpoint.

## Document Information

|                             |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Framework                   | `cosmos-framework` revision `573d46507dcd91155d7f371b366da760a7a5f718` and cookbook revision `190b6565726fea4695ad168ee6bf197d30ca7c88` (early access; `cookbooks/cosmos3/simulation/bimanual/bimanual.md`; public at GTC Berlin); the checkpoint does not load on public `c23e51f2f157ae3e51cfcd86ebfb5464850894f2`; the public bimanual documentation revision is not known yet (see Prerequisites)                                                                                                                                                                                                                                                  |
| Model                       | `nvidia/Cosmos3-Nano-Sim-Bimanual`, revision `1d95a0b5b19d49a24aceebf578cf5e85db310ac0` (early access)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Data                        | `agibot-world/AgiBotWorld-Beta`, revision `2252b3fdc88dc67fa5d5e4b771e1b86f5e71c278`, CC BY-NC-SA 4.0, user-downloaded, not bundled; episode manifest `assets/sim-bimanual/manifests/heldout_20.json`                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Tools (sha256, this folder) | `select_episodes.py` `b690b663d93282c4031efc0308d1fedc2395b314696a0a670a4c0dc78aef518d` (904 lines), `convert_agibot_episode.py` `9098f6256c7e5e0ea21d1ec31bc80622c71ac0b86ddd86b02ea9f3604b55e09d` (174 lines; the record's original `f6ea4df7…605105c`, 167 lines, hard-coded two local paths that this copy reads from environment variables), `eval_fidelity.py` `63f73ad3d47fd55b60fa76872937aa063a99f6b97588615674588c3b6f391503` (1113 lines), `make_review_page.py` `f4a6b917488e235625a3f738349e8c2d3e2c2aa3041ee1569799e98da65048c4` (793 lines); `make_manifests.py` is new (reconstruction; hash in `assets/sim-bimanual/tools/README.md`) |
| Records                     | one RECORD.md per run (run01 bundled 901, run02 bundled 61, run03 to run07 bundled controls and seeds, run08 held-out task 600 and task 596, run09 four-episode swap probe, run10 twenty-episode swap control) plus the held-out data record, in the internal run-records repository under `recipes/sim-bimanual/`; `assets/sim-bimanual/results/metrics.json` names the record behind every number                                                                                                                                                                                                                                                    |
| Measured on                 | 1× GB300, 2026-10-01 → 2026-10-07; 20 held-out AgiBotWorld-Beta episodes, two tasks; wrong-action control on the same 20 episodes, 2026-10-07                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
