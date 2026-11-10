# heldout/eval: fidelity of 20 held-out AgiBotWorld-Beta rollouts, released checkpoint (2026-11-09)

Measured on 1x NVIDIA GB300 (CUDA 13, torch 2.10.0+cu130). Checkpoint `nvidia/Cosmos3-Nano-Sim-Bimanual` revision `a9e3adf6235f56f0082be7a96e497348c286c472`
(schema 5, `normalized_unified_v1`, 59 values padded to 64). Framework: public `main` plus this skill's port commits (inference code unchanged).
Data: AgiBotWorld-Beta revision `2252b3fdc88dc67fa5d5e4b771e1b86f5e71c278`, the 20 episodes of `heldout_20.json`, frames 0-900 of each.
This record replaces the early-access run record that stood here; the early-access numbers are not comparable (different checkpoint, action schema and sampling).

## Conversion and acceptance gate

`tools/convert_agibot_episode.py` writes schema-5 sidecars (scalar domain `agibotworld`, valid slots 0-17, 33, 34-42, 58, exact zeros elsewhere), global-asinh
statistics pinned by the checkpoint (sha256 `ae695e2c...594b0`, hash checked), metric gripper reference 0.12 m, pose convention `backward_chunk_anchored_16f`,
audited `robotiq` gripper calibration (every labelled episode of tasks 600 and 596 is `robotiq`).

Gate (`tools/check_conversion_gate.py`), run before any rollout: the conversion matches the framework's own data loader on all three raw records of the
release training bundle with max |difference| 0.0 on the valid slots and exact zeros elsewhere (agibotworld-domain rows: 811, 677, 771 of 900; the other rows are the loader's camera-pose rows).
**Deviation from the brief:** the three published sidecars `agibot_{001,006,034}_normalized_unifiedv1.json` cannot be reproduced from that bundle. The bundle holds three
different episodes: first-frame RMSE against the published initial images is 73 to 96 (of 255), the scenes differ, and the camera-pose row counts differ
(bundle 89/223/129, published 62/29/43). The published sidecars were therefore checked structurally instead: shape 900x59, exact zeros outside every row's domain
mask, de-normalised rot6d unit norm within 1.8e-7, openness within [0, 1]. The gate passed on those terms; the literal reproduction was not possible.
The release's own input check (`tools/check_bimanual_mixed_inputs.py`) passed. Full output: `results/metrics.json`, key `conversion_gate`.

## Rollouts (public command, `--num-steps=4 --seed=42 --kv-cache-inference-size=64 --attention-sink-size=8`, one GPU, no compile, no guardrails)

| run | rollouts | wall incl. model load |
| --- | -------- | --------------------- |
| published AgiBot 001 example (seed 2026) | 1 (901 frames, 640x352, 30 fps) | 80 s |
| matched, task 600 | 10 | 509 s |
| matched, task 596 | 10 | 506 s |
| action swap (cross-task) | 20 | 1070 s |

All 40 receipts `status: success`, 901 frames each. Per rollout 48 to 62 s (mean 51.0 s, 17.6 frames per second).
Peak torch memory 39.95 GB allocated, 47.94 GB reserved; nvidia-smi whole-GPU peak 49973 MiB.
The frozen-first-frame control needs no rollout (`make_manifests.py frozen`).

## Scoring (`eval_fidelity.py --batch ... --align auto --device cuda`, about 90 s per batch of 20)

Frame-0 alignment of generated against real: `auto` chose `top` (rows 0-351, all 640 columns) for every episode; frame-0 PSNR 32.9 to 36.73 dB for the chosen crop,
16.2 to 23.35 dB for the other candidates (task600_ep815362: top 35.5466, center 21.2522, bottom 18.5669). The real clip is the 640x480 head camera resized to 640x360 as in training;
the generator keeps the top-left 22x40 latents of the reflection-padded 832x480 canvas, so the output is the top 352 rows. Tables: `results_tables_heldout.md`, `results_tables_swap20.md`, `results/metrics.json`.

## Working-run rule (reported as measured, not tuned)

- Matched beats the action swap overall: PSNR 19.30 vs 14.34 dB, paired difference 4.96 dB, 20 of 20 episodes better.
- Matched beats the frozen first frame overall: 19.30 vs 15.62 dB, paired 3.68 dB, 20 of 20 better.
- Per-frame mean PSNR: matched is above frozen from frame 88 onward, not from frame 60 as the recipe states (at frame 60: matched 18.65 dB, frozen 23.18 dB).
