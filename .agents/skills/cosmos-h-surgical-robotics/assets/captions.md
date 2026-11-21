<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: OpenMDW-1.1 -->

# Media captions

All media are produced by `scripts/make_comparison.py` from `scripts/evaluate.sh` outputs: ground truth |
released checkpoint | fine-tuned checkpoint, 320×176, 25 frames at 10 Hz, same first frame, actions,
prompt, seed (20260908), sampler (UniPC, 35 steps, guidance 6.0) and encoding; only the checkpoint differs.

**Attribution.** The ground-truth panels are frames of SutureBot (J. Haworth, J.-T. Chen, N. Nelson,
J. W. Kim, M. Moghani, C. Finn, A. Krieger, 2025), <https://huggingface.co/datasets/jchen396/SutureBot>,
licensed CC-BY-4.0, downscaled to 320×176. The predicted panels are generated from SutureBot first frames and
actions. Reuse of these clips must keep this attribution.

The PSNR values below are those of the measured run; per-clip values differ between training runs
(in a second run of the recipe, the hero clip's fine-tuned PSNR was 27.9 dB).

| File                   | Split           | Sample (window) | Task          | Checkpoint  | Seed     | PSNR base / fine-tuned (dB) | What to notice                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ---------------------- | --------------- | --------------- | ------------- | ----------- | -------- | --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hero.gif`, `hero.mp4` | test (tissue 3) | 1530            | needle throw  | `iter_3000` | 20260908 | 23.4 / 26.4                 | The right needle driver drives the needle through the tissue. Both models follow the recorded motion of both drivers; the fine-tuned prediction is closer in fine detail (needle arc, thread, specular highlights). At this size the difference is subtle; the PSNR gain of 3.0 dB is the largest on the test split.                                                                                                                         |
| `success.gif`          | test (tissue 3) | 1224            | needle pickup | `iter_3000` | 20260908 | 27.4 / 29.7                 | Both drivers close in on the needle and thread. Same pattern as the hero: motion followed by both models, fine detail closer for the fine-tuned model (+2.3 dB).                                                                                                                                                                                                                                                                             |
| `failure.gif`          | test (tissue 3) | 3842            | knot tying    | `iter_3000` | 20260908 | 28.4 / 22.0                 | Thread wrap around the left driver's jaws during knot tying. The released checkpoint reproduces the recorded thread path; the fine-tuned model routes the thread differently around the jaws and toward the right driver (−6.5 dB, the largest loss on the test split). Thin thread in contact with the instruments is where predictions diverge most; knot tying has the lowest mean paired difference on the test split (−0.28 ± 0.23 dB). |
