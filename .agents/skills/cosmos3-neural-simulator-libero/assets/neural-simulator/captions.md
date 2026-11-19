# Recipe 4 assets — review strips from the held-out forward-dynamics rollouts

All frames are the model canvas content (160x320: third-person view left, wrist view right) stored by `scripts/fd_rollout_eval.py` in
`eval/<run>/episodes/epNNN/frames.npz` on the training station; upscaled 2x (nearest) for display; PSNR values are per-chunk means over the chunk's 16
generated frames (uint8, whole frame), computed from the same arrays. Protocol: condition on the real frame 0 and the real 16-action chunks, re-condition
on the last generated frame, UniPC 30 steps, guidance 1.0, shift 3.0; sampling seed 0 (per-chunk seed = episode x 100 + chunk). "cell 1 FD" = DCP
checkpoint `runs/main/.../iter_000001000` (regular weights, see PROVENANCE.md), "cell 2 FD+ID" = `runs/cell2/.../iter_000001000`, "base control" =
`examples/checkpoints/Cosmos3-Nano` without post-training. Numbers in `strip_meta.json`.

## hero_ep026_chunk00.png

Held-out episode 26 ("put both the alphabet soup and the tomato sauce in the basket", 296 frames), sampling seed 0, chunk 0 = generated frames 1-16
(columns: frames 1, 6, 11, 16). Rows: real video; cell 1 FD at iteration 1000 (chunk PSNR 31.58 dB); base Cosmos3-Nano control (22.09 dB).
Episode means for seed 0: FD 21.01 dB, control 15.90 dB.

## hero_ep026_chunk14.png

Same episode and seed, chunk 14 = generated frames 225-240 (columns: frames 225, 230, 235, 240), i.e. 14 re-conditioning steps after the given frame.
Rows: real; cell 1 FD (chunk PSNR 20.35 dB); base control (13.47 dB). Drift from chunk 0 to chunk 14: -11.2 dB for FD, -8.6 dB for the control
(which is already far off at chunk 0).

## fd_vs_fdid_ep309_fd_wins_chunk14.png

Held-out episode 309 ("put the white mug on the left plate and put the yellow and white mug on the right plate", 264 frames), seed 0, chunk 14
(frames 225-240): the chunk where FD leads FD+ID most in this episode. Rows: real; cell 1 FD (21.28 dB); cell 2 FD+ID (18.84 dB). Episode means:
seed 0 FD 22.08 vs FD+ID 21.33 dB; three-seed means 22.43 vs 21.45 dB (the largest FD advantage of the 19 held-out episodes, -0.98 dB).

## fd_vs_fdid_ep062_fdid_wins_chunk12.png

Held-out episode 62 ("put the white mug on the plate and put the chocolate pudding to the right of the plate", 251 frames), seed 0, chunk 12
(frames 193-208): the chunk where FD+ID leads FD most in this episode. Rows: real; cell 1 FD (18.86 dB); cell 2 FD+ID (19.71 dB). Episode means:
seed 0 FD 21.33 vs FD+ID 21.33 dB (equal at 2 decimals); three-seed means 21.30 vs 21.57 dB (the largest FD+ID advantage, +0.27 dB). FD+ID is better in
4 of 19 episodes on PSNR (3-seed means); overall it trails FD by 0.38 dB (20.20 vs 20.58).
