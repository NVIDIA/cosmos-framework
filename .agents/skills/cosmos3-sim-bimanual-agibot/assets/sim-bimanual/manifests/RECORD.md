# heldout/eval — fidelity of 20 held-out AgiBotWorld-Beta rollouts (2026-10-01 UTC)

Framework 573d46507dcd91155d7f371b366da760a7a5f718; checkpoint 1d95a0b5b19d49a24aceebf578cf5e85db310ac0; data as in ../RECORD.md; rollouts run08_heldout_task600, run08_heldout_task596, run09_heldout_swap.
Tools: eval_fidelity.py sha256 63f73ad3d47fd55b60fa76872937aa063a99f6b97588615674588c3b6f391503, make_review_page.py f4a6b917488e235625a3f738349e8c2d3e2c2aa3041ee1569799e98da65048c4 (same files as eval/bundled).

## Comparisons and exact commands (all `--align auto`; real = converted 640x360 clip, top 352 rows compared; frame-0 alignment PSNR 32.9-36.1 dB on all 20 pairs)

| group               | entries | command                                                                                                                  | device | wall                      |
| ------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------ | ------ | ------------------------- |
| matched_task600     | 10      | eval_fidelity.py --batch entries_task600.jsonl --out matched_task600 --device cpu (ran during the task-596 rollouts)     | CPU    | 320 s (15:17:xx-15:22:56) |
| matched_task596     | 10      | --batch entries_task596.jsonl --out matched_task596 --device cpu                                                         | CPU    | 322 s (15:33:3x-15:39:01) |
| matched (pooled 20) | 20      | --batch entries_heldout.jsonl --out matched --device cuda                                                                | GPU    | 273 s                     |
| swap_vs_real        | 4       | --batch entries_swap_vs_real.jsonl --out swap_vs_real --device cuda                                                      | GPU    | 63 s                      |
| swap_vs_matched     | 4       | --batch entries_swap_vs_matched.jsonl --out swap_vs_matched --device cuda                                                | GPU    | 66 s                      |
| frozen_baseline     | 20      | --batch entries_frozen_baseline.jsonl --out frozen_baseline --device cuda (real vs a 901-frame video of its own frame 0) | GPU    | 284 s                     |

Review page: make_review_page.py --entries review_entries.jsonl --out review/index.html --aggregate matched/aggregate_summary.json --aggregate-plot matched/aggregate_drift.png (10 episodes, 5 per task; 23 videos, 138 thumbnails, 147 MB; built 15:48 UTC).

## Results

## Checkpoint frames (mean ± std across episodes, ddof=1)

| comparison                                       | metric    | f30           | f60           | f120          | f240          | f480          | f900          |
| ------------------------------------------------ | --------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| matched actions, all 20 episodes                 | PSNR (dB) | 31.08 ± 4.72  | 24.64 ± 5.93  | 19.89 ± 2.04  | 17.03 ± 2.13  | 15.63 ± 3.07  | 14.74 ± 2.18  |
| matched actions, all 20 episodes                 | SSIM      | 0.927 ± 0.041 | 0.853 ± 0.075 | 0.801 ± 0.057 | 0.743 ± 0.074 | 0.667 ± 0.116 | 0.621 ± 0.096 |
| matched actions, all 20 episodes                 | LPIPS     | 0.062 ± 0.032 | 0.112 ± 0.057 | 0.171 ± 0.049 | 0.232 ± 0.067 | 0.309 ± 0.098 | 0.362 ± 0.068 |
| matched, task 600 (10)                           | PSNR (dB) | 32.96 ± 2.04  | 26.90 ± 5.92  | 20.58 ± 2.44  | 18.24 ± 1.91  | 17.61 ± 3.03  | 16.18 ± 2.11  |
| matched, task 600 (10)                           | SSIM      | 0.946 ± 0.013 | 0.876 ± 0.081 | 0.814 ± 0.076 | 0.786 ± 0.071 | 0.759 ± 0.079 | 0.693 ± 0.078 |
| matched, task 600 (10)                           | LPIPS     | 0.054 ± 0.010 | 0.086 ± 0.038 | 0.153 ± 0.056 | 0.195 ± 0.059 | 0.244 ± 0.071 | 0.320 ± 0.062 |
| matched, task 596 (10)                           | PSNR (dB) | 29.20 ± 5.92  | 22.37 ± 5.28  | 19.21 ± 1.34  | 15.81 ± 1.63  | 13.65 ± 1.41  | 13.31 ± 1.02  |
| matched, task 596 (10)                           | SSIM      | 0.909 ± 0.051 | 0.830 ± 0.064 | 0.788 ± 0.029 | 0.700 ± 0.047 | 0.575 ± 0.056 | 0.548 ± 0.043 |
| matched, task 596 (10)                           | LPIPS     | 0.070 ± 0.044 | 0.137 ± 0.064 | 0.188 ± 0.036 | 0.269 ± 0.054 | 0.374 ± 0.074 | 0.403 ± 0.046 |
| frozen first frame vs real (20), static baseline | PSNR (dB) | 33.78 ± 7.14  | 23.32 ± 9.17  | 16.51 ± 2.37  | 15.20 ± 2.44  | 13.46 ± 2.00  | 12.59 ± 1.98  |
| frozen first frame vs real (20), static baseline | SSIM      | 0.945 ± 0.042 | 0.848 ± 0.097 | 0.791 ± 0.073 | 0.734 ± 0.084 | 0.643 ± 0.115 | 0.597 ± 0.094 |
| frozen first frame vs real (20), static baseline | LPIPS     | 0.045 ± 0.040 | 0.143 ± 0.098 | 0.210 ± 0.082 | 0.264 ± 0.090 | 0.375 ± 0.101 | 0.438 ± 0.090 |
| cross-task action swap vs real (4)               | PSNR (dB) | 33.87 ± 0.79  | 17.01 ± 2.08  | 14.92 ± 2.41  | 15.36 ± 1.92  | 13.47 ± 1.71  | 12.51 ± 1.67  |
| cross-task action swap vs real (4)               | SSIM      | 0.950 ± 0.006 | 0.772 ± 0.070 | 0.721 ± 0.102 | 0.690 ± 0.067 | 0.621 ± 0.107 | 0.573 ± 0.090 |
| cross-task action swap vs real (4)               | LPIPS     | 0.041 ± 0.012 | 0.224 ± 0.071 | 0.285 ± 0.112 | 0.318 ± 0.047 | 0.392 ± 0.039 | 0.519 ± 0.039 |
| cross-task action swap vs matched generation (4) | PSNR (dB) | 38.25 ± 0.55  | 16.90 ± 2.05  | 14.96 ± 2.49  | 17.38 ± 3.18  | 15.15 ± 0.31  | 13.19 ± 1.85  |
| cross-task action swap vs matched generation (4) | SSIM      | 0.970 ± 0.002 | 0.775 ± 0.076 | 0.732 ± 0.108 | 0.759 ± 0.066 | 0.712 ± 0.017 | 0.602 ± 0.072 |
| cross-task action swap vs matched generation (4) | LPIPS     | 0.022 ± 0.009 | 0.217 ± 0.070 | 0.268 ± 0.119 | 0.242 ± 0.094 | 0.306 ± 0.081 | 0.475 ± 0.058 |

## Window means

| comparison                                       | metric    | frames 0-299  | 300-599       | 600-900       | overall       |
| ------------------------------------------------ | --------- | ------------- | ------------- | ------------- | ------------- |
| matched actions, all 20 episodes                 | PSNR (dB) | 21.08 ± 2.53  | 15.90 ± 2.98  | 14.93 ± 2.57  | 17.30 ± 2.45  |
| matched actions, all 20 episodes                 | SSIM      | 0.801 ± 0.057 | 0.677 ± 0.110 | 0.630 ± 0.111 | 0.703 ± 0.088 |
| matched actions, all 20 episodes                 | LPIPS     | 0.173 ± 0.051 | 0.295 ± 0.092 | 0.348 ± 0.081 | 0.272 ± 0.071 |
| matched, task 600 (10)                           | PSNR (dB) | 22.46 ± 2.35  | 17.96 ± 2.84  | 16.89 ± 2.13  | 19.10 ± 2.13  |
| matched, task 600 (10)                           | SSIM      | 0.832 ± 0.061 | 0.767 ± 0.075 | 0.728 ± 0.058 | 0.776 ± 0.060 |
| matched, task 600 (10)                           | LPIPS     | 0.144 ± 0.043 | 0.231 ± 0.066 | 0.285 ± 0.053 | 0.220 ± 0.051 |
| matched, task 596 (10)                           | PSNR (dB) | 19.69 ± 1.94  | 13.83 ± 1.12  | 12.96 ± 0.90  | 15.49 ± 0.95  |
| matched, task 596 (10)                           | SSIM      | 0.770 ± 0.033 | 0.587 ± 0.045 | 0.533 ± 0.036 | 0.630 ± 0.031 |
| matched, task 596 (10)                           | LPIPS     | 0.203 ± 0.041 | 0.359 ± 0.065 | 0.410 ± 0.046 | 0.324 ± 0.046 |
| frozen first frame vs real (20), static baseline | PSNR (dB) | 19.63 ± 3.24  | 13.67 ± 1.94  | 12.60 ± 1.93  | 15.30 ± 2.23  |
| frozen first frame vs real (20), static baseline | SSIM      | 0.795 ± 0.067 | 0.655 ± 0.108 | 0.602 ± 0.109 | 0.684 ± 0.088 |
| frozen first frame vs real (20), static baseline | LPIPS     | 0.203 ± 0.070 | 0.361 ± 0.097 | 0.430 ± 0.086 | 0.331 ± 0.079 |
| cross-task action swap vs real (4)               | PSNR (dB) | 17.53 ± 1.41  | 13.68 ± 1.36  | 12.14 ± 1.58  | 14.45 ± 1.10  |
| cross-task action swap vs real (4)               | SSIM      | 0.730 ± 0.048 | 0.625 ± 0.095 | 0.529 ± 0.101 | 0.628 ± 0.071 |
| cross-task action swap vs real (4)               | LPIPS     | 0.273 ± 0.047 | 0.390 ± 0.035 | 0.554 ± 0.028 | 0.406 ± 0.026 |
| cross-task action swap vs matched generation (4) | PSNR (dB) | 18.72 ± 1.87  | 15.03 ± 0.31  | 12.63 ± 1.35  | 15.46 ± 0.79  |
| cross-task action swap vs matched generation (4) | SSIM      | 0.766 ± 0.056 | 0.703 ± 0.023 | 0.558 ± 0.071 | 0.676 ± 0.025 |
| cross-task action swap vs matched generation (4) | LPIPS     | 0.234 ± 0.069 | 0.316 ± 0.071 | 0.521 ± 0.042 | 0.357 ± 0.053 |

## Findings

1. Matched actions, 20 episodes: PSNR 31.1 dB at frame 30 (robots are nearly still in the first second), 19.9 dB at 120, 17.0 at 240, 15.6 at 480, 14.7 at 900; windows 21.1 / 15.9 / 14.9 dB; overall 17.30 ± 2.45 dB, SSIM 0.703 ± 0.088, LPIPS 0.272 ± 0.071. Per-episode overall PSNR 14.2-23.4 dB; task 600 19.10 dB, task 596 15.49 dB.
2. Static baseline (frozen first frame vs real): 15.30 dB / 0.684 / 0.331 overall; windows 19.6 / 13.7 / 12.6 dB. The rollouts beat it by +2.00 dB PSNR paired mean (min -0.28, max +4.75; 19/20 episodes positive), +1.45 / +2.23 / +2.33 dB in the three windows, i.e. the advantage over "nothing moves" persists past 300 frames.
3. Task 596 (rice cooker) is harder than task 600 (keys): 15.5 vs 19.1 dB. Frame strips show the real episodes opening the cooker lid around frame 300 while the rollout keeps it closed and still follows the arm trajectory (montage_heldout.png); the base and head are static in all 20 episodes (robot position 0.000 m, head range 0.0 deg), so the loss is object-state, not viewpoint.
4. Cross-task action swap (4 rollouts): fidelity to the real video 14.4 dB overall vs 17.3 dB matched; every swapped rollout scores below the matched rollout of the same first frame (by 1.3 to 5.5 dB; see README table); three of the four sit at or below the frozen baseline and the fourth 0.3 dB above it, while all four matched rollouts sit above it. At frame 30 the swapped rollouts still score 33.9 dB because the robot is nearly still in the first second; the divergence builds afterwards.
5. Visual inspection: the frame strips of the 10 review-page episodes (5 per task) and the 3 swap clips shown there, plus the montage_heldout.png episodes, show no black frames, frozen tails or chunk-boundary jumps; the other 10 held-out rollouts were not inspected frame by frame and no automated check was run.
