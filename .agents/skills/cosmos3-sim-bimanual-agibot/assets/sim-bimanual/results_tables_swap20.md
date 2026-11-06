# Sim-Bimanual action-swap control tables (run10)

## Cross-task action swap on all 20 held-out episodes (run10, 2026-10-07): checkpoint frames (mean ± std across episodes, ddof=1)

| comparison                                      | metric    | f30           | f60           | f120          | f240          | f480          | f900          |
| ----------------------------------------------- | --------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| cross-task action swap vs real, all 20 episodes | PSNR (dB) | 27.61 ± 6.28  | 17.80 ± 4.40  | 14.72 ± 2.13  | 14.43 ± 2.07  | 12.49 ± 1.50  | 12.07 ± 1.41  |
| cross-task action swap vs real, all 20 episodes | SSIM      | 0.902 ± 0.052 | 0.769 ± 0.092 | 0.716 ± 0.072 | 0.674 ± 0.090 | 0.570 ± 0.102 | 0.520 ± 0.088 |
| cross-task action swap vs real, all 20 episodes | LPIPS     | 0.086 ± 0.057 | 0.226 ± 0.099 | 0.293 ± 0.079 | 0.332 ± 0.082 | 0.464 ± 0.079 | 0.539 ± 0.056 |
| swap, frame episode task 600 (10)               | PSNR (dB) | 27.72 ± 6.72  | 17.60 ± 5.16  | 14.75 ± 2.64  | 15.28 ± 2.24  | 13.36 ± 1.33  | 13.07 ± 1.24  |
| swap, frame episode task 600 (10)               | SSIM      | 0.902 ± 0.062 | 0.760 ± 0.110 | 0.722 ± 0.090 | 0.737 ± 0.073 | 0.651 ± 0.054 | 0.594 ± 0.052 |
| swap, frame episode task 600 (10)               | LPIPS     | 0.095 ± 0.066 | 0.238 ± 0.109 | 0.300 ± 0.095 | 0.299 ± 0.076 | 0.440 ± 0.054 | 0.533 ± 0.053 |
| swap, frame episode task 596 (10)               | PSNR (dB) | 27.51 ± 6.17  | 17.99 ± 3.77  | 14.69 ± 1.63  | 13.59 ± 1.56  | 11.62 ± 1.13  | 11.07 ± 0.67  |
| swap, frame episode task 596 (10)               | SSIM      | 0.902 ± 0.044 | 0.778 ± 0.075 | 0.710 ± 0.053 | 0.612 ± 0.057 | 0.488 ± 0.064 | 0.446 ± 0.040 |
| swap, frame episode task 596 (10)               | LPIPS     | 0.078 ± 0.048 | 0.214 ± 0.091 | 0.286 ± 0.065 | 0.364 ± 0.078 | 0.488 ± 0.095 | 0.544 ± 0.060 |

### Window means

| comparison                                      | metric    | frames 0-299  | 300-599       | 600-900       | overall       |
| ----------------------------------------------- | --------- | ------------- | ------------- | ------------- | ------------- |
| cross-task action swap vs real, all 20 episodes | PSNR (dB) | 17.18 ± 2.23  | 12.81 ± 1.36  | 11.90 ± 1.29  | 13.96 ± 1.32  |
| cross-task action swap vs real, all 20 episodes | SSIM      | 0.734 ± 0.070 | 0.584 ± 0.093 | 0.516 ± 0.088 | 0.611 ± 0.075 |
| cross-task action swap vs real, all 20 episodes | LPIPS     | 0.270 ± 0.069 | 0.446 ± 0.070 | 0.551 ± 0.054 | 0.422 ± 0.055 |
| swap, frame episode task 600 (10)               | PSNR (dB) | 17.50 ± 2.65  | 13.69 ± 1.17  | 12.92 ± 0.95  | 14.70 ± 1.40  |
| swap, frame episode task 600 (10)               | SSIM      | 0.760 ± 0.078 | 0.661 ± 0.054 | 0.594 ± 0.047 | 0.671 ± 0.055 |
| swap, frame episode task 600 (10)               | LPIPS     | 0.261 ± 0.076 | 0.422 ± 0.056 | 0.544 ± 0.054 | 0.409 ± 0.053 |
| swap, frame episode task 596 (10)               | PSNR (dB) | 16.86 ± 1.80  | 11.94 ± 0.91  | 10.88 ± 0.52  | 13.22 ± 0.72  |
| swap, frame episode task 596 (10)               | SSIM      | 0.708 ± 0.052 | 0.508 ± 0.048 | 0.438 ± 0.029 | 0.551 ± 0.030 |
| swap, frame episode task 596 (10)               | LPIPS     | 0.279 ± 0.064 | 0.470 ± 0.077 | 0.557 ± 0.056 | 0.436 ± 0.056 |
