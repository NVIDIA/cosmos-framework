# Sim-Bimanual held-out fidelity tables (run08, run09)

Matched-action rollouts and frozen-first-frame control on the 20 held-out AgiBotWorld-Beta episodes (run08, scored 2026-10-01), all 20 and per task. The rows labelled "early four-episode probe (run09)" are the first action-swap control on only four rollouts (815362 <-> 810449 and 824031 <-> 813610, run09); they were superseded by the twenty-episode swap control of run10, which is the one in the README's headline tables and in `results_tables_swap20.md`. Mean ± std across episodes, ddof = 1. Real video cropped to its top 352 rows.

## Checkpoint frames (mean ± std across episodes, ddof=1)

| comparison                                                                         | metric    | f30           | f60           | f120          | f240          | f480          | f900          |
| ---------------------------------------------------------------------------------- | --------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| matched actions, all 20 episodes                                                   | PSNR (dB) | 31.08 ± 4.72  | 24.64 ± 5.93  | 19.89 ± 2.04  | 17.03 ± 2.13  | 15.63 ± 3.07  | 14.74 ± 2.18  |
| matched actions, all 20 episodes                                                   | SSIM      | 0.927 ± 0.041 | 0.853 ± 0.075 | 0.801 ± 0.057 | 0.743 ± 0.074 | 0.667 ± 0.116 | 0.621 ± 0.096 |
| matched actions, all 20 episodes                                                   | LPIPS     | 0.062 ± 0.032 | 0.112 ± 0.057 | 0.171 ± 0.049 | 0.232 ± 0.067 | 0.309 ± 0.098 | 0.362 ± 0.068 |
| matched, task 600 (10)                                                             | PSNR (dB) | 32.96 ± 2.04  | 26.90 ± 5.92  | 20.58 ± 2.44  | 18.24 ± 1.91  | 17.61 ± 3.03  | 16.18 ± 2.11  |
| matched, task 600 (10)                                                             | SSIM      | 0.946 ± 0.013 | 0.876 ± 0.081 | 0.814 ± 0.076 | 0.786 ± 0.071 | 0.759 ± 0.079 | 0.693 ± 0.078 |
| matched, task 600 (10)                                                             | LPIPS     | 0.054 ± 0.010 | 0.086 ± 0.038 | 0.153 ± 0.056 | 0.195 ± 0.059 | 0.244 ± 0.071 | 0.320 ± 0.062 |
| matched, task 596 (10)                                                             | PSNR (dB) | 29.20 ± 5.92  | 22.37 ± 5.28  | 19.21 ± 1.34  | 15.81 ± 1.63  | 13.65 ± 1.41  | 13.31 ± 1.02  |
| matched, task 596 (10)                                                             | SSIM      | 0.909 ± 0.051 | 0.830 ± 0.064 | 0.788 ± 0.029 | 0.700 ± 0.047 | 0.575 ± 0.056 | 0.548 ± 0.043 |
| matched, task 596 (10)                                                             | LPIPS     | 0.070 ± 0.044 | 0.137 ± 0.064 | 0.188 ± 0.036 | 0.269 ± 0.054 | 0.374 ± 0.074 | 0.403 ± 0.046 |
| frozen first frame vs real (20), static baseline                                   | PSNR (dB) | 33.78 ± 7.14  | 23.32 ± 9.17  | 16.51 ± 2.37  | 15.20 ± 2.44  | 13.46 ± 2.00  | 12.59 ± 1.98  |
| frozen first frame vs real (20), static baseline                                   | SSIM      | 0.945 ± 0.042 | 0.848 ± 0.097 | 0.791 ± 0.073 | 0.734 ± 0.084 | 0.643 ± 0.115 | 0.597 ± 0.094 |
| frozen first frame vs real (20), static baseline                                   | LPIPS     | 0.045 ± 0.040 | 0.143 ± 0.098 | 0.210 ± 0.082 | 0.264 ± 0.090 | 0.375 ± 0.101 | 0.438 ± 0.090 |
| early four-episode probe (run09): cross-task action swap vs real (4)               | PSNR (dB) | 33.87 ± 0.79  | 17.01 ± 2.08  | 14.92 ± 2.41  | 15.36 ± 1.92  | 13.47 ± 1.71  | 12.51 ± 1.67  |
| early four-episode probe (run09): cross-task action swap vs real (4)               | SSIM      | 0.950 ± 0.006 | 0.772 ± 0.070 | 0.721 ± 0.102 | 0.690 ± 0.067 | 0.621 ± 0.107 | 0.573 ± 0.090 |
| early four-episode probe (run09): cross-task action swap vs real (4)               | LPIPS     | 0.041 ± 0.012 | 0.224 ± 0.071 | 0.285 ± 0.112 | 0.318 ± 0.047 | 0.392 ± 0.039 | 0.519 ± 0.039 |
| early four-episode probe (run09): cross-task action swap vs matched generation (4) | PSNR (dB) | 38.25 ± 0.55  | 16.90 ± 2.05  | 14.96 ± 2.49  | 17.38 ± 3.18  | 15.15 ± 0.31  | 13.19 ± 1.85  |
| early four-episode probe (run09): cross-task action swap vs matched generation (4) | SSIM      | 0.970 ± 0.002 | 0.775 ± 0.076 | 0.732 ± 0.108 | 0.759 ± 0.066 | 0.712 ± 0.017 | 0.602 ± 0.072 |
| early four-episode probe (run09): cross-task action swap vs matched generation (4) | LPIPS     | 0.022 ± 0.009 | 0.217 ± 0.070 | 0.268 ± 0.119 | 0.242 ± 0.094 | 0.306 ± 0.081 | 0.475 ± 0.058 |

## Window means

| comparison                                                                         | metric    | frames 0-299  | 300-599       | 600-900       | overall       |
| ---------------------------------------------------------------------------------- | --------- | ------------- | ------------- | ------------- | ------------- |
| matched actions, all 20 episodes                                                   | PSNR (dB) | 21.08 ± 2.53  | 15.90 ± 2.98  | 14.93 ± 2.57  | 17.30 ± 2.45  |
| matched actions, all 20 episodes                                                   | SSIM      | 0.801 ± 0.057 | 0.677 ± 0.110 | 0.630 ± 0.111 | 0.703 ± 0.088 |
| matched actions, all 20 episodes                                                   | LPIPS     | 0.173 ± 0.051 | 0.295 ± 0.092 | 0.348 ± 0.081 | 0.272 ± 0.071 |
| matched, task 600 (10)                                                             | PSNR (dB) | 22.46 ± 2.35  | 17.96 ± 2.84  | 16.89 ± 2.13  | 19.10 ± 2.13  |
| matched, task 600 (10)                                                             | SSIM      | 0.832 ± 0.061 | 0.767 ± 0.075 | 0.728 ± 0.058 | 0.776 ± 0.060 |
| matched, task 600 (10)                                                             | LPIPS     | 0.144 ± 0.043 | 0.231 ± 0.066 | 0.285 ± 0.053 | 0.220 ± 0.051 |
| matched, task 596 (10)                                                             | PSNR (dB) | 19.69 ± 1.94  | 13.83 ± 1.12  | 12.96 ± 0.90  | 15.49 ± 0.95  |
| matched, task 596 (10)                                                             | SSIM      | 0.770 ± 0.033 | 0.587 ± 0.045 | 0.533 ± 0.036 | 0.630 ± 0.031 |
| matched, task 596 (10)                                                             | LPIPS     | 0.203 ± 0.041 | 0.359 ± 0.065 | 0.410 ± 0.046 | 0.324 ± 0.046 |
| frozen first frame vs real (20), static baseline                                   | PSNR (dB) | 19.63 ± 3.24  | 13.67 ± 1.94  | 12.60 ± 1.93  | 15.30 ± 2.23  |
| frozen first frame vs real (20), static baseline                                   | SSIM      | 0.795 ± 0.067 | 0.655 ± 0.108 | 0.602 ± 0.109 | 0.684 ± 0.088 |
| frozen first frame vs real (20), static baseline                                   | LPIPS     | 0.203 ± 0.070 | 0.361 ± 0.097 | 0.430 ± 0.086 | 0.331 ± 0.079 |
| early four-episode probe (run09): cross-task action swap vs real (4)               | PSNR (dB) | 17.53 ± 1.41  | 13.68 ± 1.36  | 12.14 ± 1.58  | 14.45 ± 1.10  |
| early four-episode probe (run09): cross-task action swap vs real (4)               | SSIM      | 0.730 ± 0.048 | 0.625 ± 0.095 | 0.529 ± 0.101 | 0.628 ± 0.071 |
| early four-episode probe (run09): cross-task action swap vs real (4)               | LPIPS     | 0.273 ± 0.047 | 0.390 ± 0.035 | 0.554 ± 0.028 | 0.406 ± 0.026 |
| early four-episode probe (run09): cross-task action swap vs matched generation (4) | PSNR (dB) | 18.72 ± 1.87  | 15.03 ± 0.31  | 12.63 ± 1.35  | 15.46 ± 0.79  |
| early four-episode probe (run09): cross-task action swap vs matched generation (4) | SSIM      | 0.766 ± 0.056 | 0.703 ± 0.023 | 0.558 ± 0.071 | 0.676 ± 0.025 |
| early four-episode probe (run09): cross-task action swap vs matched generation (4) | LPIPS     | 0.234 ± 0.069 | 0.316 ± 0.071 | 0.521 ± 0.042 | 0.357 ± 0.053 |
