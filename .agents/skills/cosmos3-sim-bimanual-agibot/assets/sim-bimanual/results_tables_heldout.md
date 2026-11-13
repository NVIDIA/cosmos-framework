# Sim-Bimanual held-out fidelity tables (released checkpoint)

Matched-action rollouts and the frozen-first-frame control on the 20 held-out AgiBotWorld-Beta episodes, all 20 and per task. Released checkpoint `nvidia/Cosmos3-Nano-Sim-Bimanual` revision `a9e3adf6235f56f0082be7a96e497348c286c472`, 4 steps, KV cache 64, attention sink 8, seed 42, 1x NVIDIA GB300. Mean ± std across episodes (ddof=1); real video cropped to its top 352 rows (`--align auto` chose `top` for every episode; see `manifests/RECORD.md`).

## Checkpoint frames

| comparison                                       | metric    | f30           | f60           | f120          | f240          | f480          | f900          |
| ------------------------------------------------ | --------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| matched actions, all 20 episodes                 | PSNR (dB) | 24.88 ± 2.17  | 18.65 ± 1.98  | 19.42 ± 1.89  | 20.16 ± 1.94  | 18.72 ± 2.08  | 17.57 ± 2.06  |
| matched actions, all 20 episodes                 | SSIM      | 0.893 ± 0.023 | 0.803 ± 0.046 | 0.791 ± 0.038 | 0.776 ± 0.058 | 0.723 ± 0.085 | 0.680 ± 0.082 |
| matched actions, all 20 episodes                 | LPIPS     | 0.076 ± 0.018 | 0.165 ± 0.040 | 0.166 ± 0.033 | 0.158 ± 0.038 | 0.202 ± 0.040 | 0.261 ± 0.053 |
| matched, task 600 (10)                           | PSNR (dB) | 25.88 ± 2.41  | 18.75 ± 2.34  | 19.40 ± 1.90  | 20.59 ± 2.49  | 19.71 ± 2.38  | 18.29 ± 2.64  |
| matched, task 600 (10)                           | SSIM      | 0.899 ± 0.025 | 0.808 ± 0.058 | 0.797 ± 0.040 | 0.807 ± 0.055 | 0.793 ± 0.055 | 0.739 ± 0.070 |
| matched, task 600 (10)                           | LPIPS     | 0.078 ± 0.018 | 0.172 ± 0.049 | 0.171 ± 0.034 | 0.158 ± 0.043 | 0.194 ± 0.037 | 0.257 ± 0.064 |
| matched, task 596 (10)                           | PSNR (dB) | 23.88 ± 1.38  | 18.55 ± 1.67  | 19.44 ± 1.99  | 19.72 ± 1.15  | 17.74 ± 1.12  | 16.85 ± 0.92  |
| matched, task 596 (10)                           | SSIM      | 0.887 ± 0.020 | 0.799 ± 0.031 | 0.784 ± 0.035 | 0.744 ± 0.045 | 0.653 ± 0.039 | 0.622 ± 0.040 |
| matched, task 596 (10)                           | LPIPS     | 0.074 ± 0.018 | 0.158 ± 0.031 | 0.161 ± 0.032 | 0.158 ± 0.034 | 0.211 ± 0.043 | 0.264 ± 0.042 |
| frozen first frame vs real (20), static baseline | PSNR (dB) | 34.60 ± 8.20  | 23.18 ± 9.25  | 16.75 ± 2.37  | 15.46 ± 2.27  | 13.73 ± 2.18  | 12.86 ± 2.00  |
| frozen first frame vs real (20), static baseline | SSIM      | 0.945 ± 0.046 | 0.845 ± 0.097 | 0.792 ± 0.070 | 0.741 ± 0.078 | 0.652 ± 0.108 | 0.604 ± 0.090 |
| frozen first frame vs real (20), static baseline | LPIPS     | 0.042 ± 0.041 | 0.136 ± 0.094 | 0.201 ± 0.081 | 0.251 ± 0.089 | 0.359 ± 0.096 | 0.426 ± 0.089 |

## Window means

| comparison                                       | metric    | frames 0-299  | 300-599       | 600-900       | overall       |
| ------------------------------------------------ | --------- | ------------- | ------------- | ------------- | ------------- |
| matched actions, all 20 episodes                 | PSNR (dB) | 21.15 ± 1.32  | 19.07 ± 2.06  | 17.67 ± 2.06  | 19.30 ± 1.58  |
| matched actions, all 20 episodes                 | SSIM      | 0.804 ± 0.039 | 0.730 ± 0.084 | 0.689 ± 0.092 | 0.741 ± 0.068 |
| matched actions, all 20 episodes                 | LPIPS     | 0.148 ± 0.027 | 0.196 ± 0.040 | 0.249 ± 0.049 | 0.198 ± 0.035 |
| matched, task 600 (10)                           | PSNR (dB) | 21.55 ± 1.56  | 20.27 ± 2.10  | 19.01 ± 1.99  | 20.28 ± 1.62  |
| matched, task 600 (10)                           | SSIM      | 0.824 ± 0.040 | 0.799 ± 0.054 | 0.767 ± 0.052 | 0.797 ± 0.046 |
| matched, task 600 (10)                           | LPIPS     | 0.149 ± 0.031 | 0.186 ± 0.039 | 0.229 ± 0.048 | 0.188 ± 0.036 |
| matched, task 596 (10)                           | PSNR (dB) | 20.76 ± 0.94  | 17.87 ± 1.17  | 16.32 ± 0.99  | 18.32 ± 0.74  |
| matched, task 596 (10)                           | SSIM      | 0.784 ± 0.026 | 0.661 ± 0.038 | 0.610 ± 0.037 | 0.685 ± 0.029 |
| matched, task 596 (10)                           | LPIPS     | 0.148 ± 0.025 | 0.206 ± 0.040 | 0.269 ± 0.043 | 0.208 ± 0.033 |
| frozen first frame vs real (20), static baseline | PSNR (dB) | 20.03 ± 3.20  | 13.92 ± 2.09  | 12.91 ± 2.00  | 15.62 ± 2.29  |
| frozen first frame vs real (20), static baseline | SSIM      | 0.799 ± 0.064 | 0.662 ± 0.103 | 0.610 ± 0.105 | 0.690 ± 0.084 |
| frozen first frame vs real (20), static baseline | LPIPS     | 0.194 ± 0.070 | 0.347 ± 0.094 | 0.417 ± 0.088 | 0.320 ± 0.079 |
