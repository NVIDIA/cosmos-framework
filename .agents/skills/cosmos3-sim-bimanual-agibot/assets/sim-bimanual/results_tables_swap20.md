# Sim-Bimanual action-swap control tables (released checkpoint)

Cross-task action swap on all 20 held-out episodes: each rollout keeps its frame episode's first frame and prompt and takes the 900 actions of the paired episode of the other task. Released checkpoint `nvidia/Cosmos3-Nano-Sim-Bimanual` revision `a9e3adf6235f56f0082be7a96e497348c286c472`, 4 steps, KV cache 64, attention sink 8, seed 42, 1x NVIDIA GB300. Mean ± std across episodes (ddof=1); real video cropped to its top 352 rows (`--align auto` chose `top` for every episode; see `manifests/RECORD.md`).

## Checkpoint frames

| comparison                                      | metric    | f30           | f60           | f120          | f240          | f480          | f900          |
| ----------------------------------------------- | --------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| cross-task action swap vs real, all 20 episodes | PSNR (dB) | 22.83 ± 3.57  | 15.91 ± 2.10  | 14.55 ± 1.67  | 14.88 ± 2.16  | 13.37 ± 1.47  | 12.73 ± 1.45  |
| cross-task action swap vs real, all 20 episodes | SSIM      | 0.868 ± 0.043 | 0.732 ± 0.063 | 0.708 ± 0.059 | 0.679 ± 0.097 | 0.602 ± 0.096 | 0.556 ± 0.086 |
| cross-task action swap vs real, all 20 episodes | LPIPS     | 0.102 ± 0.045 | 0.256 ± 0.068 | 0.295 ± 0.062 | 0.318 ± 0.084 | 0.413 ± 0.081 | 0.480 ± 0.073 |
| swap, frame episode task 600 (10)               | PSNR (dB) | 23.78 ± 4.27  | 16.14 ± 2.82  | 14.86 ± 2.20  | 16.05 ± 2.42  | 14.09 ± 1.45  | 13.35 ± 1.82  |
| swap, frame episode task 600 (10)               | SSIM      | 0.871 ± 0.046 | 0.731 ± 0.079 | 0.723 ± 0.077 | 0.749 ± 0.077 | 0.679 ± 0.059 | 0.622 ± 0.065 |
| swap, frame episode task 600 (10)               | LPIPS     | 0.107 ± 0.048 | 0.263 ± 0.071 | 0.293 ± 0.076 | 0.275 ± 0.080 | 0.400 ± 0.076 | 0.490 ± 0.090 |
| swap, frame episode task 596 (10)               | PSNR (dB) | 21.87 ± 2.59  | 15.68 ± 1.10  | 14.25 ± 0.92  | 13.71 ± 0.96  | 12.65 ± 1.16  | 12.12 ± 0.52  |
| swap, frame episode task 596 (10)               | SSIM      | 0.865 ± 0.042 | 0.734 ± 0.048 | 0.693 ± 0.029 | 0.610 ± 0.056 | 0.526 ± 0.054 | 0.490 ± 0.039 |
| swap, frame episode task 596 (10)               | LPIPS     | 0.096 ± 0.045 | 0.249 ± 0.067 | 0.296 ± 0.048 | 0.360 ± 0.067 | 0.427 ± 0.087 | 0.470 ± 0.055 |

### Window means

| comparison                                      | metric    | frames 0-299  | 300-599       | 600-900       | overall       |
| ----------------------------------------------- | --------- | ------------- | ------------- | ------------- | ------------- |
| cross-task action swap vs real, all 20 episodes | PSNR (dB) | 16.81 ± 1.73  | 13.67 ± 1.34  | 12.55 ± 1.17  | 14.34 ± 1.20  |
| cross-task action swap vs real, all 20 episodes | SSIM      | 0.728 ± 0.062 | 0.613 ± 0.088 | 0.548 ± 0.085 | 0.630 ± 0.072 |
| cross-task action swap vs real, all 20 episodes | LPIPS     | 0.267 ± 0.058 | 0.398 ± 0.066 | 0.494 ± 0.054 | 0.387 ± 0.048 |
| swap, frame episode task 600 (10)               | PSNR (dB) | 17.49 ± 2.17  | 14.45 ± 1.23  | 13.25 ± 1.22  | 15.06 ± 1.28  |
| swap, frame episode task 600 (10)               | SSIM      | 0.760 ± 0.067 | 0.686 ± 0.053 | 0.622 ± 0.052 | 0.689 ± 0.051 |
| swap, frame episode task 600 (10)               | LPIPS     | 0.251 ± 0.065 | 0.383 ± 0.058 | 0.503 ± 0.060 | 0.379 ± 0.050 |
| swap, frame episode task 596 (10)               | PSNR (dB) | 16.12 ± 0.76  | 12.89 ± 0.95  | 11.84 ± 0.56  | 13.61 ± 0.49  |
| swap, frame episode task 596 (10)               | SSIM      | 0.695 ± 0.035 | 0.541 ± 0.044 | 0.474 ± 0.022 | 0.570 ± 0.024 |
| swap, frame episode task 596 (10)               | LPIPS     | 0.283 ± 0.048 | 0.413 ± 0.073 | 0.486 ± 0.050 | 0.394 ± 0.048 |
