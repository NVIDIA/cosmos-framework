# Action FD LIBERO Post-Training (neural simulator)

Trains Cosmos3-Nano as a forward-dynamics model of the LIBERO-10 Franka arm: first
frame + 16 actions in, the next 16 frames out. Adapted from
[`action_fd_droid_posttrain`](https://github.com/NVIDIA/cosmos-framework/blob/c23e51f2f157ae3e51cfcd86ebfb5464850894f2/docs/action_fd_droid_posttrain.md): same model, optimizer, scheduler
and checkpoint settings; the data source is LIBERO, plus single-GPU adaptations (token budget
split into 16 micro-steps, 24 workers with within-shard window shuffling, `compile_tokenizer` and
`straggler_detection` off, monitoring callbacks every 10 iterations).

| Piece         | Value                                                                                            |
| ------------- | ------------------------------------------------------------------------------------------------ |
| Experiment    | `action_fd_libero_posttrain`                                                                     |
| TOML          | `examples/toml/sft_config/action_fd_libero_posttrain.toml`                                       |
| Launch shell  | `examples/launch_sft_action_fd_libero_posttrain.sh`                                              |
| Config module | `cosmos_framework/configs/base/experiment/action/posttrain_config/action_fd_libero_posttrain.py` |
| Dataset       | `LIBEROLeRobotDataset` (`mode="forward_dynamics"`), 20 FPS `nvidia/LIBERO_LeRobot_v3` libero_10  |
| Action space  | `frame_wise_relative` rot6d, 10-D `[dpos(3), rot6d(6), gripper(1)]`, `quantile_rot` normalized   |
| Views         | `concat_view` third-person + wrist (256x256 each -> 256x512), snapped to the 320x192 canvas      |
| Chunk         | 16 actions, 17 frames (frame 0 is conditioning)                                                  |
| Holdout       | explicit episode-id list via `LIBERO_HOLDOUT_PATH` (`split="train"` excludes them)               |

## Run

```shell
export LD_LIBRARY_PATH=''
export LIBERO_ROOT=examples/data/LIBERO_LeRobot_v3/libero_10
export LIBERO_HOLDOUT_PATH=/path/to/holdout_19.txt
export BASE_CHECKPOINT_PATH=examples/checkpoints/Cosmos3-Nano
export WAN_VAE_PATH=examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
export IMAGINAIRE_OUTPUT_ROOT=/path/to/output_root   # the launcher also accepts OUTPUT_ROOT as an alias

PYTHONPATH=. python -m cosmos_framework.scripts.train \
  --sft-toml examples/toml/sft_config/action_fd_libero_posttrain.toml --dryrun   # validate
bash examples/launch_sft_action_fd_libero_posttrain.sh                          # 1 GPU
```

## Batch definition

The DROID FD recipe's global batch is a token budget: 74000 packed tokens per rank x 8
ranks x `grad_accum_iter` 1. The single-GPU TOML halves the micro-step to 37000 tokens and
sets `grad_accum_iter = 16` with 37000 tokens per micro-step (74000 x 8 peaked at 241 GB on the GB300), so each optimizer step still sees 592k tokens.

Relaunching with the same output root and `job.name` resumes from `checkpoints/latest_checkpoint.txt`
(use a different `job.name` for smoke runs). If `trainer.max_iter` is raised for a resumed run,
raise `scheduler.cycle_lengths` with it.

## Holdout file

One episode id per line (extra columns and `#` comments ignored). `split="val"` selects
exactly these ids, which is what the evaluation uses. An empty value raises; only `None`
disables the explicit holdout.

Note: with the JSON prompt format the `idle_frame` field is always "0 out of 16" for LIBERO (the
idle-frame thresholds are metric, LIBERO actions are unit-less OSC commands); it is identical at
training and evaluation time.
