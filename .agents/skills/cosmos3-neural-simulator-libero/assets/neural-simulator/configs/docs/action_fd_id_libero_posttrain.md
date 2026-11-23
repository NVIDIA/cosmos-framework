# Action FD+ID LIBERO Post-Training (cell 2 of the neural-simulator recipe)

`action_fd_id_libero_posttrain` is [`action_fd_libero_posttrain`](./action_fd_libero_posttrain.md) with three
dataset deltas: forward and inverse dynamics mixed 50/50 per visit (`mode="forward_dynamics+inverse_dynamics"`),
video frame rates mixed 5/10/20 fps (`video_fps_choices=[5, 10, 20]`: 17 frames at source stride 4/2/1 with
dense 20 Hz actions of length 64/32/16, the sequence plan's subsampled-video case; the sample emits
`conditioning_fps` for the video and `conditioning_fps_action=20` for the actions), and a per-visit RNG seed
(`mix_seed`; draws come from `random.Random(f"{mix_seed}:{epoch}:{idx}")`, epoch via `set_epoch`).

| Piece        | Value                                                                                               |
| ------------ | --------------------------------------------------------------------------------------------------- |
| TOML         | `examples/toml/sft_config/action_fd_id_libero_posttrain.toml` (same values as cell 1 except names)  |
| Launch shell | `examples/launch_sft_action_fd_id_libero_posttrain.sh`                                              |
| Module       | `cosmos_framework/configs/base/experiment/action/posttrain_config/action_fd_id_libero_posttrain.py` |

Notes: the shipped `mode="joint"` is a 3-way FD/ID/WAM draw, not 50/50. The console `Loss` includes the
10x-weighted action term and ID items contribute zeros to the vision mean, so compare the per-modality lines
printed by `ModalityLossConsole` (every 10 iterations), not the totals. Relaunching a paused run restarts the
data stream at epoch 0; pass a different `dataloader_train.dataloader.datasets.libero.dataset.mix_seed` per
segment so the per-visit draws are not replayed. Each fps in `video_fps_choices` must divide the native fps.
