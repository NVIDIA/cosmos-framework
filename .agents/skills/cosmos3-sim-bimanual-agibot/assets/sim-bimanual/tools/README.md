# Tools of the Cosmos3-Nano-Sim-Bimanual recipe

The scripts the recipe runs (`../../../README.md`, "Commands"). Run them from the framework root with the venv of
`docs/inference_bimanual.md` active, set up with the `train` extra:
`uvx --from 'uv==0.11.29' uv sync --locked --group cu130 --extra guardrail --extra train`. The `train` extra brings
`mujoco` 3.10.0 (the AgiBot forward kinematics) and the data loader the conversion gate compares against.
`convert_agibot_episode.py` and `make_manifests.py frozen` also need `ffmpeg` and `ffprobe` on `PATH`.

Two environment variables are read, by `convert_agibot_episode.py` and `check_conversion_gate.py` only:
`COSMOS_FRAMEWORK_ROOT` (default `.`; put first on `sys.path`) and `SIM_BIMANUAL_CHECKPOINT` (default
`./Cosmos3-Nano-Sim-Bimanual`; the downloaded checkpoint, whose `action_schema.json` pins the pose convention and the
normalisation statistics). Every path argument resolves against the working directory.

| script                      | role                                                                                                                                                                                | lines | sha256                                                             |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- | ------------------------------------------------------------------ |
| `select_episodes.py`        | Select held-out episodes from `task_info/*.json` and plan which dataset tars to download                                                                                            | 904   | `b690b663d93282c4031efc0308d1fedc2395b314696a0a670a4c0dc78aef518d` |
| `convert_agibot_episode.py` | One AgiBotWorld-Beta episode (`proprio_stats.h5` + `head_color.mp4`) to a 900x59 `normalized_unified_v1` action file, the real 901-frame clip, its first frame and an inference row | 241   | `4d0aa03f76bc987267333508b239f4f898d87a5cbe8edc912ae4d393e5394752` |
| `check_conversion_gate.py`  | Acceptance gate for the converter: the framework's data loader on the checkpoint's training bundle as the oracle, plus structural checks of the published sidecars                  | 130   | `9cb361ab078d949b6c76aaffa5551c9fafb27dd135f83fbc461b9a5a3547876e` |
| `make_manifests.py`         | Fixed prompts into the rows, the 20-row action-swap manifest, the frozen-first-frame clips, the evaluation manifests                                                                | 180   | `3e01434d509e7027c79bbd7e738867093bd339eb8c137aec1d25c35c8b3cac78` |
| `eval_fidelity.py`          | Real mp4 + generated mp4 to per-frame PSNR/SSIM/LPIPS, checkpoint and window tables, drift plots                                                                                    | 1113  | `63f73ad3d47fd55b60fa76872937aa063a99f6b97588615674588c3b6f391503` |
| `make_review_page.py`       | Entries JSONL + summaries to one self-contained HTML review page                                                                                                                    | 793   | `f4a6b917488e235625a3f738349e8c2d3e2c2aa3041ee1569799e98da65048c4` |
| `make_results.py`           | The results tables and `results/metrics.json` from the `eval_fidelity.py` summaries                                                                                                 | 198   | `47b959bb07a0d4debb97ce4ee21dd33cc190ad7a7f48252f2cad41bba9f338a5` |
| `make_media.py`             | Hero figure, strip GIF and stills                                                                                                                                                   | 123   | `1ecfc486f062448ff3d0a452f0eb6cdc062bbcc45aee82801a25607bd67a3e68` |
| `run_with_peak_memory.py`   | Runs `cosmos_framework.scripts.inference` unchanged and prints the torch peak GPU memory                                                                                            | 24    | `67343235fd6787d9dba5cc22374b9dc0f3b651ec219209993c26ac3e106e8051` |

## convert_agibot_episode.py

Builds the checkpoint's schema-5 input from one episode with the framework's own code: it assembles the FK state from
`proprio_stats.h5` (effector, joint, head, waist, robot position and orientation), applies the dataset's audited gripper
calibration (`robotiq`: the observed close range 35..120 stretched to 0..120), computes the 29-D AgiBot FK actions in the
pose convention the checkpoint pins (`backward_chunk_anchored_16f`), scatters them into the 59-D unified layout (head and
right wrist 0–17, right gripper 33, left wrist 34–42, left gripper 58), normalises with the global-asinh statistics the
checkpoint pins (file checked against its sha256; gripper slots against the 0.12 m metric width) and sets every invalid
slot to exactly zero. The real clip is cut to frames `[start, start + N + 1)` and resized 640x480 to 640x360 with the
framework's `agibot_video.resize_to_640x360`, as training does.

| flag                              | required | default          | meaning                                                                       |
| --------------------------------- | -------- | ---------------- | ----------------------------------------------------------------------------- |
| `--h5 PATH`                       | yes      |                  | the episode's `proprio_stats.h5`                                              |
| `--video PATH`                    | yes      |                  | the episode's `head_color.mp4`                                                |
| `--name NAME`                     | yes      |                  | basename of every output and the row's `name`                                 |
| `--task-id N`                     | yes      |                  | AgiBot task id; selects the audited gripper hardware                          |
| `--gripper-hardware HW`           | no       | from the task id | `robotiq` or `omnipicker`                                                     |
| `--prompt TEXT`                   | no       | empty            | the row's prompt (the recipe fills it later with `make_manifests.py prompts`) |
| `--start N`                       | no       | 0                | first frame of the window                                                     |
| `--num-actions N`                 | no       | 900              | N action rows, N + 1 video frames                                             |
| `--out DIR`                       | yes      |                  | output directory                                                              |
| `--seed`, `--guidance`, `--shift` | no       | 42, 3.0, 5.0     | written into the row, as in the published example rows                        |
| `--skip-video`                    | no       | off              | do not cut the clip                                                           |

Outputs under `--out`: `actions/<name>.json` (the 900x59 sidecar the row's `action_path` names), `<name>_640x360.mp4`
(the real clip), `<name>_initial.png` (its first frame, lossless; the row's `vision_path`), `<name>.jsonl` (one
inference row in the shape of the published `sim_bimanual_unifiedv1_hf.jsonl` rows: `raw_action_dim` 59,
`action_chunk_size` 900, `num_frames` 901, `fps` 30, `domain_name` `agibotworld`) and `<name>_convert_info.json`
(provenance). About 10 s per episode on CPU.

## check_conversion_gate.py

`--bundle DIR` (the extracted `examples/training/sim_bimanual_robot_example_v1.tar.zst` of the checkpoint, the directory
holding `samples.json`), `--out FILE` (default `conversion_gate.json`). Check A runs the converter on every raw record
of the bundle and compares it with the normalised action the framework's `LocalSimulationDataset` produces for the same
record: max |difference| at most 1e-5 on the valid slots of the `agibotworld` rows and exact zeros elsewhere. Check B
reads the published `agibot_{001,006,034}` sidecars: shape, exact zeros outside each row's domain mask, de-normalised
rot6d unit norm, gripper openness in [0, 1]. It also prints the first-frame RMSE of the bundle clips against the
published initial images, the evidence that the bundle holds different episodes from those sidecars, so the sidecars
cannot be reproduced from it. Exit status 0 only when every check passes. Measured: max difference 0.0 on all three
records (`results/metrics.json`, key `conversion_gate`).

## make_manifests.py

Four sub-commands, defaults in the recipe layout (`heldout/inputs/<task>/`, `outputs/`, `heldout/frozen`, `heldout/eval`):

- `prompts --inputs DIR --prompts FILE`: write the fixed prompt of every episode into its row and concatenate one
  `task<task>.jsonl` per task. Run it before any rollout.
- `swap --inputs DIR --out FILE`: the 20-row action-swap manifest. The i-th task-600 episode (ids ascending) is paired
  with the i-th task-596 episode, both directions; each row keeps the frame episode's first frame and prompt and takes
  the partner's action file. Row name `swap_<frame episode>_actions_<action episode>`.
- `frozen --inputs DIR --out DIR`: one 901-frame clip per episode holding the real frame 0, cropped to the generator's
  640x352 top rows (libx264, crf 8, yuv420p, 30 fps).
- `eval --inputs DIR --outputs DIR --frozen DIR --out DIR`: `entries_heldout.jsonl`, `entries_frozen.jsonl`,
  `entries_swap20_vs_real.jsonl` (the `--batch` format of `eval_fidelity.py`) and `review_entries.jsonl`.

The rollout output directories are fixed: `outputs/cosmos3_nano_simulation_humanoid_heldout` (matched) and
`outputs/cosmos3_nano_simulation_humanoid_heldout_swap20` (swap).

## eval_fidelity.py

Decodes both videos with PyAV, aligns the real frames to the generated size (`--align auto` picks the top, centre or
bottom crop, or a resize, by frame-0 PSNR; the top crop wins on every pair here), and computes per-frame PSNR and SSIM
on the CPU and LPIPS (AlexNet) on `--device`. It reports checkpoint frames (`--frames`, default 30 60 120 240 480 900)
and the windows [0, 300), [300, 600), [600, end) and overall, mean ± std with ddof 1. Single mode takes `--real`,
`--gen`, `--name`; batch mode takes `--batch FILE.jsonl` with `{"name", "real", "gen", "prompt"}` rows and adds
`aggregate_summary.json`, `aggregate_per_frame.csv` and `aggregate_drift.png` in `--out`. Exit 1 when an episode
failed. Deterministic. About 90 s per batch of 20 on one GPU.

## make_review_page.py

`--entries FILE.jsonl --out DIR/index.html [--aggregate FILE.json --aggregate-plot FILE.png]`: one HTML page with each
episode's real and generated videos side by side (plus the frozen and swap clips), thumbnails at fixed frames and the
metric tables. Videos are copied into `DIR/assets/`, all links are relative, so the directory can be moved. CPU only.

## select_episodes.py

Picks episodes from the dataset's `task_info/task_<id>.json` files by length (`--min-frames`, default 901) and an
exclusion list (`--exclude-text`, here `../manifests/exclusions.json`: the bundled examples' captions and task names),
and plans the fewest observation tars that hold them (`--fetch-index` lists the Hub, `--max-obs-tars`, `--min-tasks`).
Not needed to reproduce the measured set: its 20 episodes and three tars are in `../manifests/heldout_20.json`. The
measured set was taken by even spread over the sorted eligible ids of tasks 600 and 596, not by the selector's seeded
shuffle. `python select_episodes.py --help` lists every flag.

## make_results.py, make_media.py, run_with_peak_memory.py

- `make_results.py --eval heldout/eval --outputs outputs --out-dir <this folder> --meta meta.json` writes
  `results_tables_heldout.md`, `results_tables_swap20.md` and `results/metrics.json`; `meta.json` names the hardware,
  revisions, the inference console logs (per-rollout seconds) and the peak-memory lines.
- `make_media.py hero --real R --matched M --frozen F --title T --out-dir DIR --stem STEM` writes the hero PNG, a strip MP4 and the GIF (this folder keeps the PNG and the GIF);
  `make_media.py stills ... --out FILE.png [--frames 30 900]` writes a stills figure.
- `run_with_peak_memory.py` takes the arguments of `cosmos_framework.scripts.inference`, runs it unchanged and prints
  the torch peak allocated and reserved memory at the end.
