# Training on Windows: the overrides, the environment, the evaluator

`scripts/train_lora.ps1` applies everything below. This file says why each line exists, so that
a run that bypasses the launcher can be checked against it.

## The command shape

```
python -m cosmos_framework.scripts.train --sft-toml=<recipe>.toml -- <overrides>
```

Plain `python`, not `torchrun`: the shipped launchers assume eight GPUs. Overrides come after a
bare `--`. The recipe schema rejects `model.config.*` and `dataloader_*` keys ("Extra inputs are
not permitted") and TOML cannot express `null`, so nine of the eleven Windows overrides can only
live on the command line.

## The eleven overrides and what each prevents

| #       | Override                                                                             | Without it                                                                                                                                                  |
| ------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1       | `optimizer=adamw`                                                                    | the default `fusedadamw` imports Transformer Engine, which has no Windows build; fails after the model is loaded                                            |
| 2       | `trainer.distributed_parallelism=ddp`                                                | FSDP2 has no Windows path; DDP at world size 1 is the working configuration (a process group is still mandatory: the trainer runs an unconditional barrier) |
| 3       | `model.config.parallelism.data_parallel_shard_degree=1`                              | the Nano recipe assumes four or eight GPUs; sharding needs collectives gloo cannot run on CUDA tensors                                                      |
| 4       | `model.config.parallelism.data_parallel_replicate_degree=1`                          | same, replicate axis                                                                                                                                        |
| 5       | `model.config.policy.lora_enabled=true`                                              | full-parameter SFT of 8.8 B parameters does not fit                                                                                                         |
| 6 to 8  | `dataloader_train.num_workers=0`, `persistent_workers=false`, `prefetch_factor=null` | Windows spawns workers; forked CUDA decoding can return zero frames; the other two are invalid at zero workers                                              |
| 9 to 11 | the same three for `dataloader_val`                                                  | as above                                                                                                                                                    |

Three more the launcher always adds:

- `policy.attn_implementation=sdpa` **and** `model.config.policy.attn_implementation=sdpa`:
  the resolved config has two policy objects and the model reads the nested one; with #303 the
  default `cosmos` resolves to sdpa on Windows anyway, but setting both keys makes the run
  self-describing (why sdpa: `validation.md` § W14).
- `checkpoint.dcp_async_mode_enabled=false`: #303 ignores async on Windows with a warning;
  the explicit flag keeps the config honest.
- `checkpoint.load_path=` and `job.name=<name>`: an empty load path means a fresh run (the
  recipe's `"???"` aborts resolution; any value is ignored when `latest_checkpoint.txt` exists).

## The environment block

| Variable                                                                                                             | Value                                                           | Why                                                              |
| -------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------- |
| `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`, `LOCAL_RANK`                                                     | `127.0.0.1`, `29500`, `0`, `1`, `0`                             | the gloo `env://` bootstrap needs all five at world size 1       |
| `USE_LIBUV`                                                                                                          | `0`                                                             | Windows torch ships without libuv                                |
| `PYTHONIOENCODING`                                                                                                   | `utf-8`                                                         | a cp1252 console kills the run on a logged box-drawing character |
| `IMAGINAIRE_OUTPUT_ROOT`                                                                                             | the output root                                                 | the default is drive-relative on Windows and degrades silently   |
| `VLM_SAFETENSORS_PATH`                                                                                               | the converted model                                             | read by the recipe's `[model.backbone]`                          |
| `COSMOS_VIDEO_TRAIN_ANNOTATION`, `COSMOS_VIDEO_TRAIN_MEDIA`, `COSMOS_VIDEO_VAL_ANNOTATION`, `COSMOS_VIDEO_VAL_MEDIA` | your files                                                      | the `cosmos_video_conversation` experiment's data                |
| `COSMOS_VIDEO_NUM_FRAMES`                                                                                            | 8                                                               | frames sampled per clip                                          |
| `COSMOS_VIDEO_MAX_PIXELS`                                                                                            | 1179648 for 8 frames at 480x288; the default 81920 gives 128x64 | a per-clip pixel budget, not per frame                           |
| `COSMOS_VIDEO_SYSTEM_PROMPT`                                                                                         | the evaluator's prompt                                          | train with what you evaluate with                                |
| `COSMOS_VIDEO_DECODER_DEVICE`, `COSMOS_DATALOADER_SEED`                                                              | `cuda`, `42`                                                    | GPU decode; the shuffle seed                                     |

Skill note: before August 2026 the same knobs were named `WTS_*` and the experiment `wts_vlm`;
scripts written for that era set variables main ignores, and the input budget silently falls back
to 128x64 without a system prompt. Check the first log lines for the frame count and pixel budget.

## Accepted and ignored

`trainer.max_iter` on the command line parses, is not logged, and does nothing: the recipe's
`scheduler.cycle_lengths`, `trainer.num_epochs` and `trainer.steps_per_epoch` decide the length.
To shorten or lengthen a run, edit those four together and count the logged iterations.

## Recipe lengths

The shipped recipe is a 20-step smoke (`steps_per_epoch = 10`, `num_epochs = 2`,
`max_iter = 20`, `cycle_lengths = [20]`). For a real run on N records: `steps_per_epoch = N`,
`num_epochs = 3`, `max_iter = 3N`, `cycle_lengths = [3N]`, `save_freq_in_epoch = 1`. Measured
on the 128 GB unit with N = 960 at 8 frames of 480x288, main with sdpa, 1400 MHz cap: 113
minutes with no pacer pause, last-epoch mean loss 0.079 (the earlier stack with `cosmos`
attention at 1600 MHz: 75 minutes including 9 minutes paused). On the 64 GB unit set
`max_sequence_length = 8192`.

## What the log must show

`inject_lora_pre_fsdp: rank=16 alpha=32.0 ... targets=['q_proj','k_proj','v_proj','o_proj']`
followed by `LoRA injection successful: 144 modules wrapped, 15,335,424 trainable LoRA params`.
Without those lines the run is a full-parameter update and will run out of memory.

## Evaluating

`cosmos-reasoner-evaluate --config <toml>` with `RANK=0`, `WORLD_SIZE=1`, `LOCAL_RANK=0` set.
Template: `assets/evaluate_binary.toml`. Keys that matter:

- `[task] type = "binary"` for yes/no questions (the scorer needs both classes in the ground
  truth); `"mcq"` with `answer_type = "letter"` for multiple choice.
- `[dataset] annotation_path`, `media_dir`, `system_prompt`: the same LLaVA records as training.
- `[model] model_name`: the exported directory, or the converted base model for a baseline.
- `[vision]`: `video_decoder = "torchcodec-cuda-on-demand"`, `decoder_device = "cuda"`,
  `num_frames`, `min_pixels`, `max_pixels` per frame (147456 with `min_pixels` 4096 is the
  training budget for 8 frames; it does not make the evaluator reproduce 480x288, because the
  evaluator rounds each frame to multiples of 64 and feeds 512x256, see SKILL.md Variant 5 and
  `validation-spark-a.md` § Why the evaluator disagreed), and the loader: `dataloader_num_workers = 1` with
  `dataloader_prefetch_factor = 2` and the spawn context, or zero workers with prefetch
  disabled; the mixed setting fails at start.
- `results_dir`: receives `<model directory name>\freeform\general\results.json` (one row per
  record: `video_id` holding the record id, `question`, `response`, `full_response`, `gt`,
  `task`) and `<model directory name>\freeform\metrics.json`. `WEIGHTED_SCORE 0` in the log is
  a reporting artifact; read `overall.accuracy`.
- `[model] attn_implementation = "sdpa"` is accepted and attested in the log
  (`COSMOS_ATTENTION_IMPLEMENTATION_ATTESTATION requested=sdpa actual=sdpa`); it did not change
  the result in the one comparison made (`validation-spark-a.md`).
