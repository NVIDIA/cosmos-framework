# Deployment and dataset configuration

This page covers the model endpoints and the VANTAGE / Metropolis benchmarks
that NVIDIA added to the toolkit; datasets inherited from upstream VLMEvalKit keep
their own download behavior. For these, model endpoints and remote dataset stages
are opt-in: they come from the environment variables below, and without them the
benchmarks read from the local `LMUData` tree only.
Configuration does not grant access to a dataset or service.

## Model endpoints and local installations

The following environment variables supply the endpoint for each model entry.
Set these variables before loading the model configuration (normally before
launching `run.py`). Models are still constructed lazily. An `api_base` passed
explicitly to the model constructor has priority over the environment variable.
The Qwen, Cosmos Reason1, and ARM Thinker values are complete chat-completions
URLs; `LVS_service` expects its service base URL.

| Model entry | Environment variable |
| --- | --- |
| `qwen3_235b_a22b` | `VLMEVAL_QWEN3_235B_API_BASE` |
| `qwen3_30b_a3b` | `VLMEVAL_QWEN3_30B_API_BASE` |
| `qwen3_8b` | `VLMEVAL_QWEN3_8B_API_BASE` |
| `cosmos_reason1_7b` | `VLMEVAL_COSMOS_REASON1_API_BASE` |
| `LVS_service` | `VLMEVAL_LVS_API_BASE` |
| `Qwen3-VL-8B--crop--arm_thinker_prompt--sglang` | `VLMEVAL_ARM_THINKER_SGLANG_API_BASE` |
| `Qwen3-VL-8B--crop--official_prompt--vllm` | `LMDEPLOY_API_BASE` (read by the ARM Thinker wrapper; required unless an `api_base` is passed) |

API keys are resolved separately from endpoints. The Qwen and Cosmos Reason1
entries read `COSMOS_API_KEY` (a `key` argument is ignored). Both ARM Thinker
entries use `LMDEPLOY_API_KEY` when it is set, otherwise their configured
placeholder key `EMPTY`. `LVS_service` uses its `key` argument.

`INTERN_S1_MINI_MODEL_PATH` optionally overrides the existing Intern-S1-mini
model directory. `MINIGPT4_ROOT` optionally overrides the existing MiniGPT-4
checkout directory. Explicit `model_path` and `root` arguments take priority.
Other model entries keep their existing settings.

## Local dataset inputs

The DSS route reuses an existing local cache without importing `nvdataset` or
requiring credentials.
For an uncached DSS dataset, set both `NVDATASET_TENANTID` and `NGC_API_KEY`;
there is no default tenant, and the download stops with a `ValueError` when
either is missing.

Metropolis2D adapters accept `data_root` (and, for grounding, `annotation_file`)
through their constructor or dataset configuration (`datasets.yaml` ships
placeholders only).
Without either, the VANTAGE rows resolve to the hydrated local cache under
`LMUData`. Missing or nonexistent grounding annotation files produce a clear
`ValueError`.

## Dataset storage

The VANTAGE and Metropolis loaders look for their data under `LMUData` first
(paths in the table below). A remote fetch is attempted only when the local copy
is absent **and** the matching bucket variable is set. Without the bucket, the
VQA, Temporal, DVC, event-verification, pointing and SOT loaders raise
`FileNotFoundError` naming the expected directory; the Metropolis2D adapters
(Astro2D and 2D grounding) raise `ValueError` instead. `MetropolisDVC` uses the
DSS route above rather than a bucket. Set the storage variables before starting
the run.

`VANTAGE_2DGrounding` does not read the bucket variables. It needs either the
hydrated local cache (see the table below) or both `data_root` and
`annotation_file`, passed to the constructor or set in `datasets.yaml`. Both may
be local paths or `s3://` URIs; `s3://` URIs are fetched through
`AWS_ENDPOINT_URL` with the `VLMEVAL_BENCHMARK_S3_PROFILE` profile when it is set.
With neither the cache nor both paths, it raises a
`ValueError`.

| Variable | Used for |
| --- | --- |
| `VLMEVAL_BENCHMARK_S3_BUCKET` | VANTAGE stages, `MetropolisEventVerification`, and the optional video frame cache (`VideoCache`; off when unset) |
| `VLMEVAL_METROPOLIS_S3_BUCKET` | `MetropolisVQA` and `MetropolisTemporal` |
| `AWS_ENDPOINT_URL` | S3-compatible endpoint for the buckets above (standard AWS variable; unset = AWS default) |
| `VLMEVAL_BENCHMARK_S3_PROFILE` | Optional AWS named profile for `VLMEVAL_BENCHMARK_S3_BUCKET` and for `s3://` paths passed to the Metropolis2D adapters |
| `VLMEVAL_METROPOLIS_S3_PROFILE` | Optional AWS named profile for `VLMEVAL_METROPOLIS_S3_BUCKET` |

Without a `*_S3_PROFILE` variable, AWS credentials come from the standard AWS
credential chain: `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` in the environment,
`AWS_PROFILE`, or the shared AWS config and credentials files. Set a
`*_S3_PROFILE` variable only when a bucket needs a different named profile.

### Where each VANTAGE benchmark expects its local data

`LMUData` (environment variable; default `~/LMUData`) is the dataset root. Place
the downloaded data at the paths below, or pass `data_root` / `annotation_file`
(2D adapters) or `prepared_data_dir` (SOT) through `datasets.yaml` or the
benchmark's `data` section in the composed config.

| Benchmark | Expected path under `LMUData` |
| --- | --- |
| `VANTAGE_2DGrounding` | `vantage_2d_grounding/annotations/refdrone_test_llava.json` plus the image tree it references |
| `VANTAGE_Astro2D` (3 sequences) | `vantage_astro2d/<sequence>/images/` and `.../labels/` for `C0065-QA.m4v`, `IVA-0009-KPI-05_190916_10ft-60-deg.mp4`, `IVA-0009-KPI-11_220320_NVR_ch2_main_20220201180404_20220201190000_4_cut_4.mp4` |
| `VANTAGE_2DPointing` | `datasets/VANTAGE_2DPointing/Metropolis2DPointing.tsv` plus `images_annotated/` |
| `VANTAGE_DVC` | `datasets/VANTAGE_DVC/events/` and `videos/` |
| `VANTAGE_EventVerification` | `videos/VANTAGE_EventVerification/` (per-corpus `test_annotation.json` trees) |
| `VANTAGE_Temporal` | `datasets/VANTAGE_Temporal/VANTAGE_Temporal.tsv` plus `videos/` |
| `VANTAGE_VQA` | `datasets/VANTAGE_VQA/VANTAGE_VQA.tsv` plus `videos/` |
| `VANTAGE_SOT` | `datasets/VANTAGE_SOT/<sequence>/gt.json` plus the prepared 8-frame images |

## Judge

`build_judge` uses plain OpenAI model ids. Set `VLMEVAL_JUDGE_API_STYLE=gateway`
when the judge endpoint expects provider-prefixed ids (`azure/openai/...`).

## Brev benchmark launchable

The [Brev launchable](launchable/README.md) runs public MMSI-Video-Bench (the
default full reference run), Perception Test and VideoPhy2 from Jupyter
notebooks against the pinned public `nvidia/Cosmos3-Nano` checkpoint, with
pinned public downloads and local scoring only. It needs no private dataset,
dataset credentials, bucket variables or judge endpoint. Another compatible
model is configured with `COSMOS_API_BASE`, `COSMOS_MODEL`,
`COSMOS_MODEL_REVISION` and, for its input evidence, `COSMOS_PROCESSOR_PROFILE`.

To evaluate your own labeled video multiple-choice data, follow the
[customer-data recipe](launchable/CUSTOM_DATA_RECIPE.md) and
[Customer_Data.ipynb](launchable/Customer_Data.ipynb). Questions, answer options
and the correct answers are required; footage alone cannot produce a score.
