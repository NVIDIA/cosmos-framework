# Deployment and dataset configuration

The evaluation toolkit accepts caller-provided model endpoints and local dataset
inputs. Existing private model paths, dataset locations, and storage defaults
are retained, so established internal commands continue to work.
Configuration does not grant access to a dataset or service.

## Model endpoints and local installations

The following optional environment variables override existing private defaults.
Set these variables before loading the model configuration (normally before
launching `run.py`). Models are still constructed lazily. An explicit constructor
argument has priority over the resolved configuration.
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

`INTERN_S1_MINI_MODEL_PATH` optionally overrides the existing Intern-S1-mini
model directory. `MINIGPT4_ROOT` optionally overrides the existing MiniGPT-4
checkout directory. Explicit `model_path` and `root` arguments take priority.
Other model entries keep their existing settings.

## Local dataset inputs

The DSS route reuses an existing local cache without importing `nvdataset` or
requiring credentials.
For an uncached DSS dataset, `NVDATASET_TENANTID` overrides the existing tenant;
`NGC_API_KEY` is still required for the download.

Metropolis2D adapters accept `data_root` and `annotation_file` through their
constructor or dataset configuration. Original private dataset YAML defaults
remain available. Missing or nonexistent grounding annotation files produce a
clear `ValueError`. Omni3D retains its existing private configuration and can be
pointed at another authorized dataset through `raw_image_path` and `json_path`.

## Brev Nano and public MVBench evaluation

The [Brev launchable](launchable/README.md) uses the Framework-native kit,
the pinned Nano revision, and `OpenGVLab/MVBench` at
`230a2d4fac8900333c61754641c7a13e069ac9c6`. It provides public download,
coverage validation, native inference, and local exact-match scoring for 20
tasks and 4,000 annotated multiple-choice questions. No private annotation
bundle, private S3, GPU, or model endpoint is required for score-only execution.

Set `LAUNCHABLE_STATE_DIR` to an absolute writable host directory outside the
checkout. `MVBENCH_DATA_DIR` defaults to its `mvbench` child and is mounted
read-write at `/mvbench`; cache and results use separate mounts. The notebook
downloads the selected public scope itself. Approved media terms restrict use
to academic research; public access is not commercial-use or redistribution
permission. Full coverage requires manually acquired authorized NTU RGB+D
media under `video/nturgbd`; removed videos are never fetched from historical
revisions or mirrors. Missing media produces a coverage error, not an incomplete
full score. Default smoke is five fixed Egocentric Navigation rows.

Native `run.py` configuration uses the local dataset class directly:

```json
{
  "data": {
    "MVBench": {
      "class": "MVBenchLocal",
      "dataset": "MVBench",
      "data_root": "/mvbench",
      "nframe": 16,
      "smoke": false
    }
  }
}
```

The controller composes this with `CosmosReason2` and the existing generation
recipe, defaulting to one API worker. Temporal sampling and prompts are native
MVBench, with sampled-image input and isolated clip/frame caches. The starting
16-frame target and scale/center-crop-224 frame preprocessing follow the original
recipe; Cosmos image encoding/normalization and answer parsing still differ from VideoChat2.
Native count-based accuracy (fraction), task macro, coverage and provenance
remain explicit. No LLM judge or published-score claim is involved.

Use `launchable.py score --scope smoke|full` with either an existing controller
`--work-dir` or native `--predictions` plus a fresh `--score-dir`. Exact selected
identities and complete coverage are required. Trusted answers come from pinned
annotations, never prediction-supplied fields. Score-only skips endpoint checks
and model construction. BYOM uses `COSMOS_API_BASE`, `COSMOS_MODEL`, optional
environment-only `COSMOS_API_KEY`, and an explicit `COSMOS_MODEL_REVISION`;
setup then skips local serving. This launchable does not implement a
customer-data evaluation agent or alter other model deployment tracks.
