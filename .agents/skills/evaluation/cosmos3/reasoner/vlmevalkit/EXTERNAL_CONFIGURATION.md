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

## Brev Nano and public Perception Test validation

The [Brev launchable](launchable/README.md) uses the Framework-native kit,
the pinned public Nano checkpoint, and Google DeepMind's public Perception Test
validation MCQ data: 19,140 questions over 5,260 videos. Default smoke selects
`video_8241`, official question IDs 0–4 (stable indices 13404–13408).
Full validation is explicit opt-in, never the held-out test split.
Dataset source revision is `3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`;
official annotations, cut mapping, README and public archives are SHA-256 pinned
in `vlmeval/perception_test_local.py`. No private dataset or dataset credentials
are required. The Perception Test authors' non-code materials are
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); retain attribution to
the [authors/source](https://github.com/google-deepmind/perception_test).
Code is Apache 2.0. Public access is not an additional clearance determination.

Set `LAUNCHABLE_STATE_DIR` to an absolute writable host directory outside the
checkout. `PERCEPTION_TEST_DATA_DIR` defaults to its `perception-test` child,
mounted read-write at `/perception-test`; cache and results use separate mounts.
Compose project is `cosmos3-perception-test`, evaluator service `perception-eval`.
Authenticated Jupyter binds only to a private or loopback address. The local
model has no host port; no Docker socket or cloud credentials are mounted.

Native `run.py` configuration uses the local dataset class directly:

```json
{
  "data": {
    "PerceptionTest": {
      "class": "PerceptionTestLocal",
      "dataset": "PerceptionTest",
      "data_root": "/perception-test",
      "nframe": 16,
      "smoke": false
    }
  }
}
```

The controller composes this with `CosmosReason2` and the existing generation
recipe, defaulting to one API worker. The launchable prototype samples 16 frames
uniformly inside official source cutpoints: `-1` means all frames, positive cuts
are exclusive stops and invalid cuts fail. Aspect-preserving downscale only to
maximum side 768, without cropping, precedes direct sampled-image input. This
protocol and text-to-option parser are adaptations, not official parser parity
or a published-model/reference-score reproduction.

`launchable.py download --scope metadata|smoke|full` stages public inputs.
`prepare --smoke` and `prepare` require complete smoke/full media respectively.
`run --mode infer|all` generates predictions and optionally scores them; the
notebook defaults to `all`, not a manual Docker benchmark command.
`score --scope smoke|full` accepts either an existing successful inference
`--work-dir` or native `--predictions` plus a fresh `--score-dir`. Score-only
downloads metadata only and prepares with `require_media=False`: no video files,
endpoint, GPU, or model construction. Canonical annotation/cut/scope fingerprints
are independent of media presence; media hashes are recorded separately.

Exact selected identities are mandatory. Missing/blank/transport-failure rows
reject incomplete runs; nonempty malformed/ambiguous answers count wrong and
invalid. Trusted labels come only from pinned annotations. Primary metric is
local top1 (`correct / total selected questions`), with per-area, per-reasoning
and per-tag diagnostics, never an API judge. Fresh outputs, failed-run artifacts,
streamed progress, source/model revisions and UTC elapsed timings are retained.
The notebook shows concise metrics/coverage and artifact links.

BYOM uses complete `COSMOS_API_BASE`, exact `COSMOS_MODEL`, optional
environment-only `COSMOS_API_KEY`, and explicit `COSMOS_MODEL_REVISION`;
setup then skips local serving. Score-only setup uses `bash setup.sh --score-only`.
This launchable does not implement a customer-ingestion evaluation agent or alter
other model deployment tracks. Source changes are not hardware or accuracy proof.
