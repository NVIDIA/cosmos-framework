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

## Brev Nano and fully local VANTAGE scoring

The [Brev launchable](launchable/README.md) uses this Framework-native kit, the
pinned Nano revision, and a caller-supplied, labeled `scoring-assets.json` bundle.
Prepare authorized sources offline with `python -m vlmeval.scoring_assets`; that
step writes canonical tables and a fingerprinted manifest, not original labels
or media. Keep the bundle and runtime state outside every source checkout.
During evaluation Compose mounts the completed bundle read-only; local scoring
does not use an API judge, dataset downloader, credential mount, or external
submission workflow.

For `MetropolisVQA`, `MetropolisTemporal`, `MetropolisEventVerification`, and
`MetropolisDVC`, provide `local_data_file` and `local_video_root` together to
bypass cache and remote-source discovery. The local table must include genuine
ground truth and bundle-relative media identities; missing labels or media fail.
`Astro2DBenchDataset` accepts a `data_roots` mapping covering all registered child
datasets. Existing explicit local grounding, pointing, and SOT constructor paths
are reused. Local DVC scoring requires a fingerprinted, revisioned local
roberta-large checkpoint and never consults the remote BERTScore endpoint.

Use `python -m vlmeval.local_scoring` for existing native predictions without a
model endpoint or GPU. Prediction identity columns must match trusted rows;
prediction-supplied answers and paths cannot replace ground truth. Subsets
require an explicit opt-in and are reported as subsets. There is no aggregate
cross-task score or claim of official parity. Dataset distribution approval and
a named reference result remain release gates. Original labels and native
filtering are preserved; annotation anomalies are recorded in private local
diagnostics rather than repaired or silently omitted.
