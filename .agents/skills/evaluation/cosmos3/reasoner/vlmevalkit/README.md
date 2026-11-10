<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cosmos3 evaluation: public benchmark launchable and native recipes

This directory is an **NVIDIA-customized [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)**
maintained in Cosmos Framework. It contains reference settings for NVIDIA
Metropolis "smart infra" evaluation; [`cosmos_eval/`](cosmos_eval/) composes the
native inference and scoring engine. Toolkit docs live in
[`README-vlmevalkit.md`](README-vlmevalkit.md). These settings do not make private
datasets publicly reproducible or establish official score parity.

For user-hosted endpoints and local dataset inputs, see
[Deployment and dataset configuration](EXTERNAL_CONFIGURATION.md).
For the Brev notebook workflow, use the
[public benchmark launchable](launchable/README.md): MMSI-Video-Bench is the
default full reference run, with Perception Test and VideoPhy2 also
available. Public inputs are downloaded by the customer; local scoring does
not use an external judge or private dataset credentials. Expected pinned
measurements and their comparison limits are included.

For your own labeled video multiple-choice data, follow the
[standalone customer-data recipe](launchable/CUSTOM_DATA_RECIPE.md), then use
[Customer_Data.ipynb](launchable/Customer_Data.ipynb). Videos alone do not
provide an accuracy score: actual questions, options and ground-truth labels
are required. Existing agents perform one-off schema conversion; no benchmark
builder, packaging workflow or automatic ingestion agent is included.
Data and state stay outside the checkout. The launchable does not claim
identical published-reference protocols. Existing native recipes below remain
independent and unchanged.

The kit ships, for each benchmark, reference `run.py --config` settings —
split into a shared dataset layer (`cosmos_eval/data/`) and a per-family model layer
(`cosmos_eval/models/`) — plus a parallel launcher (`cosmos_eval/run_all.py`) and a standalone score
reporter (`cosmos_eval/parse_score.py`).

You deploy the model behind an OpenAI-compatible endpoint, point the launcher at it, and it composes
each config, runs the stock vlmevalkit `run.py`, and reports the headline score per benchmark. No
extra evaluation logic lives in the kit — inference and scoring are vlmevalkit's; the kit only composes
the inputs and reads the outputs.

> **This rollout ships the NVIDIA Metropolis "smart infra" benchmarks — `AETCBench_all` (the official
> **TAR** / Traffic Anomaly Reasoning benchmark) + the 8 `VANTAGE_*` (9 total).**
>
> **⚠️ The datasets are not publicly available yet** — both the VANTAGE and TAR
> challenges are still ongoing. **We will update this code as soon as the data and evaluation are ready.**
> Until then these configs are published as a reference; `run_all` runs every benchmark in the manifest.

```
.                          # NVIDIA-customized VLMEvalKit (stock inference + scoring engine)
├── README.md              # this file
├── README-vlmevalkit.md   # upstream VLMEvalKit toolkit docs
├── run.py                 # stock vlmevalkit entrypoint (driven once per benchmark)
└── cosmos_eval/                       # the reproduction kit
    ├── data/<domain>/<bench>.json     # dataset section per benchmark (model-independent)
    ├── models/<family>.json           # per-family model layer (day 1: cosmos)
    ├── manifest.json                  # per-bench run.py CLI flags
    ├── run_all.py                     # parallel launcher (compose → run.py → parse_score)
    └── parse_score.py                 # standalone score reporter
```

---

## 1. Environment setup

Use a clean virtual environment with **Python 3.10+**, run from the **repo root** (the directory that
holds this README):

```bash
python -m venv .venv && source .venv/bin/activate

# vlmevalkit's own dependencies
pip install -r requirements.txt

# Extra packages required at runtime but not pinned by requirements.txt:
#   einops, accelerate   -> imported when vlmevalkit loads its dataset modules
#   setuptools<81        -> jieba imports pkg_resources, which setuptools>=81 drops
#                           and Python 3.12 venvs omit
#   evaluate             -> TAR (AETCBench_all) scoring metric
pip install einops accelerate "setuptools<81" evaluate

# install vlmevalkit itself so `import vlmeval` resolves
pip install -e .

# TAR (AETCBench_all) only: also install `nvdataset` — NVIDIA's INTERNAL DSS client. It fetches
# the TAR (and cosmos-DVC) data and requires the NVDATASET_TENANTID / NGC_API_KEY credentials in §3.
```

Sanity check:

```bash
python -c "import vlmeval; print('vlmeval OK')"
```

---

## 2. Deploy the model endpoint (vLLM)

Serve the model behind an OpenAI-compatible `/v1/chat/completions` endpoint (see also
**[Reasoner with vLLM](../../../../cosmos3-inference-reasoning/references/reasoner-readme.md#run-with-vllm)**):

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve nvidia/Cosmos3-Nano \
  --async-scheduling \
  --allowed-local-media-path / \
  --port 8080 \
  --max-model-len 128000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --enable-chunked-prefill \
  --mm-processor-cache-gb 0 \
  --mm-encoder-tp-mode data \
  --media-io-kwargs '{"video": {"num_frames": -1, "fps": -1}}'
```

The `--media-io-kwargs` / `--mm-*` flags let the server honor the per-benchmark frame sampling the
configs request, and `--allowed-local-media-path /` lets it read local media. (Raise
`--tensor-parallel-size` for larger models.) Any OpenAI-compatible server works (vLLM, SGLang, TGI, a
hosted NIM, …) as long as it answers `POST /v1/chat/completions` and `GET /v1/models`; the wrapper
health-checks the endpoint and auto-detects the served model id from `/v1/models`.

The launchable pins `vllm/vllm-openai:cosmos3@sha256:db0bb920b0b54e82ea96a98659bbd21921f87d0dcfc86feffdafa2db3f08be55`
and model revision `e59a53c25979a090fa8706c9acc0c254a6e89b92`. Its inspected
vLLM version is `0.23.1rc1.dev1306+gbfee6a802`. This build does **not** expose
`--trust-request-mm-kwargs` in `vllm serve --help=all`: the supported equivalent
is its unconditional request override path. `renderers/online_renderer.py`
passes request `mm_processor_kwargs` into `RenderParams`; `renderers/hf.py`
forwards those settings into the multimodal prompt/processor. Do not add the
unsupported flag or replace this digest merely to obtain it. Other runtime
versions must be checked for their own trust/override policy. The sampler
environment override above also applies to the launchable's serving container.

Public benchmark runs save redacted requested/observed execution evidence per
response and an `execution-summary.json` aggregate, referenced by their
summary/provenance. Native `CosmosReason2` collection is opt-in via
`execution_evidence_dir` and `execution_processor_profile`; disabled collection
does not change its payload or return API. Evidence requests raw token IDs but
retains only visual groups, actual response model, usage and finish reasons—not
prompts, text token IDs, labels, media, endpoint URLs or credentials.

Nano's explicit processor profile comes from the pinned public
[`config.json`](https://huggingface.co/nvidia/Cosmos3-Nano/resolve/e59a53c25979a090fa8706c9acc0c254a6e89b92/config.json)
and [`preprocessor_config.json`](https://huggingface.co/nvidia/Cosmos3-Nano/resolve/e59a53c25979a090fa8706c9acc0c254a6e89b92/preprocessor_config.json):
patch 16, spatial merge 2, temporal patch 2. Each bounded consecutive visual
token group yields spatial grid area `tokens × (16 × 2)²` **pixels**; video
groups yield **frame slots** `groups × 2`, not decoded or unique frame counts.
These derivations assume that processor profile and unpruned placeholder
expansion, not a server-attested revision. Per-group geometry is checked, not
just total tokens. Width/height, decoder FPS, decoded frame counts and effective
generation settings remain explicitly unknown when the response cannot expose
them. Non-32-aligned image inputs can be resized by the processor, so original
input dimensions are not relabeled as effective pixel geometry. Requested
generation/revision and historical scientific provenance remain requests and
provenance; they are not newly verified by a successful completion.


---

## 3. Environment variables

```bash
# Model under test
export COSMOS_API_BASE=http://<host>:8080/v1/chat/completions
export COSMOS_MODEL=<served-model-id>      # or any placeholder; auto-detected from /v1/models
export COSMOS_API_KEY=<key-or-EMPTY>

export NVDATASET_TENANTID=<dss-tenant>   # DSS tenant for the nvdataset client
export NGC_API_KEY=<ngc-key>             # NGC key for the nvdataset client
```

`run_all.py` substitutes `${COSMOS_MODEL}` / `${COSMOS_API_BASE}` into each composed config at launch.

---

## 4. Run

From the vlmevalkit repo root:

```bash
# all benchmarks in the manifest
python cosmos_eval/run_all.py --model cosmos --concurrency 2 --work-dir ./out

# a specific few
python cosmos_eval/run_all.py --benchmarks VANTAGE_VQA,AETCBench_all --work-dir ./out
```

Per benchmark, the launcher composes `model_conf = {class} ∪ defaults ∪ benchmarks[bench]` and
`dataset_conf = data[bench] ∪ {model_family}`, writes `<work-dir>/_configs/<bench>.json`, runs
`run.py --config … --verbose --save-eval-results` with the manifest's CLI flags (one `run.py`
subprocess per benchmark, `--concurrency` at a time), then reports the score.

**Useful flags**

| Flag | Effect |
|---|---|
| `--benchmarks A,B` | run only these benchmarks (default: all in the manifest) |
| `--concurrency N` | up to N benchmarks (each its own `run.py`) at once |
| `--dry-run` | print the `run.py` commands and exit |
| `--export-configs DIR` | compose + write each `--config` JSON to `DIR` (placeholders kept), then exit |
| `--import-configs DIR` | run from a dir of (possibly hand-edited) configs instead of composing |

---

## 5. Data & scoring

These benchmarks' datasets are being published to Hugging Face:
- **VANTAGE** (`nvidia/PhysicalAI-VANTAGE-Bench`) — test inputs are released; ground truth is withheld,
  so scores come from the **VANTAGE leaderboard**.
- **TAR — Traffic Anomaly Reasoning** (config key `AETCBench_all`; `nvidia/PhysicalAI-Traffic-Anomaly-Reasoning`) —
  test answers are redacted; submit predictions to the **AI City Challenge** evaluation server.

`run_all` produces per-benchmark predictions (and a local score where ground truth is available); follow
each benchmark's leaderboard instructions to obtain official scores.

---

## 6. Confirm the results

After a run, each benchmark has its own output dir and a parsed score in the summary table:

```
=== cosmos_eval summary ===
  AETCBench_all       ok         <score>
  VANTAGE_2DGrounding ok         <score>
  ...
  9/9 ok, 0 failed
```

`run.py` writes its evaluation output to `<work-dir>/<bench>/<model>/T*/<model>_<bench>.{dict,df}.eval.json`.
Report any single output directly with the standalone reporter:

```bash
python cosmos_eval/parse_score.py --work-dir ./out/VANTAGE_VQA --dataset VANTAGE_VQA
# VANTAGE_VQA  Overall: <score>
```

`parse_score.py` prints the headline **Overall** (0–100) plus every native scalar sub-score key the
benchmark's eval JSON contains. It reads both eval-output shapes vlmevalkit emits
(`*.dict.eval.json`, `*.df.eval.json`). The scores it reports match the internal evaluation pipeline,
benchmark for benchmark.

Each benchmark dir also keeps `run.log` (full `run.py` output) for debugging; the composed config
used is under `<work-dir>/_configs/<bench>.json`.

---

## 7. Benchmarks & headline metrics

`parse_score.py` reports each benchmark's headline as **Overall** (0–100) plus all native sub-scores.
The headline metric and the eval-JSON key it reads, per benchmark:

| Benchmark | Modality | Headline metric | eval-JSON key |
|---|---|---|---|
| `VANTAGE_2DPointing` | image | accuracy | (generic) |
| `VANTAGE_2DGrounding` | image | Mean_IoU | `Mean_IoU` |
| `VANTAGE_Astro2D` | image | F1 @ IoU=0.5 | `f1` |
| `VANTAGE_VQA` | video | accuracy | `accuracy` |
| `VANTAGE_EventVerification` | video | macro-avg F1 | `macro avg--f1-score` |
| `VANTAGE_Temporal` | video | overall IoU | `overall.iou` |
| `VANTAGE_DVC` | video | SODA_c (BERTScore-based) | `overall.SODA_c` |
| `VANTAGE_SOT` | video | mean success-AUC | `Overall` |
| `AETCBench_all` (TAR) | video | weighted_mean | `weighted_mean` |

---

## 8. Tips for reliable runs

- **Concurrency vs. replicas:** each benchmark already issues ~`api_nproc` parallel requests, so a high
  `--concurrency` (benchmarks in parallel) can overload a single GPU replica. Keep it modest on one
  replica and scale up by adding replicas.
- **Resume:** outputs are isolated under `<work-dir>/<bench>/`; if a run is interrupted, just re-run the
  affected `--benchmarks`.
- **Verify before trusting a score:** scan each benchmark's `run.log` for dropped or empty-input warnings.
