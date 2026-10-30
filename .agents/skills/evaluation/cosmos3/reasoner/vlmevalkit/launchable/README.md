# Cosmos3-Nano MVBench evaluation on Brev

Authenticated Jupyter orchestration for **public download → preparation → native inference → local scoring → results**, entirely inside the notebook after host startup. Existing native predictions can also be scored on CPU without a model endpoint, GPU, private S3, API judge, or external scoring submission. This is a benchmark launchable, **not a customer-data evaluation-agent implementation**.

[MVBench](https://arxiv.org/abs/2311.17005) has **20 tasks and 4,000 annotated multiple-choice questions**. The paper's [Ask-Anything repository](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2) supplies its original evaluation recipe. This launchable pins public [OpenGVLab/MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench/tree/230a2d4fac8900333c61754641c7a13e069ac9c6) data to `230a2d4fac8900333c61754641c7a13e069ac9c6` and reuses native VLMEvalKit MVBench task definitions, temporal bounds, sampling and prompt construction, plus the native exact-match scorer. No labeled private annotation bundle is required.

**Licensing and full-data prerequisite:** read the pinned dataset card and individual source terms before downloading. Its video terms are academic-research-only; public access does not imply commercial-use or redistribution rights. The card mentions 320 NTU RGB+D videos, but this pinned revision's exact manual list contains **200 RGB AVI files**, matching its 200 Fine-grained Pose questions. Request the exact [pinned file list](https://huggingface.co/datasets/OpenGVLab/MVBench/blob/230a2d4fac8900333c61754641c7a13e069ac9c6/video/MVBench_videos_ntu.txt) through [NTU/ROSE Lab's account and release-agreement process](https://rose1.ntu.edu.sg/dataset/actionRecognition/); only RGB is needed. Supply authorized media under `video/nturgbd` in the external data root. The downloader never recovers removed media from old revisions or mirrors. Full preparation fails with an actionable coverage report while required media is missing; an incomplete download is not a full benchmark score.

## Start authenticated Jupyter

Use native Linux x86-64, Docker Engine/Compose v2, and NVIDIA Container Toolkit for local Nano serving. The maintained source is [NVIDIA/cosmos-framework](https://github.com/NVIDIA/cosmos-framework), not `NVIDIA/cosmos`. Select the approved branch/commit containing this launchable and retain its exact revision. Disable Brev's managed Jupyter because Compose provides its own authenticated service.

From the Framework checkout root:

```bash
export LAUNCHABLE_STATE_DIR=/absolute/path/outside/checkouts/private-mvbench-state
# Optional: defaults to "$LAUNCHABLE_STATE_DIR/mvbench".
export MVBENCH_DATA_DIR=/absolute/path/outside/checkouts/mvbench-data
bash .agents/skills/evaluation/cosmos3/reasoner/vlmevalkit/launchable/setup.sh --brev
```

Omit `--brev` on a non-Brev host. Add **`--score-only`** to start only the CPU evaluator/Jupyter service, never a GPU endpoint. Public annotations and videos can be downloaded by the notebook itself; setup does not require preexisting dataset files or start inference. Host path/configuration validation completes before starting a paid model service.

State and data directories must be absolute, writable, outside the checkout, and separate from each other’s cache/results mounts. Compose project `cosmos3-mvbench` mounts the external data directory **read-write at `/mvbench`**, cache at `/cache`, and private results at `/results`. Keep these path exports in the host shell for subsequent lifecycle commands. No weights or dataset contents are embedded in the image/repository. No Docker socket, AWS profile, home directory, or private key is mounted.

`--brev` discovers the private `wt0` mesh IPv4 address and fails without it. An explicit private `JUPYTER_BIND_ADDRESS` takes precedence; non-Brev defaults to loopback. There is no public/wildcard fallback. Setup persists only the notebook token and non-secret bind address in restricted `.env`. Use Brev's authenticated secure port-8888 link or an SSH tunnel. From the launchable directory on the **host**, retrieve the login token with:

```bash
docker compose exec mvbench-eval jupyter server list
```

Never share that token or include it in screenshots. Open `/opt/vlmevalkit/launchable/Evaluate_Cosmos3.ipynb`. Jupyter's authenticated file root is `/`, so authorized NTU files can be staged via its file browser/terminal and result links work. The local model API has **no host port**, only the private container network. Host Docker commands are for setup/stop/lifecycle, not benchmark execution.

### Pinned Nano and BYOM

The local service targets `nvidia/Cosmos3-Nano`, revision `e59a53c25979a090fa8706c9acc0c254a6e89b92`, with serving image `vllm/vllm-openai:cosmos3@sha256:db0bb920b0b54e82ea96a98659bbd21921f87d0dcfc86feffdafa2db3f08be55`. `VLLM_IMAGE` can override the image, with the override recorded. Allow storage for weights, images, public video archives, extracted media, frames, and results. The launchable does not change Ultra/LaunchPad workflows.

For BYOM, set `COSMOS_API_BASE` to the full reachable `/v1/chat/completions` URL and `COSMOS_MODEL` to the exact served ID **before setup**; optionally supply `COSMOS_API_KEY` in the environment. Setup skips local serving. Set `COSMOS_MODEL_REVISION` to your real deployment/checkpoint revision; unknown BYOM revisions remain explicitly `unknown` rather than being attributed to the pinned Nano checkpoint. Container `localhost` is not the host. The endpoint must accept sampled-image messages and expose `/v1/models`. Keys stay in the environment, never config/notebook source.

## Execute one chosen notebook scope

The notebook starts with `RUN_FULL = False`: exactly the first five Egocentric Navigation questions (stable native indices **3400–3404**), one task and five of 4,000 rows. It is prominently identified as a **smoke subset**, never a full result. Its public archive is approximately 40 MB (`video/vlnqa.zip`).

Set **`RUN_FULL = True`** to opt into full evaluation directly. The notebook downloads the chosen scope, prepares it, checks the endpoint only for inference, runs inference plus scoring, and displays coverage/metrics/provenance. It does **not** unnecessarily rerun smoke before full. Full preparation requires all 20 official JSON files, all 4,000 questions and required media, including manually acquired NTU. Downloading current public archives alone cannot meet that gate. Missing coverage is written to `/mvbench/.mvbench/coverage-full.json` and reported clearly; nothing is silently truncated or dropped.

For existing predictions, set `SCORE_ONLY = True`, choose explicit `SCORE_SCOPE = 'smoke'` or `'full'`, and provide exactly one of `PREDICTIONS_FILE` or `EXISTING_WORK`. Set `DOWNLOAD_PUBLIC = False` when using already staged inputs offline. Score-only does not check the endpoint or construct an inference model.

### CLI equivalents inside Jupyter's evaluator environment

These commands can also be invoked from Jupyter's terminal; no host Docker benchmark command is necessary:

```bash
python launchable.py download --scope smoke
python launchable.py prepare --smoke
python launchable.py run --smoke --mode all --work-dir /results/smoke-unique
python launchable.py results --work-dir /results/smoke-unique

# Metadata-only acquisition, without video downloads:
python launchable.py download --scope metadata

# Full scope: authorized NTU files must already be placed under /mvbench/video/nturgbd.
python launchable.py download --scope full
python launchable.py prepare
python launchable.py run --mode all --work-dir /results/full-unique

# Separate native inference from local scoring:
python launchable.py run --smoke --mode infer --work-dir /results/infer-unique
python launchable.py score --scope smoke --work-dir /results/infer-unique

# Native predictions from another run, retaining original identity columns:
python launchable.py score --scope smoke --predictions /results/native-predictions.tsv \
  --score-dir /results/score-unique
```

`--data-root` overrides the `MVBENCH_DATA_DIR` default (`/mvbench`). `run` defaults to **full scope**, `--mode all`, `--nframe 16`, and **`--api-nproc 1`**. Full is never silently limited. All run/score output directories must be fresh. `--dry-run` still requires valid selected annotations/media and emits configuration only; it does not prove model compatibility. Score scope is mandatory and must match all expected identities exactly; there is no arbitrary partial-full option.

## Protocol, metrics, and provenance

The original VideoChat2 MVBench notebook uses **16 frames and resolution 224**, and computes average accuracy as total correct / total questions. This launchable starts with **16 frames** (configurable via `--nframe`), applies reference bicubic GroupScale(224) and GroupCenterCrop(224) before native frame stacking/JPEG output, and uses the existing CosmosReason2 generation recipe: temperature 0, seed 1, thinking disabled, maximum 8,192 output tokens, presence penalty unset, ten API retries, and 600-second request timeout. It preserves native MVBench temporal bounds and sampled-image prompts, avoiding lossy video re-encoding and second-stage server video sampling. Cosmos image encoding/normalization and answer parsing still differ from VideoChat2; reference-model tensor normalization is not applied to API images. This is **not a paper-score reproduction or parity claim**. Revision/media-path/temporal-bound/frame-count/preprocessing-isolated caches prevent distinct clips or protocols from sharing stale frames.

Scoring uses native `model='exact_matching'`, never an LLM/API judge. The primary metric is **count-based question accuracy as a fraction**. Native counts, per-task metrics and task-macro accuracy are reported separately. A five-question subset cannot stand in for the full 20-task metric. Supplied predictions are checked for exact identities/row counts, duplicates, blanks, and failed transport outputs; trusted labels and identity columns come from pinned official annotations, never prediction-supplied answers/paths. Nonempty malformed model answers retain native wrong-answer semantics. Failed scoring is not a successful zero or fabricated result.

The local wrapper additionally prevents punctuation-only option tokens from receiving the upstream matcher's accidental substring credit and preserves literal model strings such as `nan` as wrong answers rather than missing values. These guards are recorded in scorer provenance; they are not a claim of byte-for-byte VideoChat2 evaluator parity.

`run.json` records scope, exact selected IDs, coverage, dataset revision/manifest hash, complete emitted native config, frame count, API concurrency, generation parameters, model/deployment revision, Framework revision, source hashes, prediction location, metric summary and state transitions. `config.json`, streamed `run.log`, native predictions, `scores/summary.json`, and `/opt/evaluation-packages.txt` provide private reproducibility artifacts. Summaries retain prediction/scorer provenance. API concurrency affects scheduling and host RAM usage, not the frame count or generation protocol. Increase it only after measuring host resources.

### Observed validation (2026-10-29)

Runtime candidate `b0861ce448a648b2eb59c41a9358980bbb5df31c` was built and exercised on native Linux x86-64:

- All **28 behavioral regressions passed inside the newly built evaluator image**.
- JupyterLab **Run All Cells** executed all six code cells against the pinned Nano endpoint on one RTX PRO 6000 Blackwell Server Edition. The fixed five-question subset produced five valid predictions with matching identities and native local scores; correctness results remain private and are not a full benchmark score.
- The saved real predictions were rescored in a separate CPU-only container with **`--network none`**, with the same correctness counts.
- SHA-verified current public archives were downloaded and safely extracted. Actual full preparation found media for **3,800 of 4,000 rows** and rejected the run before inference because the **200 Fine-grained Pose NTU clips** were missing. No rows were dropped and no partial full score was produced.

This reused an existing warm GPU endpoint with a manually staged evaluator; it was **not a cold one-click launch**. The complete 20-task run remains blocked on authorized NTU acquisition. No published-score reproduction, reference-result parity, commercial/distribution approval, or customer-data agent implementation is claimed. No benchmark labels/media, credentials, executed notebook outputs, or private result artifacts belong in the PR.

## Stop, restart, and maintain

From the launchable directory on the **host**:

```bash
docker compose --profile local ps
docker compose --profile local stop
# Restart only Jupyter to score retained predictions:
bash setup.sh --score-only
```

Stopping containers does not stop Brev VM billing. Stop the instance separately after preserving authorized results; do not delete state/volumes merely to stop a run. Clear notebook outputs before sharing.

Maintainers should run `python -m unittest discover -s launchable -p 'test_*.py'` after integration, then exercise notebook smoke, offline CPU rescore, missing-NTU full-coverage failure, and an authorized complete full run when its prerequisites are available. Behavioral fixtures use synthetic artifacts and do not establish model/hardware results. The committed notebook is unexecuted. Native Linux x86-64 is the supported runtime; local ARM64 Docker emulation is not notebook/hardware validation.
