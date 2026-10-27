# Cosmos3 VANTAGE evaluation on Brev

Jupyter and CLI orchestration for native VLMEvalKit inference **and fully local scoring**. Use a labeled, fingerprinted local asset bundle; score existing predictions without a model endpoint or GPU. There is no external scoring submission, upload, or separate public harness.

**Release gates:** Dataset access does not grant permission to redistribute labels or media. Public inference-only inputs are insufficient for local scoring. Do not advertise published-score reproducibility until the labeled bundle is approved for distribution and a named reference result and release protocol are agreed. No private data, credentials, or benchmark outputs belong in this repository or PR.

This work targets the [16B BF16 Cosmos3-Nano checkpoint](https://huggingface.co/nvidia/Cosmos3-Nano/blob/e59a53c25979a090fa8706c9acc0c254a6e89b92/README.md), `nvidia/Cosmos3-Nano` at `e59a53c25979a090fa8706c9acc0c254a6e89b92`, and evaluates its reasoner. Other models and deployment tracks are outside this launchable's scope.

## Supply the scoring bundle

Set absolute **host** paths outside every source checkout before setup:

```bash
export SCORING_ASSETS_DIR=/absolute/path/to/authorized-scoring-bundle
export LAUNCHABLE_STATE_DIR=/absolute/path/to/private-runtime-state
```

The bundle contains `scoring-assets.json`, original labels, canonical tables, and media. DVC additionally requires the pinned local roberta-large checkpoint. Compose mounts the bundle read-only at `/scoring-assets`; frame caches use `/cache/lmudata`, and results use `/results`. Do not mount AWS profiles, home directories, or credential files. No shared NVIDIA profile, NGC key, dataset-service credential, or API judge is required at evaluation time.

Preparation requires a writable authorized source directory; execution requires the completed bundle mounted **read-only**, never a source checkout, credential store, or writable cache/results directory. Setup requires both exported paths and rejects checkout-local paths. Keep these exports in the host shell for later Compose lifecycle commands.

Each requested task must be complete. Missing labels/media or modified fingerprints fail before inference. A VQA-only bundle is sufficient for the five-question smoke; full evaluation requires all eight tasks. Internal access and release approval are separate requirements.

### Maintainers: prepare authorized sources

Preparation is an offline conversion, not a dataset downloader. Keep the original filtered release layout, including `VQA`, `Temporal`, `Dense Video Caption`, `Spatial`, `2dbbox`, `datasets/Metropolis2DPointing`, `event_verification_subset`, and `tracking`. It does not copy media or rewrite original labels. Never put the source tree inside the code checkout.

From the maintained Framework kit root (`.agents/skills/evaluation/cosmos3/reasoner/vlmevalkit`), build the CPU evaluator:

```bash
docker build --platform linux/amd64 -t cosmos3-vantage-eval:local -f launchable/Dockerfile .
```

For DVC, stage the genuine public metric model first. This download needs network access but no model-serving endpoint. It does not download benchmark data:

```bash
docker run --rm --platform linux/amd64 \
  -v "$SCORING_ASSETS_DIR:/scoring-assets" --entrypoint python cosmos3-vantage-eval:local \
  -c 'from huggingface_hub import snapshot_download; snapshot_download("FacebookAI/roberta-large", revision="722cf37b1afa9454edce342e7895e588b6ff1d59", local_dir="/scoring-assets/metric-models/roberta-large", allow_patterns=["config.json","merges.txt","model.safetensors","tokenizer.json","tokenizer_config.json","vocab.json"])'

docker run --rm --platform linux/amd64 --network none \
  -v "$SCORING_ASSETS_DIR:/scoring-assets" --entrypoint python cosmos3-vantage-eval:local \
  -m vlmeval.scoring_assets --assets-dir /scoring-assets --asset-id YOUR_IMMUTABLE_DATASET_VERSION \
  --bert-score-model metric-models/roberta-large \
  --bert-score-revision 722cf37b1afa9454edce342e7895e588b6ff1d59
```

For VQA only, omit the metric model and pass `--tasks VQA`. Preparation validates original ground truth before native conversion, including complete tracking-frame labels. Explicit VQA labels are trimmed/uppercased, never defaulted to A; binary and four-choice source questions are supported. Invalid/missing labels fail rather than manufacturing answers. Scorer-source fingerprints record this normalization; compare results only under an agreed reference protocol.

Do not regenerate or modify the manifest between inference and scoring. The controller records its hash and rejects a different snapshot. DVC model files are fingerprinted separately and checked before use.

## Start the launchable

Use Linux x86-64, Docker Engine/Compose v2, and NVIDIA Container Toolkit for local serving. The source repository is [NVIDIA/cosmos-framework](https://github.com/NVIDIA/cosmos-framework), not `NVIDIA/cosmos`. Select the approved branch/commit containing this launchable and record its exact Framework commit; upstream `main` must not be assumed to include an unpublished branch. Brev's managed Jupyter must be disabled because Compose provides it.

From the Framework checkout root, with the authorized bundle prepared and both host paths exported:

```bash
bash .agents/skills/evaluation/cosmos3/reasoner/vlmevalkit/launchable/setup.sh --brev
```

On a non-Brev host, omit `--brev`. For **existing-prediction scoring only**, add `--score-only`; this starts Jupyter without starting the GPU endpoint. Normal setup builds the evaluator and validates VQA assets before starting serving.

Brev access must use its authenticated/secure port-8888 link. `--brev` discovers the private `wt0` mesh IPv4 address and fails without it; non-Brev defaults to loopback. An explicit private `JUPYTER_BIND_ADDRESS` takes precedence. There is no public/wildcard fallback. Setup persists only its notebook token and non-secret bind address in permission-restricted `.env`. Retrieve the login token on the host with `docker compose exec vantage-eval jupyter server list`; never share it or capture it in screenshots.

Open `/opt/vlmevalkit/launchable/Evaluate_Cosmos3.ipynb`. Jupyter's authenticated file root is `/`, allowing links to `/results`. The evaluator has no Docker socket. The model API has no host port. Container lifecycle commands belong in the host terminal.

Serving uses `vllm/vllm-openai:cosmos3@sha256:db0bb920b0b54e82ea96a98659bbd21921f87d0dcfc86feffdafa2db3f08be55` (override with `VLLM_IMAGE`). Allow space for model weights, images, the labeled bundle, frame caches, and results.

### Validation status and limits

- The runtime code passed 20 native-scoring/artifact regressions in the packaged evaluator on native Linux x86-64. The committed fixtures are synthetic and use temporary directories; no benchmark labels, media, or executed notebook outputs are included.
- A private five-question Brev notebook smoke exercised Nano inference and native local scoring. Saved predictions were also rescored in a CPU evaluator with networking disabled. This demonstrates the subset workflow, not a full benchmark or published-score reproduction.
- DVC's genuine pinned local metric model was exercised offline with controlled captions, not model-generated benchmark answers. No DVC model-quality claim is made.
- Preparation preserves original labels, native filtering, and ground-truth denominators. Annotation anomalies are recorded in private local diagnostics rather than repaired or silently omitted.
- Full eight-task model evaluation, immutable one-click cold launch, end-to-end BYOM, and named reference-score reproduction remain unverified. Scoring assets, detailed operational evidence, and benchmark results must be supplied and retained separately.

## Run and score

The notebook defaults to a deterministic five-question VQA inference-and-scoring smoke. `RUN_FULL = False` prevents Run All from launching full eight-task inference. Score-only mode requires an existing native prediction table and explicitly selected task; it skips endpoint checks and model construction.

CLI equivalents inside the evaluator:

```bash
python launchable.py prepare --benchmarks VANTAGE_VQA
python launchable.py run --benchmarks VANTAGE_VQA --smoke --mode all --work-dir /results/smoke-unique
python launchable.py results --work-dir /results/smoke-unique

# Full native inference and local scoring, only after all task assets validate.
python launchable.py prepare --benchmarks all
python launchable.py run --benchmarks all --mode all --work-dir /results/full-unique

# Separate inference from scoring; the latter never contacts a model endpoint.
python launchable.py run --benchmarks VANTAGE_VQA --mode infer --work-dir /results/infer-unique
python launchable.py score --work-dir /results/infer-unique

# Native prediction files from another run: retain original identities/questions.
python -m vlmeval.local_scoring --task VANTAGE_VQA --predictions /results/native-predictions.tsv \
  --assets-dir /scoring-assets --output-dir /results/score-unique
```

All output directories must be fresh. Direct scoring requires full task coverage unless `--allow-subset` is explicitly supplied. A smoke/subset is always identified in coverage and is not a reference benchmark result. Duplicate, unknown, mismatched, blank, or failed predictions are rejected. Only verified prediction strings are transferred into trusted dataset rows: prediction-supplied labels, image dimensions, or paths cannot override ground truth.

`--mode all` always performs native inference first, then the same local scorer used for existing outputs. It never substitutes inference-only behavior or runs a remote scoring service. `--dry-run` composes configuration but does not prove inference or media compatibility.

## Metric semantics and provenance

| Task | Native primary metric | Unit / aggregation |
| --- | --- | --- |
| VQA | accuracy | Fraction; question micro |
| EventVerification | macro F1 | Fraction; class macro |
| Temporal | temporal IoU | Fraction; video macro of question means |
| DVC | SODA-c | Fraction; video macro, genuine local BERTScore |
| 2DGrounding | Acc@0.5 | Percent; expression micro |
| 2DPointing | accuracy | Fraction; question micro |
| SOT | success AUC | Fraction; sequence macro; initialization excluded |
| Astro2D | F1 | Percent; unweighted sequence macro |

Each private `summary.json` includes native metrics, explicit units/aggregation, expected/scored row counts, asset ID and annotation hashes, prediction snapshot hash, scorer-source hashes, and DVC metric-model fingerprints. There is **no invented cross-task overall VANTAGE score** or unverified leaderboard-parity claim. DVC requires a real local model; remote endpoints and dummy semantic scores are not allowed in local scoring. Missing or silently skipped examples fail scoring. Native failures leave a private `scoring-error.log`, not a success summary.

`run.json` records controller state and the exact scoring-asset manifest hash. Generated native configuration and `/opt/evaluation-packages.txt` identify runtime inputs. A BYOM endpoint's revision cannot be inferred from its display name; record its deployment revision separately. Treat predictions, native diagnostics, and summaries as private until reviewed. Nothing is uploaded automatically.

## BYOM, stop, and restart

At initial deployment, set `COSMOS_API_BASE` to the full `/v1/chat/completions` URL, `COSMOS_MODEL` to its exact served ID, and `COSMOS_API_KEY` only if required. Setup skips local serving for BYOM. The endpoint must support the native image/video requests and be reachable from the evaluator; container `localhost` is not the host. Native recipe exports intentionally retain environment placeholders; the controller supplies the checked endpoint URL before inference. The API adapter reads its key from the environment, not the generated configuration. Keep credentials in the environment, never notebook source or URLs.

From the launchable directory on the host:

```bash
docker compose --profile local ps
docker compose --profile local stop
# Recreate with the same bundle and intended mode:
bash setup.sh --score-only
```

Stopping a container does not stop underlying Brev VM billing. Stop the instance separately after preserving authorized results. Do not delete state or volumes to stop a run. Clear notebook outputs before sharing.

## Local development checks

In the installed evaluator environment, run `python -m unittest discover -s launchable -p 'test_*.py'`. Behavioral fixtures exercise native scorers without private data or inference services; DVC additionally needs a smoke with the real metric checkpoint. The committed notebook contains no executed outputs.

Platform limit: local ARM64 Docker's `linux/amd64` Jupyter kernel previously stalled even in a clean upstream image. Native Linux x86-64 is the supported runtime. CLI/native metric checks under emulation are not proof of notebook execution or a fresh Brev GPU deployment.
