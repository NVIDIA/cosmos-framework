# Cosmos3-Nano · Perception Test validation MCQ on Brev

Run the public **Perception Test validation split**, from download to local top1,
inside an authenticated notebook. The default smoke is five questions (IDs 0–4)
from `video_8241`, stable indices 13404–13408. Full validation is opt-in: **19,140
questions / 5,260 MCQ videos**. This is not the held-out test split. Source changes
and an unexecuted notebook are not evidence of hardware execution or accuracy.

This launchable is a standalone public-data validation workflow, not a
customer-ingestion evaluation agent. It needs no private annotation bundle,
internal dataset service, dataset credentials, or API judge.

## Start on Brev

Use a native Linux x86-64 GPU instance with Docker Compose, NVIDIA container
support, and enough GPU capacity for the public Nano checkpoint. Full video
archive is 75,375,143,328 bytes before extraction; provision room for the archive,
extracted videos, model cache, and outputs. Smoke downloads the approximately
225 MB official sample archive; only `video_8241` is selected for inference.

From this directory on the host:

```bash
export LAUNCHABLE_STATE_DIR=/mnt/cosmos3-perception-test-state
# Optional separate data disk; defaults to $LAUNCHABLE_STATE_DIR/perception-test.
export PERCEPTION_TEST_DATA_DIR=/mnt/perception-test-data
bash setup.sh --brev
```

Without Brev, `bash setup.sh` binds Jupyter to loopback by default; use an SSH
tunnel. `--brev` discovers the private `wt0` mesh address and refuses public or
wildcard binding. Jupyter always requires a generated token. Retrieve the login
URL in the host terminal with:

```bash
docker compose exec perception-eval jupyter server list
```

Open **Evaluate_Cosmos3.ipynb** and run all cells. The notebook downloads the
selected public data, validates coverage **before any model request**, streams
native inference progress, scores locally, and displays top1/counts, coverage,
and artifact links. Leave `RUN_FULL = False` for smoke; set it to `True` to run
full validation directly, without first rerunning smoke. There is no manual
Docker benchmark command required. Committed notebook outputs are empty.

Compose project is `cosmos3-perception-test`; evaluator service is
`perception-eval`. Data is mounted read-write at `/perception-test`, cache at
`/cache`, and fresh run directories at `/results`. Host paths must be absolute,
outside the checkout, with data separate from cache/results. No Docker socket,
cloud credential directory, model weights, videos, annotations, or answer-bearing
results are baked into the evaluator. The model service has **no host port**.
Only authenticated Jupyter is exposed on the selected private address.

### Public model and serving pins

- Model: [`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano), revision
  `e59a53c25979a090fa8706c9acc0c254a6e89b92`.
- Serving image: `vllm/vllm-openai:cosmos3@sha256:db0bb920b0b54e82ea96a98659bbd21921f87d0dcfc86feffdafa2db3f08be55`.
- Existing `CosmosReason2` generation recipe: temperature 0, seed 1, thinking
  disabled, maximum 8,192 output tokens; default one API worker.

Public downloads do not require a Hugging Face token. For an existing compatible
endpoint, export complete `COSMOS_API_BASE` (`.../v1/chat/completions`), exact
`COSMOS_MODEL`, and explicit `COSMOS_MODEL_REVISION` before setup. Optionally set
`COSMOS_SERVING_IMAGE` and environment-only `COSMOS_API_KEY`. Setup skips local
GPU serving for BYOM; do not write keys into the notebook or source. An endpoint's
model display name does not prove its checkpoint revision.

## Dataset identity, attribution, and source cuts

Dataset: [Perception Test](https://github.com/google-deepmind/perception_test),
by the Perception Test authors / Google DeepMind; see the authors' [dataset
README](https://github.com/google-deepmind/perception_test/blob/3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf/README.md)
and [paper](https://arxiv.org/abs/2305.13786). Their repository identifies code as
Apache 2.0 and other materials as **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**.
Retain attribution, license link, and describe modifications when redistributing
derived materials. The launchable protocol below is an adaptation, not an
endorsement or additional commercial clearance determination.

`vlmeval/perception_test_local.py` pins direct official public URLs and verifies
SHA-256 before safe extraction. It retains the official README locally. The
validation annotation is `mc_question_valid.json`; source frame bounds are
`cut_frame_mapping_valid.json`. Source repository revision is
`3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`. Questions are enumerated by numeric
`video_<N>` and official question ID, retaining original option order, so full
stable indices are 0–19139. Video and question identity are retained separately.
The full archive may contain videos outside the MCQ selection; these are not
additional scored questions.

Prototype frame protocol:

- Exactly **16** uniformly sampled frames in the source-allowed interval.
- Source cut `-1` means all frames; a positive cut is the **exclusive** stop
  frame. Every selected MCQ video must have a mapping; 362 of the 5,260 MCQ
  videos have positive cuts. Bad cuts are rejected, not silently clamped.
- Aspect-preserving **downscale only**, maximum frame side **768**, no crop.
- Send sampled images directly; never re-encode a video and let the endpoint
  choose a different sampling policy.

This is a **launchable prototype protocol**, not reference implementation parity
or a published Nano score reproduction. The text-to-option parser is an adapter
protocol, not a claim of official letter-parser parity.

## Controller and native APIs

The notebook calls `launchable.py`; the same public controller is available
inside the evaluator terminal:

```bash
python launchable.py download --scope metadata  # annotations, source cuts, README
python launchable.py download --scope smoke     # also official sample videos
python launchable.py download --scope full      # also official validation videos
python launchable.py prepare --smoke
python launchable.py prepare                    # full coverage required
python launchable.py run --smoke --mode all --work-dir /results/new-smoke
python launchable.py results --work-dir /results/new-smoke
```

`run --mode infer` retains predictions without scoring; `--mode all` performs
both. Every work directory and scoring directory must be fresh; failed runs and
their logs remain available for diagnosis and are not eligible for scoring.
`--dry-run` prepares real selected media and writes a plan without model access;
it is never a scored run. The controller fixes `--nframe 16` to the documented
prototype protocol.

Native `run.py` configuration uses:

```json
{"data":{"PerceptionTest":{"class":"PerceptionTestLocal","dataset":"PerceptionTest","data_root":"/perception-test","nframe":16,"smoke":true}}}
```

The controller composes that with `cosmos_eval/models/cosmos.json`. Native
predictions are `cosmos_PerceptionTest.tsv` (or XLSX/JSON) in the native dated run
directory. Helper API in `vlmeval.perception_test_local`:

- `fetch_dataset(root, scope='smoke')`: metadata / smoke / full public download.
- `prepare_dataset(root, smoke=False, require_media=True)`: trusted annotation,
  cut and scope manifest, stable selection, coverage, separate media hashes.
- `score_predictions(predictions, root, output_dir, smoke=False, nframe=16)`:
  reconstruct trusted labels/identities and score locally into a fresh directory.

## Score existing predictions without videos or a model

Start only authenticated Jupyter with `bash setup.sh --score-only` (add `--brev`
for private mesh access). Set `SCORE_ONLY = True`, choose explicit `SCORE_SCOPE`,
and set exactly one of `EXISTING_WORK` or `PREDICTIONS_FILE` in the notebook.
The notebook downloads **metadata only** and prepares with
`require_media=False`; video files, model construction, endpoint requests and GPU
are not required. Set `DOWNLOAD_PUBLIC = False` to rescore completely offline
with already staged pinned metadata. Equivalent evaluator-terminal commands:

```bash
python launchable.py score --scope smoke --work-dir /results/inferred-run \
  --score-dir /results/fresh-rescore
python launchable.py score --scope full --predictions /results/predictions.tsv \
  --score-dir /results/fresh-full-score
```

Annotation/cut/scope fingerprints must match inference even after videos are
removed; media hashes are separate, not part of that canonical identity. Exact
selected IDs are required: missing, duplicate, unexpected, blank, or
transport-failure rows reject the run before scoring. A malformed or ambiguous
**nonempty** model answer is wrong and counted invalid, never silently dropped.
The literal model text `nan` is not treated as a missing value. Prediction-supplied
labels never override trusted annotations. Top1 is `correct / all selected
questions`, with per-area, per-reasoning, and per-tag diagnostics; no network judge,
dummy answers, random answers, or partial-full-score fallback.

## Results and lifecycle

The notebook shows one concise human result and links to the score summary,
run provenance and log. `run.json` records source `perception-test-local`, scope,
annotation and media fingerprints, source hashes, model/source revisions,
generation parameters, frame protocol, and UTC start/end plus measured elapsed
seconds. Score artifacts include `summary.json` and `controller.json` with local
scoring timing/provenance. No cost or peak-memory estimate is invented.

```bash
# Host terminal: preserve mounted data/results while stopping containers.
docker compose --profile local stop
```

Stopping containers does not stop underlying Brev VM billing; stop the VM
separately after retaining needed results. Do not commit credentials, videos,
labels, prediction tables, or answer-bearing artifacts.

## Observed validation · 2026-10-30

- On an existing Brev instance with one RTX PRO 6000 Blackwell Server Edition
  (96 GB), the actual Jupyter **Run All Cells** completed all four code cells.
  Nano returned five valid choices for the exact smoke IDs; local top1 was
  **4/5 (80%)**, with zero malformed answers.
- First local score was produced about **44 seconds** after Run All, using an
  already-running model and staged smoke inputs. Controller inference took
  20.53 seconds and local scoring 0.74 seconds. This is **not a cold launch**.
- A separate CPU container with **network disabled and no videos or GPU**
  rescored the actual predictions identically, including diagnostic counts and
  annotation/prediction identities.
- All **43 native behavioral checks** passed in the Linux CPU evaluator.

These observations cover five questions from one video, not full validation,
the held-out test leaderboard, reference-score parity, BYOM, or a fresh-VM
launch. Driver was 595.112.07; peak memory and cost were not measured.
