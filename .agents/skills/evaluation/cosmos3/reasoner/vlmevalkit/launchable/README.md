# Cosmos3-Nano · Public video benchmarks on Brev

The authenticated notebook overview links to **MMSI-Video-Bench** (the default
full reference run), **Perception Test**, **VideoPhy2**, and
**[your labeled video data](Customer_Data.ipynb)**. Each page runs inference
and local scoring with **Run → Run All Cells**; no manual Docker benchmark
command is required. Source notebooks contain no executed outputs.

**Customer data needs labels, not just footage:** questions, answer options and
actual correct answers, plus permission to process the videos. Read the
**[standalone customer-data recipe](CUSTOM_DATA_RECIPE.md)** before provisioning
a GPU. Existing coding agents perform the one-off conversion; this is not a
benchmark builder, ingestion agent, packaging workflow or automatic converter.

The launchable uses no private annotations, internal dataset service or dataset
credentials, external submission, or API judge. Published Cosmos reference
configurations have not been independently reproduced; the included
**[expected pinned measurements](baselines.json)** state that boundary.

## Start on Brev

Use a native Linux x86-64 GPU instance with Docker Compose, NVIDIA container
support, and enough GPU capacity for the pinned public Nano checkpoint. Put data,
model cache, and results on a disk with sufficient space **and the persistence
you need**; instance-local NVMe can be lost when a VM stops or is replaced.

| `BENCHMARK` | Full public scope | Protocol / primary metric | Download budget before extraction |
| --- | --- | --- | --- |
| `perception-test` | Validation MCQ: 19,140 questions / 5,260 videos | 16 source-cut-aware frames; top1 accuracy | Smoke: 225 MB. Full: 75.38 GB |
| `videophy2` | Test: 3,397 video-caption pairs | 32 × 224 × 224 frames; mean SA/PC Pearson ×100, **not accuracy** | 3.93 GB publisher archive plus original URLs for absent entries; smoke may also need the archive |
| `mmsivideo-bench` | Main: 1,106 questions | Uniform-50 canonical frames plus references; official-extractor accuracy | Both scopes: about 16.1 GB frames plus references; no original videos |

Allow additional extraction space, model cache, and outputs. A small question
smoke does not imply a small download. Perception Test uses public validation,
not the held-out test leaderboard.

From this directory on the host:

```bash
export LAUNCHABLE_STATE_DIR=/mnt/cosmos3-perception-test-state
# Optional separate data disk; defaults to $LAUNCHABLE_STATE_DIR/perception-test.
export PERCEPTION_TEST_DATA_DIR=/mnt/perception-test-data
# Other benchmark data; defaults to $LAUNCHABLE_STATE_DIR/datasets.
export BENCHMARK_DATA_DIR=/mnt/public-benchmark-data
bash setup.sh --brev
```

Without Brev, `bash setup.sh` binds Jupyter to loopback by default; use an SSH
tunnel. `--brev` discovers the private `wt0` mesh address and refuses public or
wildcard binding. Jupyter always requires a generated token. Retrieve the login
URL in the host terminal with:

```bash
docker compose exec perception-eval jupyter server list
```

Open **[Evaluate_Cosmos3.ipynb](Evaluate_Cosmos3.ipynb)**, the benchmark overview.
Its links work immediately, without running any cells:

- **[Perception Test](Perception_Test.ipynb)**
- **[VideoPhy2](VideoPhy2.ipynb)**
- **[MMSI-Video-Bench](MMSI_Video_Bench.ipynb)**
- **[Your labeled video data](Customer_Data.ipynb)** — follow the
  [recipe](CUSTOM_DATA_RECIPE.md) to supply canonical JSONL and configuration.

Each notebook has its own title and an **All benchmarks** back link; it does not
repeat the overview banner or catalog. The benchmark is fixed by the page.
Choose the scope in that notebook's configuration cell:

```python
SCOPE = "full"  # MMSI default; use "smoke" only for a pipeline check
```

Perception Test and VideoPhy2 still default to smoke. Only a complete full run
can be compared with a full baseline. The expected metric, model revision,
scope and protocol are displayed before a full run; paper values are labeled
separately from measured launchable values.

**Is my setup working?** For MMSI-Video-Bench with the pinned
`nvidia/Cosmos3-Nano`, expect **27.94%** official accuracy. A full run is in the
expected range when all 1,106 questions are scored and accuracy is within
**±1.0 point** (26.94–28.94%, about 298–320 correct). The result card reports
whether the run is inside that range. Outside it means check the setup against
the pinned model, serving image and protocol; it is not proof the setup is
broken. The paper's 27.2% is a reference, not the target. Other models (BYOM)
have no expected range, so their result cards show none.

Run all cells in the selected benchmark notebook. It downloads pinned public
inputs, checks coverage **before model requests**, streams real progress, scores
locally, and displays its metric, coverage, warnings, protocol, and private
artifact links. All three pages reuse `benchmarks.py` and `notebook_ui.py`;
there are no separate runner implementations or additional model instances.
Result identity must match both benchmark and scope before display. Every run
gets a fresh results folder under its benchmark's own directory.
Run one notebook at a time; `API_NPROC` supports 1–4 workers.

The existing Compose project remains `cosmos3-perception-test`; the evaluator
service remains `perception-eval`. Perception data is mounted at
`/perception-test`; other datasets at `/datasets/<benchmark>`; cache at `/cache`;
results at `/results/<benchmark>/<scope>-<UTC timestamp>`. Host paths must be
absolute and outside the checkout, with data separate from cache/results.
No Docker socket, cloud credential directory, model weights, videos, annotations,
or answer-bearing results are baked into the evaluator. The model has no host port.
Only authenticated Jupyter is exposed on the selected private address.

### Public model and serving pins

- Model: [`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano), revision
  `e59a53c25979a090fa8706c9acc0c254a6e89b92`.
- Serving image: `vllm/vllm-openai:cosmos3@sha256:db0bb920b0b54e82ea96a98659bbd21921f87d0dcfc86feffdafa2db3f08be55`.
- Perception Test: existing `CosmosReason2` recipe, temperature 0, seed 1,
  thinking disabled, maximum 8,192 output tokens.
- VideoPhy2: temperature 0, seed 42, thinking disabled, maximum 256 output tokens.
  Every accepted response must have 16 bounded groups of 49 video tokens
  (784 total), consistent with 32 frame slots and a 224×224 spatial grid under
  the pinned processor profile—not proof of decoded or unique frames.
  Incompatible endpoints fail the audit.
- MMSI-Video-Bench: temperature 0, seed 1, maximum 2,048 output tokens, low image
  detail and processor maximum 151,200 pixels. Server thinking defaults are not
  overridden. Invalid and truncated answers stay in the denominator, with issues
  reported rather than selectively regenerating outputs.

Public downloads do not require a Hugging Face token. For an existing compatible
endpoint, export complete `COSMOS_API_BASE` (`.../v1/chat/completions`), exact
`COSMOS_MODEL`, and explicit `COSMOS_MODEL_REVISION` before setup. Optionally set
`COSMOS_SERVING_IMAGE` and environment-only `COSMOS_API_KEY`. Setup skips local
GPU serving for BYOM; do not write keys into the notebook or source. An endpoint's
model display name does not prove its checkpoint revision.

For a model other than the pinned Nano, also export `COSMOS_PROCESSOR_PROFILE`:
a JSON object copied from that checkpoint's processor configuration, with
`patch_size`, `spatial_merge_size`, `temporal_patch_size`, `image_token_id`,
`video_token_id`, `vision_start_token_id`, `vision_end_token_id`, and
`profile_model` set to the exact served model name (optionally
`profile_revision`). VideoPhy2's input audit requires it; without it, other
benchmarks run but report input geometry as unknown. A profile whose
`profile_model` differs from the served model is rejected. Do not copy Nano's
values to another model unless its own configuration matches them.

The serving configuration disables FlashInfer sampling with
`VLLM_USE_FLASHINFER_SAMPLER=0`. This exact pinned vLLM build
(`0.23.1rc1.dev1306+gbfee6a802`) does **not** expose
`--trust-request-mm-kwargs`; its renderer forwards request processor overrides
unconditionally. Adding that unsupported flag would prevent startup. Effective
input evidence, not the presence of a flag, is the verification boundary.

## Customer-data recipe and model comparison

Follow [CUSTOM_DATA_RECIPE.md](CUSTOM_DATA_RECIPE.md) and then
[Customer_Data.ipynb](Customer_Data.ipynb). Supply a fixed external JSONL
manifest with stable IDs, videos, questions, 2–26 ordered candidates and one
full-text ground-truth answer per row. Optional category and temporal bounds
are documented. The config names the dataset, manifest/media paths, model
revision, frame/pixel budget and generation settings; no public benchmark
checksum or row-count gate is weakened.

Validation checks every selected input before inference. Scores use local
strict single-letter MCQ accuracy, with invalid/truncated answers counted
wrong. Transport failures cannot become a completed score. Change only the
model configuration/endpoint for comparison, retain the same dataset identity,
and inspect effective input budgets. Custom results are **not** official
benchmark scores or leaderboard submissions.

## VideoPhy2 and MMSI-Video-Bench

`benchmarks.py` dispatches the notebook's selection. `videophy2.py` and
`mmsivideo.py` contain benchmark-specific public download, qualification, native
inference, local scoring, and provenance. Dataset/code revisions and SHA-256
digests are pinned in those modules; no runtime credentials are committed.

VideoPhy2 uses the official categorical 1–5 rating questions/parser and segment-
center sampling. The publisher archive covers most rows; missing entries use
only their exact official CSV URLs. All selected videos must be verified before
inference. Its two-row smoke may have undefined Pearson correlation because
ratings do not vary; `null` is intentional, not a fabricated zero or failure.
Attribution: [VideoPhy2 / Hritik Bansal and collaborators](https://github.com/Hritikbansal/videophy/tree/01d366c979e4a82875852adbf427ccc036e7373a/VIDEOPHY2);
publisher code/data license MIT.

MMSI uses maintained `MMSIVideoBench` prompts and Uniform-50 sampling, with
selected IDs, options, labels, prompts, frames, and references checked against the
official public annotations. Its primary scorer is the pinned publisher's
`evaluation.py` extractor. Maintained no-judge exact matching is retained only as
a diagnostic because it can reject valid verbose answers. Full scope also runs
the unmodified official scorer CLI and checks agreement. Attribution:
[MMSI-Video-Bench / Jingli Lin et al.](https://github.com/InternRobotics/MMSI-Video-Bench/tree/412f5a4a9eb6d89613c60b541b17df74f11a6736),
arXiv:2512.10863; publisher CC BY 4.0. Underlying third-party redistribution rights
are not independently cleared; retain media locally, not in the image or PR.

Both adapters retain raw responses, exact IDs, failure/finish reasons, source and
model revision records, and measured timing in each fresh run folder. Transport
or incomplete coverage cannot produce a completed full score. Neither protocol
is claimed to reproduce the paper's Nano references (45.6 / 27.2); the exact
reference recipes remain unconfirmed. Do not average these distinct metrics.

Equivalent evaluator-terminal commands, **not required by the notebook**:

```bash
python benchmarks.py download --benchmark videophy2 --scope smoke --data-root /datasets/videophy2
python benchmarks.py run --benchmark videophy2 --scope smoke --data-root /datasets/videophy2 \
  --work-dir /results/videophy2/new-smoke
```

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

## Perception-specific controller and native APIs

For Perception Test, `benchmarks.py` delegates to `launchable.py`. Its native
controller is also available inside the evaluator terminal:

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
for private mesh access). Open **Perception_Test.ipynb**, set `SCORE_ONLY = True`,
choose `SCOPE`, and set exactly one of `EXISTING_WORK` or `PREDICTIONS_FILE`.
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

The notebook shows one identity-checked result card with metric units, selected
coverage, measured time, warnings, protocol, and private artifact links. Every
benchmark has a `summary.json`; Perception's original native report remains at
`scores/summary.json`. Score-only Perception retains its native `summary.json`
and writes the notebook report separately as `notebook-summary.json`.

Each benchmark retains its own provenance and predictions. Perception's
`run.json` and `scores/controller.json` additionally record native annotation/cut
identities, source/model revisions, frame/generation protocol, and timing.
VideoPhy2 and MMSI retain `provenance.json` plus per-response evidence and local
scorer diagnostics. No cost or peak-memory estimate is invented.

`execution-summary.json` aggregates per-response **requested settings and
observed input evidence**. Supported explicit processor profiles allow
conditional derivation of image-grid pixel areas and video frame slots from
bounded visual-token groups. Frame slots are not proof of decoded/unique
frames; area is not width/height/crop. Effective generation settings and
checkpoint revision are not attested by ordinary chat responses and remain
unknown. Unknown evidence is never presented as verification. Customer runs
also retain `requested-execution.json` and their immutable data identity.

Reviewed aggregate expected measurements in `baselines.json` are deliberately
shipped; label-bearing runtime reports, media and predictions are not.

```bash
# Host terminal: preserve mounted data/results while stopping containers.
docker compose --profile local stop
```

Stopping containers does not stop underlying Brev VM billing; stop the VM
separately after retaining needed results. Do not commit credentials, videos,
labels, prediction tables, or answer-bearing artifacts.

## Kaizen UI presentation and developer skills

The notebook adapts **Kaizen UI 1.19** typography, semantic color tokens,
density spacing, entity cards, status labels, and read-only key/value patterns
to native notebook HTML in `notebook_ui.py`. Jupyter owns the application shell
and light/dark theme. No global CSS reset, downloaded fonts, React application,
CDN request, or Node runtime is required to run the notebook.

The benchmark configuration stays visible. Helper, execution, and result-rendering
inputs are collapsed by default and can be expanded. Native keyboard-accessible
disclosures retain the full protocol, attribution, and scoring notes.
After updating an already-open notebook, restart its kernel before Run All so
Python loads the updated presentation helper.

The overview is static Markdown/HTML, so its navigation does not depend on code
execution or saved outputs. Its conservative markup uses Jupyter's native link
colors because Jupyter sanitizes Markdown CSS. The richer run/result panels
remain scoped to `notebook_ui.py`; no application shell is duplicated.

Optional design-tool setup, from the **Framework repository root**:

```bash
npm ci
```

The root `package.json` pins `@nvidia/foundations-react-core` as a development-only
dependency. Its `prepare` script runs `kui-sync-skills --dir .agents`, generating
`.agents/skills/kaizen-ui` locally; that generated folder is not committed.
`kui.disablePostInstall` suppresses the install hint. Keep the lockfile in source
control; `node_modules` is excluded from Git and Docker.

