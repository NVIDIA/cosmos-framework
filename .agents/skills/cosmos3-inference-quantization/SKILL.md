---
name: cosmos3-inference-quantization
description: >
  Quantize Cosmos3 generator checkpoints to static-scale FP8 with the
  cosmos_framework.quantization recipe (NVIDIA TensorRT Model Optimizer): the Edge,
  Nano and Super video models and the Super Text-to-Image / Image-to-Video base and
  4-step distilled students, exported as drop-in Diffusers-layout checkpoints with
  calibrated scales and an hf_quant_config.json that load on vLLM-Omni. Use when the
  user asks "quantize Cosmos3 to FP8", "make / calibrate / export my own FP8 Cosmos3
  checkpoint", "run quantize_fp8_checkpoint", "DEMO=1 vs DEMO=0 calibration", or
  "what does the FP8 export contain". Do not use it to run inference with a published
  FP8 checkpoint (nvidia/Cosmos3-Edge-FP8, -Nano-FP8, -Super-FP8) — that is the FP8
  route of cosmos3-generate-media and of the Reasoner / Action / Transfer cookbook
  notebooks (their skills follow in later rows) — nor for the framework's load-time
  --quantization-method flags.
---

# Cosmos3 FP8 Quantization

One skill for part 2 of the retired quantization cookbook — making your own FP8 checkpoint: three variants (= the
three notebooks) that share one API call and one output layout. The README (`references/quantization-readme.md`):
"Convert the Cosmos3 checkpoints to static-scale **FP8** with NVIDIA TensorRT Model Optimizer (ModelOpt). FP8 reduces
weight storage and can accelerate inference on NVIDIA GPUs with FP8 tensor cores; total memory also includes
activations and unquantized components. Quality depends on calibration and the task. The output is a Diffusers-layout
checkpoint (FP8 weights + per-tensor `weight_scale` / `input_scale` sidecars + `hf_quant_config.json`) that loads on
vLLM-Omni." "The recipe lives in `cosmos_framework.quantization`: checkpoint loading, calibration, FP8 export and
metadata assembly are owned by cosmos-framework. These notebooks are walkthroughs of that API." Part 1 of the
cookbook — serving a published or self-made FP8 checkpoint — is the FP8 route of the inference skills ("Serving the
FP8 output" below).

## Read this first

Before the first command of any variant — including a `DEMO=1` run:

1. Read the variant's reference file (linked under its heading) in full — § 1 Prerequisites, § 2 Configure Paths and
   Environment, § 3 Install Dependencies, § 5 Verify GPU and Python Environment, § 6 Load the Quantization Toolkit,
   then the model section(s) — and `references/quantization-readme.md` (Setup (once), Demo vs. full runs, What gets
   produced, Inference). The variant sections below only summarise the notebooks; the reference files are the
   notebooks migrated 1:1 (every markdown and code cell, in order). Do not look for the original `.ipynb` files: this
   migration retired them, the reference files replace them, and the README's `jupyter nbconvert` commands therefore
   have no target in this checkout (each variant quotes its command as text and gives the executable path: the
   notebook's cells, run as ONE script).
2. Check the GPU with `nvidia-smi`. The CUDA major version picks the dependency group (`cu130-train`; "cu128-train on
   CUDA 12.8", README Setup block). The memory decides which section can run: "By default **DEMO=1** uses one
   calibration prompt and a smaller shape; it still loads the full checkpoint and runs every sampler step" (README) —
   `DEMO` never shrinks the model (table below).
3. Assets: none are vendored — the notebooks ship none. Skill note (not in the source): the calibration prompts are
   streamed at run time by `cosmos_framework/quantization/calibration.py` (`prepare_calibration_prompts`: the Hugging
   Face dataset `WenhaoWang/VideoUFO`, streaming), and Variant 3's conditioning images come from `I2V_COND_DIR` or the
   dataset fallback its notebook names — so a run needs Hub egress and `HF_TOKEN` (gated checkpoints), and there is
   nothing to materialise.
4. Set `OUTPUT_ROOT` (notebook § 2 override) to a large volume outside the checkout and outside your artifacts
   directory: "each output is roughly the size of the bf16 source" (notebook § 1). Skill note (not in the source): the
   notebook's default `OUTPUT_ROOT`, `QUANTIZE_ROOT / "outputs"`, is the `outputs/` subfolder of the retired cookbook
   directory, which the configure cell would re-create inside the checkout.
5. uv: `uv sync` in this checkout requires the uv version pinned by `pyproject.toml` (`[tool.uv]`
   `required-version = "==0.11.29"`); any other uv aborts the Setup (once) block. Skill note (not in the source): that
   block's first line installs 0.11.29 unless `uv --version` already reports it and puts `~/.local/bin` first on `PATH`.
6. After a failure, go back to the reference section that covers the failing step before improvising a fix. Verify
   what you produced (Verify the output below) and record the facts in `result_summary.md`.

## When to use this skill

- Use to convert a Cosmos3 generator checkpoint — Edge / Nano / Super video (`t2v` profile), Super Text-to-Image
  base or 4-step student (`t2i`), Super Image-to-Video base or 4-step student (`i2v`) — into a static FP8 drop-in
  checkpoint with calibrated activation scales
- Use to understand what the export contains (`hf_quant_config.json`, the
  `quantization_config.runtime.diffusion_step_policy` in `transformer/config.json`, the symlinked components) and how
  `DEMO=1` differs from a full calibration
- Not for inference with a published or self-made FP8 checkpoint: `../cosmos3-generate-media/SKILL.md` (Variant 1
  diffusers `COSMOS3_DIFFUSERS_PRECISION=fp8`, Variant 3 vLLM-Omni "Quantized Checkpoints", Variant 4 SGLang "FP8")
  and the Reasoner / Action / Transfer cookbook notebooks listed in the README reference's Inference table carry that
  route (their skills follow in later rows; "Serving the FP8 output" below)
- Not for the framework's load-time quantization flags (`--quantization-method {mxfp8,nvfp4,fp8}`;
  `docs/inference.md` § Model Arguments: "Its exported weights, scales, and activation policy are distinct from the
  load-time quantization flags below") — see `../cosmos3-inference/SKILL.md`
- Environment or import failures: `../cosmos3-env-troubleshoot/SKILL.md`; checkout and uv: `../cosmos3-setup/SKILL.md`

## Bundled assets

| Asset  | Path | Used by                                                                                                                           |
| ------ | ---- | --------------------------------------------------------------------------------------------------------------------------------- |
| (none) | —    | the notebooks reference no files; calibration prompts and the I2V fallback images are datasets fetched at run time (item 3 above) |

## Model sizes vs GPU

| Checkpoint (notebook default; local override variable)                                                           | Size, as the sources state it                                                                                                                                                                                                                 | GPU, as the sources state it                                                                                                                                                                                   |
| ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `nvidia/Cosmos3-Edge` (`C3_EDGE_DIR`)                                                                            | "Edge (4B)" — "the 4B base checkpoint, the fastest to quantize and a good first run"                                                                                                                                                          | "Edge (4B) uses the same recipe with a smaller checkpoint; allow room for those components too" (§ 1)                                                                                                          |
| `nvidia/Cosmos3-Nano` (`C3_NANO_DIR`)                                                                            | "Nano (16B)" — "the same recipe as Edge with a larger model"                                                                                                                                                                                  | "Nano (16B) also needs room for BF16 weights, activations, and quantizer state" (§ 1)                                                                                                                          |
| `nvidia/Cosmos3-Super` (`C3_SUPER_DIR`)                                                                          | "Super (64B)"                                                                                                                                                                                                                                 | "Quantizing the Super (64B) checkpoints needs a GPU with enough memory to hold the model in bf16 plus the quantizers" (§ 1); "This is the heaviest run in the notebook; make sure your GPU has enough memory." |
| `nvidia/Cosmos3-Super-Text2Image` (`C3_T2I_DIR`), `nvidia/Cosmos3-Super-Text2Image-4Step` (`C3_T2I_4STEP_DIR`)   | no size in these sources; base and student are "the *same* network" served with different samplers (the generate-media figure for the 4-step students: skill note below the table)                                                            | the Super (64B) sentence of § 1 (repeated in both Super notebooks)                                                                                                                                             |
| `nvidia/Cosmos3-Super-Image2Video` (`C3_I2V_DIR`), `nvidia/Cosmos3-Super-Image2Video-4Step` (`C3_I2V_4STEP_DIR`) | as above                                                                                                                                                                                                                                      | as above                                                                                                                                                                                                       |

Disk (§ 1 of every notebook): "You also need free disk for the Hugging Face checkpoint cache and for the FP8 output
(each output is roughly the size of the bf16 source). Point `HF_HOME` at a large volume in the next step." Access
(§ 1): "Use a Linux machine with an NVIDIA GPU, model access on Hugging Face, and either `uvx hf@latest auth login` or
`HF_TOKEN` set." Skill note (not in the source): on one 80 GB or 94 GB GPU the Edge section is the single-GPU path this
skill's eval exercises; Nano is unverified by any run of this skill. `../cosmos3-generate-media/SKILL.md` measured
131.4 GB = 122.4 GiB of bf16 safetensors per 4-step student (a generate-media figure, not one of this skill's sources),
so the 4-step students exceed such a GPU in bf16 by that measured figure; the 64B Super video checkpoint and the Super
base models have no measured size in these sources (unknown) — attempt any Super section only to document the measured
blocker, never fabricate an output.

## Setup (once) — the README block

Run once per checkout, as ONE shell call, with `/path/to/cosmos-framework` replaced by your framework checkout root
(skill note: its first line installs the pinned uv 0.11.29 if needed, item 5; the notebooks' § 3 install cell repeats the
rest without the `cd`, the exports and `RUN_ARTIFACTS`). README: "Use a Linux machine with an NVIDIA GPU and Hugging
Face model access (`uvx hf@latest auth login` or `HF_TOKEN`). Follow the
[Framework environment setup](../cookbooks/cosmos3/README.md#cosmos-framework), including system libraries. Use the
local checkout, not a separate PyPI or Git installation:"

```bash
uv --version 2>/dev/null | grep -q '^uv 0\.11\.29' || { curl -LsSf https://astral.sh/uv/0.11.29/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; hash -r; }   # skill note (not in the source): the pinned uv, Read this first item 5 — the installer writes no ~/.local/bin/env when that directory is already on PATH, so PATH is exported instead of sourcing it; the README block starts on the next line
cd /path/to/cosmos-framework
export FRAMEWORK_ROOT="$PWD"
export COSMOS3_QUANTIZE_VENV="${COSMOS3_QUANTIZE_VENV:-$FRAMEWORK_ROOT/.venv}"
# cu128-train on CUDA 12.8; COSMOS3_TORCH_BACKEND=cu128 is also honored by notebooks.
export COSMOS3_UV_GROUP="${COSMOS3_UV_GROUP:-cu130-train}"
export UV_PROJECT_ENVIRONMENT="$COSMOS3_QUANTIZE_VENV"
export GIT_LFS_SKIP_SMUDGE=1
uv sync --all-extras --group="$COSMOS3_UV_GROUP"
uv pip install --python "$COSMOS3_QUANTIZE_VENV/bin/python" \
  "nvidia-modelopt[torch]==0.44.0" datasets ipykernel jupyter-client nbconvert
"$COSMOS3_QUANTIZE_VENV/bin/python" -m ipykernel install --user \
  --name cosmos3-quantize --display-name "Cosmos3 Quantize (Python 3.13)"
export RUN_ARTIFACTS="$(mktemp -d /tmp/cosmos3-quantize.XXXXXX)"
```

Notebook § 3: "Run this cell once; re-running `uv sync` later may remove these added packages, so re-run the
`uv pip install` command after syncing." Notebook § 1: "**Headless servers:** if you see `libGL.so.1: cannot open
shared object file` when the VAE loads, install the system graphics libraries:" `apt-get install -y libgl1 libglib2.0-0`
(skill note: the evaluator's task container runs as root — no `sudo`; the setup guide's System packages section,
`../cookbooks/cosmos3/README.md`, says the same of NGC containers and has the full package list). README: "The notebooks work from the
Framework root or their own directory. They resolve the checkout separately from `.agents/skills`, including after
switching kernels. Open a notebook and run top to bottom, switching to **Cosmos3 Quantize (Python 3.13)** after
step 3. For headless execution, the commands below keep executed notebooks in `RUN_ARTIFACTS`; `OUTPUT_ROOT` selects
the checkpoint destination. Set `HF_HOME` / `HF_DATASETS_CACHE` for cache storage and the `C3_*_DIR` variables shown in
each notebook to use local source checkpoints." The kernel switch (notebook § 4) is Jupyter plumbing; this skill runs
the cells as a script instead (next section).

Notebook § 2 overrides (edit the placeholders; the configure cell reads them and defaults `HF_HOME` to
`~/.cache/huggingface`): `export COSMOS3_QUANTIZE_VENV=/path/to/cosmos-framework/.venv`,
`export COSMOS3_TORCH_BACKEND=cu130       # or cu128`, `export HF_HOME=/path/to/large/huggingface/cache`,
`export OUTPUT_ROOT=/path/to/fp8/outputs`.

### How this skill runs a notebook (skill note, not in the source)

The notebook files are retired, so each variant below writes the notebook's code cells — verbatim, in notebook order,
taken from the reference file's `python` fences — into ONE script under `RUN_ARTIFACTS` and runs it with the venv's
python from the framework checkout root (the notebook's `find_framework_root()` walks up from the cwd to the directory
that holds `pyproject.toml` and `cosmos_framework/`). Cell numbers are the 0-based index of the `python` fences in the
reference: 0 = § 2 configure (the § 4 "restore cell", index 1, is the same code and is skipped), 2 = § 5 GPU and
import check, 3 = § 6 toolkit import, `DEMO` / `NUM_SAMPLES` and `inspect_checkpoint`, then the model sections'
Quantize and Inspect cells as listed per variant. Each block is ONE shell call run from this skill directory (the one
with `references/`; or set `SKILL_DIR`). A new shell call has none of the Setup (once) block's exports, so each block
repeats that block's `FRAMEWORK_ROOT` / `COSMOS3_QUANTIZE_VENV` / `RUN_ARTIFACTS` values as defaults — edit or export
`FRAMEWORK_ROOT` first — and stops before any download when `OUTPUT_ROOT` is unset (item 4), the checkout or its venv is
missing, or the reference's `python` fence count is not the expected one (an added or re-ordered fence would shift every
index and run the wrong cell). The run prints the recipe's progress lines (`[init] gen_layers=…`, `[calib] <mode>
diffusion calibration — <n> prompts x <steps> steps`, `[quant] format=fp8 algo=max`, `[assemble] wrote …/hf_quant_config.json`,
`[done] <output_dir>`; printed by `cosmos_framework/quantization/__init__.py`, `cosmos_framework/quantization/calibration.py`
and `cosmos_framework/quantization/export.py`) and then the inspect summary; `tee` keeps them in `RUN_ARTIFACTS`.

## Variant 1 — Edge, Nano and Super video models

Reference: [`references/quantize-cosmos3.md`](references/quantize-cosmos3.md) — read § 1–6 and the model section
before the first command. README: "Convert the base Edge (4B), Nano (16B), and Super (64B) checkpoints with the same
recipe. The notebook runs Edge first, then Nano and Super." Notebook: "All three models use the **same recipe on
different checkpoints**. We quantize Edge first, then Nano and Super."

Edge — "Load `nvidia/Cosmos3-Edge`, calibrate it, and write the FP8 checkpoint to `OUTPUT_ROOT/edge-fp8`. In DEMO mode
this uses one calibration prompt at a small shape; set `DEMO=0` for the full calibration run." The Quantize cell,
verbatim (it needs the § 2, § 5 and § 6 cells in the same Python process — run it through the block below, not alone),
followed by the Inspect cell `inspect_checkpoint(OUTPUT_ROOT / 'edge-fp8')`:

```python
model_name_or_path = os.environ.get('C3_EDGE_DIR', "nvidia/Cosmos3-Edge")
output_dir = OUTPUT_ROOT / 'edge-fp8'

quantize_fp8_checkpoint(
    model_name_or_path=model_name_or_path, output_dir=output_dir,
    profile='t2v', sampler=SAMPLER_VIDEO_BASE, shape=SHAPE_VIDEO_DEMO if DEMO else SHAPE_VIDEO,
    num_samples=NUM_SAMPLES, calibration_behavior='framework',
)
```

Nano ("Nano is the 16B checkpoint. It uses the same recipe as Edge with a larger model.") is the same call with
`C3_NANO_DIR` / `"nvidia/Cosmos3-Nano"` and `OUTPUT_ROOT / 'nano-fp8'`; Super ("Super is the 64B checkpoint. The call is
identical to Nano — only the checkpoint changes. This is the heaviest run in the notebook; make sure your GPU has enough
memory.") first releases the previous model (`gc.collect()`, `torch.cuda.empty_cache()`) and writes
`OUTPUT_ROOT / 'super-fp8'`.

Run (skill note, not in the source — "How this skill runs a notebook" above). Cells 4/5 = Edge Quantize + Inspect,
6/7 = Nano, 8/9 = Super; the whole notebook top to bottom is `0 2 3 4 5 6 7 8 9`:

```bash
set -o pipefail
export FRAMEWORK_ROOT="${FRAMEWORK_ROOT:-/path/to/cosmos-framework}"   # skill note: the Setup (once) values, repeated — a new shell call has none of its exports
export COSMOS3_QUANTIZE_VENV="${COSMOS3_QUANTIZE_VENV:-$FRAMEWORK_ROOT/.venv}"; export RUN_ARTIFACTS="${RUN_ARTIFACTS:-$(mktemp -d /tmp/cosmos3-quantize.XXXXXX)}"
SKILL_DIR="${SKILL_DIR:-$PWD}"; : "${OUTPUT_ROOT:?set OUTPUT_ROOT to a large volume outside the checkout and outside your artifacts directory}"
test -f "$FRAMEWORK_ROOT/pyproject.toml" -a -d "$FRAMEWORK_ROOT/cosmos_framework" || { echo "FRAMEWORK_ROOT=$FRAMEWORK_ROOT is not a cosmos-framework checkout"; exit 1; }
test -x "$COSMOS3_QUANTIZE_VENV/bin/python" || { echo "$COSMOS3_QUANTIZE_VENV/bin/python is missing: run the Setup (once) block first"; exit 1; }
test -f "$SKILL_DIR/references/quantize-cosmos3.md" || { echo "run this block from the skill directory or set SKILL_DIR"; exit 1; }
python3 - "$SKILL_DIR/references/quantize-cosmos3.md" "$RUN_ARTIFACTS/quantize_cosmos3_edge.py" 0 2 3 4 5 <<'PY'
import pathlib, re, sys
cells = re.findall(r"^```python\n(.*?)\n```$", pathlib.Path(sys.argv[1]).read_text(), re.S | re.M)
len(cells) == 10 or sys.exit(f"reference changed: {len(cells)} python fences, expected 10 - re-check the cell indices")
pathlib.Path(sys.argv[2]).write_text("\n\n".join(cells[int(i)] for i in sys.argv[3:]) + "\n")
PY
cd "$FRAMEWORK_ROOT" && DEMO=1 "$COSMOS3_QUANTIZE_VENV/bin/python" "$RUN_ARTIFACTS/quantize_cosmos3_edge.py" 2>&1 | tee "$RUN_ARTIFACTS/quantize_cosmos3_edge.log"
```

The README's own "Run" command for this notebook, quoted as text (it executes the retired `.ipynb` through
`nbconvert`, so it has no target in this checkout): `cd "$FRAMEWORK_ROOT/.agents/skills/cookbooks/cosmos3/quantization"`
then `DEMO=1 "$COSMOS3_QUANTIZE_VENV/bin/jupyter" nbconvert --to notebook --execute --output-dir "$RUN_ARTIFACTS" --ExecutePreprocessor.kernel_name=cosmos3-quantize --ExecutePreprocessor.timeout=7200 quantize_cosmos3.ipynb`.

## Variant 2 — Super Text-to-Image (base and 4-step distilled)

Reference: [`references/quantize-super-text2image.md`](references/quantize-super-text2image.md) — read § 1–6 and both
model sections before the first command. README: "Quantize the full-quality Text-to-Image model, then its 4-step
distilled student. The two use different checkpoint weights and sampler presets (50-step UniPC vs. 4-step
FlowMatchEuler, guidance off). The notebook calibrates each separately." Notebook: "The two are the *same* network;
they are served with a different **sampler** (scheduler, number of steps, guidance). Calibration replays the sampler
each version actually uses, so we quantize each one with its own sampler".

Base — "The full-quality text-to-image model. It is served with a 50-step UniPC sampler, so that is what we
calibrate." The Quantize cell, verbatim (same rule as Variant 1: run it through the block below), followed by
`inspect_checkpoint(OUTPUT_ROOT / 'super-t2i-fp8')`:

```python
model_name_or_path = os.environ.get('C3_T2I_DIR', 'nvidia/Cosmos3-Super-Text2Image')
output_dir = OUTPUT_ROOT / 'super-t2i-fp8'

quantize_fp8_checkpoint(
    model_name_or_path=model_name_or_path, output_dir=output_dir,
    profile='t2i', sampler=SAMPLER_IMAGE_BASE, shape=SHAPE_IMAGE_DEMO if DEMO else SHAPE_IMAGE,
    num_samples=NUM_SAMPLES, calibration_behavior='framework',
)
```

Distilled — "The distilled model is the **same network** as the base, but it is served in just 4 steps. The only thing
that changes for quantization is the **sampler** — so calibration replays the 4-step schedule instead of the 50-step
one"; "The recipe reads the correct scheduler and step schedule from each checkpoint, so from your side the call below
is identical to the base — only `sampler=SAMPLER_DISTILLED` changes" (with `C3_T2I_4STEP_DIR` /
`'nvidia/Cosmos3-Super-Text2Image-4Step'`, `OUTPUT_ROOT / 'super-t2i-distilled-fp8'`, after the cell's `gc.collect()` /
`torch.cuda.empty_cache()` lines). The notebook's sampler table:

|           | scheduler                      | steps | guidance           |
| --------- | ------------------------------ | ----- | ------------------ |
| base      | UniPC                          | 50    | 6.0                |
| distilled | FlowMatchEuler (4 fixed steps) | 4     | 1.0 (guidance off) |

Run (skill note, not in the source). Cells 4/5 = base Quantize + Inspect, 6/7 = distilled; the whole notebook is
`0 2 3 4 5 6 7`:

```bash
set -o pipefail
export FRAMEWORK_ROOT="${FRAMEWORK_ROOT:-/path/to/cosmos-framework}"   # skill note: the Setup (once) values, repeated — a new shell call has none of its exports
export COSMOS3_QUANTIZE_VENV="${COSMOS3_QUANTIZE_VENV:-$FRAMEWORK_ROOT/.venv}"; export RUN_ARTIFACTS="${RUN_ARTIFACTS:-$(mktemp -d /tmp/cosmos3-quantize.XXXXXX)}"
SKILL_DIR="${SKILL_DIR:-$PWD}"; : "${OUTPUT_ROOT:?set OUTPUT_ROOT to a large volume outside the checkout and outside your artifacts directory}"
test -f "$FRAMEWORK_ROOT/pyproject.toml" -a -d "$FRAMEWORK_ROOT/cosmos_framework" || { echo "FRAMEWORK_ROOT=$FRAMEWORK_ROOT is not a cosmos-framework checkout"; exit 1; }
test -x "$COSMOS3_QUANTIZE_VENV/bin/python" || { echo "$COSMOS3_QUANTIZE_VENV/bin/python is missing: run the Setup (once) block first"; exit 1; }
test -f "$SKILL_DIR/references/quantize-super-text2image.md" || { echo "run this block from the skill directory or set SKILL_DIR"; exit 1; }
python3 - "$SKILL_DIR/references/quantize-super-text2image.md" "$RUN_ARTIFACTS/quantize_super_t2i_base.py" 0 2 3 4 5 <<'PY'
import pathlib, re, sys
cells = re.findall(r"^```python\n(.*?)\n```$", pathlib.Path(sys.argv[1]).read_text(), re.S | re.M)
len(cells) == 8 or sys.exit(f"reference changed: {len(cells)} python fences, expected 8 - re-check the cell indices")
pathlib.Path(sys.argv[2]).write_text("\n\n".join(cells[int(i)] for i in sys.argv[3:]) + "\n")
PY
cd "$FRAMEWORK_ROOT" && DEMO=1 "$COSMOS3_QUANTIZE_VENV/bin/python" "$RUN_ARTIFACTS/quantize_super_t2i_base.py" 2>&1 | tee "$RUN_ARTIFACTS/quantize_super_t2i_base.log"
```

The README's own "Run" command, quoted as text (retired `.ipynb`, no target here):
`cd "$FRAMEWORK_ROOT/.agents/skills/cookbooks/cosmos3/quantization"` then
`DEMO=1 "$COSMOS3_QUANTIZE_VENV/bin/jupyter" nbconvert --to notebook --execute --output-dir "$RUN_ARTIFACTS" --ExecutePreprocessor.kernel_name=cosmos3-quantize --ExecutePreprocessor.timeout=7200 quantize_super_text2image.ipynb`.

## Variant 3 — Super Image-to-Video (base and 4-step distilled)

Reference: [`references/quantize-super-image2video.md`](references/quantize-super-image2video.md) — read § 1–6,
Conditioning Images and both model sections before the first command. README: "Same as Text-to-Image, but the
image-to-video regime: calibration conditions on a clean first frame (a real image, VAE-encoded) held fixed while the
rest of the clip denoises. Provide the conditioning images through `I2V_COND_DIR` (a local folder); if unset, the
notebook falls back to a public calibration image dataset."

Conditioning Images — "Point `I2V_COND_DIR` at a local folder of images to condition on. If you leave it unset, the
recipe falls back to a public calibration image dataset (`nemotron_vlm_dataset_v2`), which requires Hugging Face access
to that dataset." Its cell (index 4; every run of this notebook includes it):

```python
I2V_COND_DIR = os.environ.get('I2V_COND_DIR')  # e.g. '/data/my_cond_images'
I2V_COND_DATASET = None if I2V_COND_DIR else 'nemotron_vlm_dataset_v2'
print('conditioning from:', I2V_COND_DIR or f'dataset {I2V_COND_DATASET}')
```

Base — "The full-quality image-to-video model, served with a 50-step UniPC sampler." The Quantize cell, verbatim
(run it through the block below), followed by `inspect_checkpoint(OUTPUT_ROOT / 'super-i2v-fp8')`:

```python
model_name_or_path = os.environ.get('C3_I2V_DIR', 'nvidia/Cosmos3-Super-Image2Video')
output_dir = OUTPUT_ROOT / 'super-i2v-fp8'

quantize_fp8_checkpoint(
    model_name_or_path=model_name_or_path, output_dir=output_dir,
    profile='i2v', sampler=SAMPLER_VIDEO_BASE, shape=SHAPE_VIDEO_DEMO if DEMO else SHAPE_VIDEO,
    num_samples=NUM_SAMPLES, calibration_behavior='framework',
    i2v_cond_dir=I2V_COND_DIR, i2v_cond_dataset=I2V_COND_DATASET,
)
```

Distilled — "The distilled model is the **same network**, served in just 4 steps. As with Text-to-Image, the only
change for quantization is the **sampler**; the image conditioning is identical" — `sampler=SAMPLER_DISTILLED` with
`C3_I2V_4STEP_DIR` / `'nvidia/Cosmos3-Super-Image2Video-4Step'` and `OUTPUT_ROOT / 'super-i2v-distilled-fp8'`, after
the cell's `gc.collect()` / `torch.cuda.empty_cache()` lines; the sampler table is the one of Variant 2.

Run (skill note, not in the source). Export `I2V_COND_DIR` first when you have local images (the README's Run sets
`I2V_COND_DIR=/path/to/cond_images` inline); unset, the cell above selects the dataset fallback. Cells 5/6 = base
Quantize + Inspect, 7/8 = distilled, always with 4; the whole notebook is `0 2 3 4 5 6 7 8`:

```bash
set -o pipefail
export FRAMEWORK_ROOT="${FRAMEWORK_ROOT:-/path/to/cosmos-framework}"   # skill note: the Setup (once) values, repeated — a new shell call has none of its exports
export COSMOS3_QUANTIZE_VENV="${COSMOS3_QUANTIZE_VENV:-$FRAMEWORK_ROOT/.venv}"; export RUN_ARTIFACTS="${RUN_ARTIFACTS:-$(mktemp -d /tmp/cosmos3-quantize.XXXXXX)}"
SKILL_DIR="${SKILL_DIR:-$PWD}"; : "${OUTPUT_ROOT:?set OUTPUT_ROOT to a large volume outside the checkout and outside your artifacts directory}"
test -f "$FRAMEWORK_ROOT/pyproject.toml" -a -d "$FRAMEWORK_ROOT/cosmos_framework" || { echo "FRAMEWORK_ROOT=$FRAMEWORK_ROOT is not a cosmos-framework checkout"; exit 1; }
test -x "$COSMOS3_QUANTIZE_VENV/bin/python" || { echo "$COSMOS3_QUANTIZE_VENV/bin/python is missing: run the Setup (once) block first"; exit 1; }
test -f "$SKILL_DIR/references/quantize-super-image2video.md" || { echo "run this block from the skill directory or set SKILL_DIR"; exit 1; }
python3 - "$SKILL_DIR/references/quantize-super-image2video.md" "$RUN_ARTIFACTS/quantize_super_i2v_base.py" 0 2 3 4 5 6 <<'PY'
import pathlib, re, sys
cells = re.findall(r"^```python\n(.*?)\n```$", pathlib.Path(sys.argv[1]).read_text(), re.S | re.M)
len(cells) == 9 or sys.exit(f"reference changed: {len(cells)} python fences, expected 9 - re-check the cell indices")
pathlib.Path(sys.argv[2]).write_text("\n\n".join(cells[int(i)] for i in sys.argv[3:]) + "\n")
PY
cd "$FRAMEWORK_ROOT" && DEMO=1 "$COSMOS3_QUANTIZE_VENV/bin/python" "$RUN_ARTIFACTS/quantize_super_i2v_base.py" 2>&1 | tee "$RUN_ARTIFACTS/quantize_super_i2v_base.log"
```

The README's own "Run" command, quoted as text (retired `.ipynb`, no target here):
`cd "$FRAMEWORK_ROOT/.agents/skills/cookbooks/cosmos3/quantization"` then
`DEMO=1 I2V_COND_DIR=/path/to/cond_images "$COSMOS3_QUANTIZE_VENV/bin/jupyter" nbconvert --to notebook --execute --output-dir "$RUN_ARTIFACTS" --ExecutePreprocessor.kernel_name=cosmos3-quantize --ExecutePreprocessor.timeout=7200 quantize_super_image2video.ipynb`.

## Demo vs. full runs (README)

|                     | `DEMO=1` (default)              | `DEMO=0` (full calibration)                    |
| ------------------- | ------------------------------- | ---------------------------------------------- |
| Calibration prompts | 1                               | 8                                              |
| Shape               | small (e.g. 480×720, 29 frames) | production (720×1280, up to 189 frames)        |
| Purpose             | quick end-to-end check          | calibrate with more prompts and a larger shape |

"The FP8 **weights** are identical either way — they are computed directly from the bf16 weights
(`weight_scale = max|W| / 448`) and do not depend on calibration. DEMO only affects the calibrated activation
**scales** (fewer samples, smaller shape). Neither mode guarantees reproduction of published scales or generation
quality." The notebooks' § 6 cell: "DEMO=1 (default): one calibration prompt at a small shape, so a run takes minutes.
Set DEMO=0 for the production shape and 8 calibration prompts (a larger calibration run)." Skill note (not in the
source): the shapes behind the table are the constants of `cosmos_framework/quantization/__init__.py` —
`SHAPE_VIDEO_DEMO` 480×720×29 frames and `SHAPE_IMAGE_DEMO` 512×512×1 (DEMO=1), `SHAPE_VIDEO` 720×1280×189 and
`SHAPE_IMAGE` 720×1280×1 (DEMO=0).

## What gets produced (README)

"Each run writes a drop-in checkpoint at `OUTPUT_ROOT/<name>-fp8/`:"

```
<name>-fp8/
├── transformer/                 # FP8 weights + weight_scale/input_scale + quantization_config
│   ├── diffusion_pytorch_model-*.safetensors
│   ├── diffusion_pytorch_model.safetensors.index.json
│   ├── modelopt_state.pth       # Diffusers ModelOpt state
│   └── config.json              # includes quantization_config.runtime.diffusion_step_policy
├── modelopt_state.pth           # reasoner ModelOpt state for Transformers
├── quantization_metadata.json   # producing environment, source and calibration recipe
├── hf_quant_config.json         # FP8 quantization_config at the root
├── model.safetensors.index.json # regenerated whole-model index
└── vae/ scheduler/ ...          # symlinked back to the source checkpoint
```

"Point your serving stack at this directory." Mixed activation precision (README and notebook § 6): "These are static
checkpoint FP8 weights and calibrated scales, exported with `calibration_behavior="framework"`; this is separate from
quantizing BF16 weights at server load time. The exporter records the activation policy in `transformer/config.json`
under `quantization_config.runtime.diffusion_step_policy`. The default policy requests A16 activations for the
first/last three denoising iterations and the generator reasoner, and native FP8 in the middle. Overlapping first/last
windows use A16, so a four-step student uses A16 throughout. Weights remain FP8. Standalone vLLM reasoner inference
uses ModelOpt FP8 independently of this diffusion policy. Backend support for the exported policy must be checked with
the chosen runtime image."

## Verify the output

The notebooks' only check is the Inspect cell: "A quick look at the drop-in checkpoint just written: the quantization
method, how many weights were converted to FP8, and the per-tensor scales that accompany them." `inspect_checkpoint`
prints `FP8 checkpoint: <dir>`, `quant_algo   : <algo>  (method=<method>)` (read from the root `hf_quant_config.json`),
`FP8 weights  : <n>` (tensors of dtype `F8_E4M3` in `transformer/*.safetensors`), `scale tensors: <n>` (`.input_scale` /
`.weight_scale`) and up to three `e.g. <key> = <value>` lines. Skill notes (not in the source), tool commands included:

- Expected values: the exporter writes `"quant_algo": "FP8"` and `"quant_method": "modelopt"`
  (`cosmos_framework/quantization/export.py`) into the `quantization_config` of `transformer/config.json` and mirrors
  it, minus `runtime`, to the root `hf_quant_config.json`; both counts must be non-zero. Check without the venv:
  `python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c["quant_algo"], c["quant_method"])' "$OUTPUT_ROOT/edge-fp8/hf_quant_config.json"`
  → `FP8 modelopt`.
- Layout: `ls -la "$OUTPUT_ROOT/edge-fp8" "$OUTPUT_ROOT/edge-fp8/transformer"` must show the README tree above —
  `transformer/` as a real directory with the sharded `diffusion_pytorch_model-*.safetensors`, its index,
  `modelopt_state.pth` and `config.json`; the root `hf_quant_config.json`, `model.safetensors.index.json`,
  `modelopt_state.pth` and `quantization_metadata.json` as files; every other entry a symlink into the source
  checkpoint (relative links; `cosmos_framework/quantization/export.py` `assemble_output_dir`). `du -sh "$OUTPUT_ROOT/edge-fp8/transformer"` is the
  output's own size ("roughly the size of the bf16 source").
- Peak GPU memory: the notebooks print none — sample it beside the run,
  `nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 1 > "$RUN_ARTIFACTS/gpu.log" &` before the block and
  kill the sampler afterwards; record checkpoint, `DEMO`, shape, `NUM_SAMPLES`, wall-clock time, the peak reading and
  the inspect summary in `result_summary.md`. `[calib]` prints prompts x steps, no shape; the shape is `recipe.shape` in
  `quantization_metadata.json` — a wrong prompt count or shape is a configuration error, not a result.
- Where the inspect lines are: the run blocks `tee` everything into `$RUN_ARTIFACTS/<name>.log`, so the inspect summary is
  that log's tail (`grep -A 6 'FP8 checkpoint:' "$RUN_ARTIFACTS/quantize_cosmos3_edge.log"`); copy those lines into the file
  a task asks for (for example `outputs/edge-fp8-inspect.txt`) rather than re-running the section.

## Serving the FP8 output

README § Inference: "The FP8 output is a **drop-in checkpoint** — the same layout as the bf16 source, plus an
`hf_quant_config.json` and a `quantization_config` in `transformer/config.json`. A compatible serving backend detects
the ModelOpt FP8 format from checkpoint metadata. Serve it from a **local directory** — either the output produced
above, or a published `nvidia/Cosmos3-Edge-FP8`, `nvidia/Cosmos3-Nano-FP8`, or `nvidia/Cosmos3-Super-FP8` checkpoint
downloaded with `snapshot_download` (pass `--served-model-name` so clients see the usual model id)" — its two
examples, quoted as text: `vllm serve /path/to/OUTPUT_ROOT/nano-fp8 --served-model-name nvidia/Cosmos3-Nano ...` (A, the
local output) and `snapshot_download('nvidia/Cosmos3-Nano-FP8', local_dir='cosmos3-nano-fp8')` then
`vllm serve cosmos3-nano-fp8 --served-model-name nvidia/Cosmos3-Nano ...` (B, the published repo). "Passing the FP8
checkpoint by `--revision` alone is **not** honored by the vLLM-Omni Cosmos3 pipeline (it loads the default revision),
so the FP8 weights must be materialized to a local directory first." "For the four specialized Super checkpoints, use
the locally quantized outputs; this guide does not assume published `fp8` revisions. The audiovisual notebook maps all
seven output directories to their original served model IDs. Generated outputs contain symlinks: mount the source
checkpoint (and any backing HF cache) at its original absolute path inside Docker as shown in the serving notebooks."

"**Runnable inference examples live in the other Cosmos3 cookbooks, not here.**" — today they are the inference skills:

- `../cosmos3-generate-media/SKILL.md` Variant 3 → [its vLLM-Omni reference § Quantized Checkpoints](../cosmos3-generate-media/references/vllm-omni.md#quantized-checkpoints):
  the `FP8_MODEL_DIR` / `FP8_SOURCE_DIR` mounts for a generated output, the table mapping `edge-fp8`, `nano-fp8`,
  `super-fp8`, `super-t2i-fp8`, `super-t2i-distilled-fp8`, `super-i2v-fp8`, `super-i2v-distilled-fp8` to their
  `SERVED_MODEL`, and the shared `vllm serve "$FP8_MODEL_DIR" --served-model-name "$SERVED_MODEL"` launch. Variant 1
  → [its diffusers reference § Quantized Checkpoints (FP8)](../cosmos3-generate-media/references/diffusers.md#quantized-checkpoints-fp8):
  `COSMOS3_DIFFUSERS_PRECISION=fp8`; "Optional: point `COSMOS3_FP8_MODEL_PATH` at a local Diffusers checkpoint
  directory." Variant 4 → [its SGLang reference § FP8](../cosmos3-generate-media/references/sglang.md#fp8): "To
  serve a checkpoint you quantized yourself, pass its local directory to `--model-path`. If the server runs in
  Docker, mount that directory and use its path inside the container."
- The Reasoner, Action and Transfer cookbooks' vLLM / vLLM-Omni notebooks keep their "Quantized Checkpoints"
  sections; the README reference's Inference table links them. Their skills (Visual Reasoning, Action, Transfer)
  follow in later rows and take over those links.

## Reference files

The full cookbook content lives in `references/`, migrated 1:1 — read the matching file before the first command (see
the top). Do not look for the notebooks; these files replace them.

| File                                       | Content                                                                                                                                                   |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `references/quantization-readme.md`        | The cookbook README: single-call API, Mixed activation precision, Setup (once), the three Run commands, Demo vs. full runs, What gets produced, Inference |
| `references/quantize-cosmos3.md`           | Variant 1 — the former `quantize_cosmos3.ipynb`: Edge, Nano and Super video models (`t2v`)                                                                |
| `references/quantize-super-text2image.md`  | Variant 2 — the former `quantize_super_text2image.ipynb`: Super Text-to-Image base and 4-step distilled (`t2i`)                                           |
| `references/quantize-super-image2video.md` | Variant 3 — the former `quantize_super_image2video.ipynb`: Super Image-to-Video base and 4-step distilled (`i2v`), conditioning images                    |
| `../cookbooks/cosmos3/README.md`           | Shared environment setup guide (Cosmos Framework section, system packages, vLLM / vLLM-Omni server install); not part of this skill                       |

Origin: the retired cookbooks/cosmos3/quantization notebooks and README of the public cosmos-framework repository
(part 2 of the COOKBOOK "Quantized (FP8)" row); superseded by this skill.

## Things not obvious from the docs

Skill notes (not in the source) unless a sentence is quoted:

- **The notebooks are gone, the cells are not.** The three reference files carry every cell; the run blocks assemble
  them into one script per section. Their `.ipynb`-specific steps — the § 4 kernel switch, the duplicated "restore
  cell", the README's `nbconvert` commands — have no equivalent here. The configure cell is the only cell that reads the
  cwd: run the script from the framework checkout root.
- **`OUTPUT_ROOT` and a staging directory.** The export stages the quantized transformer in
  `<OUTPUT_ROOT>/.quantized_transformer_<profile>.tmp` and moves it into `<name>-fp8/transformer/`
  (`cosmos_framework/quantization/__init__.py`, `cosmos_framework/quantization/export.py`), so `OUTPUT_ROOT` needs room
  for the output plus that staging copy during the export; it must not be the notebook default inside the checkout (item 4).
- **The output is not self-contained.** "vae/ scheduler/ ... symlinked back to the source checkpoint" (README) are
  relative symlinks into the Hugging Face cache snapshot (or the `C3_*_DIR` source): keep `HF_HOME` where it was, and
  mount the source at its original absolute path when serving from Docker (README § Inference).
- **Calibration data are downloaded at run time.** Prompts: `WenhaoWang/VideoUFO` via `datasets` streaming
  (`cosmos_framework/quantization/calibration.py`, `prepare_calibration_prompts`; the `datasets` package is in the Setup
  block for this). Variant 3 images: `I2V_COND_DIR` or `nemotron_vlm_dataset_v2` through ModelOpt's
  `get_vlm_dataset_dataloader` — "which requires Hugging Face access to that dataset" (notebook).
- **One process per section.** `DEMO`, `NUM_SAMPLES`, `OUTPUT_ROOT`, `inspect_checkpoint` and the imports are defined by
  cells 0, 2 and 3; a Quantize cell run in a fresh interpreter fails with `NameError`, which is a tooling error, not a
  recipe result. Running several sections in one process is what the notebook does (the Super and distilled cells
  release the previous model first).
- **Unverified hardware envelope.** No run of this skill has measured peak memory or wall-clock time for any section;
  the only figures are the sources' parameter counts and the generate-media skill's measured 122.4 GiB bf16 size of the
  4-step students. Record what you measure; do not infer a supported GPU range from the table.
- **ModelOpt pin.** `nvidia-modelopt[torch]==0.44.0` is the version the Setup block and § 3 install; the diffusers
  loader of the generate-media skill restores `modelopt_state.pth` with "ModelOpt 0.44" too (its diffusers.md § Quantized
  Checkpoints (FP8)).

## Related skills

| Skill                                  | When to use                                                                                                          |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `../cosmos3-setup/SKILL.md`            | Framework checkout; the pinned uv is installed by the first line of Setup (once)                                     |
| `../cosmos3-env-troubleshoot/SKILL.md` | CUDA / import / library failures during `uv sync` or the GPU check                                                   |
| `../cosmos3-generate-media/SKILL.md`   | Serve the FP8 output or a published FP8 checkpoint (Variant 1 diffusers, Variant 3 vLLM-Omni, Variant 4 SGLang)      |
| `../cosmos3-inference/SKILL.md`        | Framework-native inference, including the load-time `--quantization-method` flags that are distinct from this export |
| `../cosmos3-skill-authoring/SKILL.md`  | Editing this skill's eval cases                                                                                      |
