---
name: cosmos3-inference-transfer
description: >
  Run Cosmos3 video transfer (control-to-video): a prompt.json caption plus one or more
  spatial control videos (edge/Canny, blur, depth, segmentation, world-scenario map) in, a
  new clip out, with Cosmos3-Nano (single GPU) or Cosmos3-Super (multi-GPU) through the five
  documented paths: cosmos_framework python/torchrun, the diffusers Cosmos3OmniModularPipeline,
  and the vLLM-Omni, SGLang and TensorRT-LLM servers. Use when the user asks "run video
  transfer / control-to-video with Cosmos3", "generate a video from an edge, blur, depth,
  segmentation or WSM control video", "run the transfer specs with the framework", "serve
  Cosmos3 transfer with vLLM-Omni / SGLang / TensorRT-LLM", or which transfer backend fits
  their GPU. Not for the Transfer-Auto and Sim-Transfer world-simulation models, for
  text/image-to-video without a control video (cosmos3-generate-media), for Reasoner inference,
  or for quantizing a checkpoint yourself (cosmos3-inference-quantization). Ships the edge, blur
  and depth control videos, every prompt, the negative prompt and the framework specs in assets/
  and specs/; the seg and wsm control videos are fetched by a documented command.
---

# Cosmos3 Video Transfer (control-to-video)

One skill for the Cosmos3 transfer cookbook: "Cosmos3 video **transfer** examples — **Nano** (single GPU) and **Super** (multi-GPU, 32B) — on the
native PyTorch (Cosmos Framework) path, the Diffusers modular pipeline, and the OpenAI-compatible TensorRT-LLM and vLLM-Omni server paths"
(cookbook README), plus the SGLang notebook. "Video transfer generates a target clip from a `prompt.json` caption and one or more spatial control
signals." Five variants, one per cookbook notebook, sharing the same specs, control videos and prompts.

## Read this first

Before the first command of any variant — including a single edge smoke:

1. Read the variant's reference file in full (linked under its heading, with the sections the smoke path needs) and the matching backend section
   of [`references/transfer-readme.md`](references/transfer-readme.md) (the cookbook README: "Transfer Definition", the backend's "Quickstart",
   "Spec field reference", "Troubleshooting"). The variant sections below only summarise the sources; the reference files are the notebooks
   migrated 1:1 and the README copied verbatim. Do not look for the original notebooks or the cookbook README: the reference files replace them.
2. Check the GPU with `nvidia-smi`. The sources size the models as "Nano (single GPU)" and "Super (multi-GPU, 32B)" (table below); the driver's
   CUDA major version picks the `cu130` / `cu128` wheel tag ("CUDA driver and the `cuXXX` backend" in the setup guide).
3. Environment setup is the shared setup guide [`../cookbooks/cosmos3/README.md`](../cookbooks/cosmos3/README.md) — every README quickstart opens
   with "Set up the environment: … setup", pointing at its "Cosmos Framework", "Diffusers", "TensorRT-LLM Generator" or "vLLM-Omni" section; the
   framework notebook is self-contained instead (§ 4–§ 6 install the system packages, clone the framework and run `uv sync`).
4. Where the inputs live (skill note): `specs/` and `assets/` of this skill directory (tables below). The notebooks resolve everything from
   `find_repo_root()` → `.agents/skills/cookbooks/cosmos3/generator/transfer`; in this skill that directory is the skill directory itself. Run
   everything from here; the replacement for the notebooks' path cells is in "Paths" below.
5. Guardrails, as the sources state them: "Blur, segmentation, and WSM use the Guardrail and require access to the gated
   [nvidia/Cosmos-1.0-Guardrail](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail) HF repository. The edge and depth calls explicitly disable
   the safety checker." (diffusers notebook; the vLLM-Omni notebook's edge and depth requests do the same). Server-wide switches named by the
   sources: `--no-guardrails` for `cosmos_framework.scripts.inference` and for `vllm serve`, `SGLANG_DISABLE_COSMOS3_GUARDRAILS=1` for SGLang,
   `TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1` or `use_guardrails: false` in `extra_params` for TensorRT-LLM (setup guide "Prerequisites").
6. After a failure, go back to the reference section that covers the failing step before improvising a fix.
7. Verify what you produced before reporting it ("Verify the output" below) and record the facts in `result_summary.md`.

## When to use this skill

- Use to generate a video from a caption and a precomputed control video — edge (Canny), blur, depth, segmentation or world scenario map (WSM) —
  with Cosmos3-Nano or Cosmos3-Super, in-process (Cosmos Framework, Diffusers) or against a server (vLLM-Omni, SGLang, TensorRT-LLM)
- Use for multi-control transfer: "Cosmos Framework, TensorRT-LLM, and vLLM-Omni support multi-control transfer. Per-hint weighting is supported
  only by Cosmos Framework; the serving backends accept multiple unweighted controls. Diffusers also accepts multiple precomputed controls
  without per-hint weights." (README); the framework example derives edge and blur from a raw source video (`specs/multi_control.json`)
- Use to pick the transfer backend and model size for the GPUs at hand
- Not for Transfer-Auto or Sim-Transfer: "**Transfer-Auto** is driven like a transfer model (control video in, photoreal video out) but belongs
  to the Simulation family" (simulation cookbook README) — use `../cosmos3-nano-transfer-auto/inference/SKILL.md` and
  `../cosmos3-nano-sim-transfer/inference/SKILL.md`
- Not for text-, image- or video-to-video generation without a control video (`../cosmos3-generate-media/SKILL.md`), not for Reasoner
  inference, not for making an FP8 checkpoint (see "FP8" in the variant blocks)
- For environment problems, hand off to `../cosmos3-env-troubleshoot/SKILL.md`

## Bundled assets

"Sample assets under `assets/` cover spatial control signals paired with `prompt.json` files" and "All examples share `assets/negative_prompt.json`
for the negative caption" (README). The README's table, with what ships in this skill (paths relative to this skill directory):

| Control (README)     | Asset folder                  | Inference input                                    | Generation duration (README) | Framework spec             | In this skill                                                                                                                                                                                |
| -------------------- | ----------------------------- | -------------------------------------------------- | ---------------------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Edge (Canny)         | `assets/edge/`                | `control_edge.mp4` + `prompt.json`                 | 121 frames @ 30 FPS          | `specs/edge.json`          | both vendored (`control_edge.mp4` 0.66 MiB)                                                                                                                                                  |
| Blur                 | `assets/blur/`                | `control_blur.mp4` + `prompt.json`                 | 121 frames @ 30 FPS          | `specs/blur.json`          | both vendored (`control_blur.mp4` 0.39 MiB)                                                                                                                                                  |
| Depth                | `assets/depth/`               | `control_depth.mp4` + `prompt.json`                | 121 frames @ 30 FPS          | `specs/depth.json`         | both vendored (`control_depth.mp4` 1.00 MiB)                                                                                                                                                 |
| Segmentation         | `assets/seg/`                 | `control_seg.mp4` + `prompt.json`                  | 121 frames @ 30 FPS          | `specs/seg.json`           | `prompt.json` vendored; `control_seg.mp4` (4.77 MiB) NOT vendored — materialise it (below)                                                                                                   |
| World scenario (WSM) | `assets/wsm/`                 | `control_wsm.mp4` + `prompt.json`                  | 100 frames @ 10 FPS          | `specs/wsm.json`           | `prompt.json` vendored; `control_wsm.mp4` (3.44 MiB) NOT vendored — materialise it (below)                                                                                                   |
| Multi-control        | `assets/multi_control/`       | `vision_path` + multiple hints (Framework example) | 93 frames @ 16 FPS           | `specs/multi_control.json` | `prompt.json` vendored (the folder's only file); the source video is the spec's `vision_path` URL — § 19: "computed on-the-fly via `vision_path`. No pre-computed control files are needed." |
| Negative caption     | `assets/negative_prompt.json` | every spec's `negative_prompt_file`                | —                            | —                          | vendored                                                                                                                                                                                     |

Skill note (not in the source): `specs/*.json` are byte-identical copies of the cookbook's `specs/` (which stays in the cookbook for the framework
tests), kept next to `assets/` so the specs' relative `control_path`, `prompt_path` and `negative_prompt_file` values (one directory up, into
`assets/`) resolve exactly as in the cookbook — the framework validates a spec inside the spec's own folder, and the notebooks' `resolve_spec_path`
resolves against the spec's parent. The vendored control videos are plain blobs (nested `.gitattributes`), identical to the cookbook's LFS objects
(same sha256). The two videos over the vendoring limit stay in the cookbook's `assets/` as LFS objects; fetch them into this skill's `assets/` before
a `seg` or `wsm` run (the LFS pointers give sizes 5,000,598 and 3,611,984 bytes and the sha256 `oid`s `cb3faf8e…29b64b` and `614a207c…60b6ae`):

```bash
# Skill note (not in the source): materialise the two non-vendored control videos from the public repository's LFS media (verified 2026-11-07: HTTP 200, sizes and sha256 match the pointers)
for c in seg wsm; do
  curl -L --fail -o "assets/$c/control_$c.mp4" \
    "https://media.githubusercontent.com/media/NVIDIA/cosmos-framework/bbd4a67c3cf54cc6699464ad1259f3a0f6537da9/.agents/skills/cookbooks/cosmos3/generator/transfer/assets/$c/control_$c.mp4"
done
ls -l assets/*/control_*.mp4   # five videos. Alternative: `git lfs pull --include=".agents/skills/cookbooks/cosmos3/generator/transfer/assets/**"` in a checkout with git-lfs, then copy the two files here
```

`scripts/preview_helpers.py` is the cookbook's `preview_helpers.py`, unchanged. Skill note (not in the source): the notebooks import
`preview_helpers` from their own directory (`sys.path.insert(0, str(COSMOS3_TRANSFER_ROOT))`, or the `_root` search for the cookbook transfer
directory in the framework notebook's preview cells); here put `scripts/` on `sys.path`, as the "Paths" block does, or copy the file next to the
code you run — do not edit it. Its `preview_transfer` / `preview_multi_control` need ffmpeg (`imageio_ffmpeg` or `ffmpeg` on PATH) and read
`COSMOS3_TRANSFER_ROOT` / `COSMOS3_TRANSFER_OUTPUT_ROOT`.

## Model sizes vs GPU (as the sources state them)

| Checkpoint                                            | Size, as the sources state it                                                                                                                                                                                   | GPUs, as the sources run it                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `nvidia/Cosmos3-Nano`                                 | "Nano (single GPU)" (README)                                                                                                                                                                                    | Framework: `.venv/bin/python` launcher, `--parallelism-preset=latency`, 1 GPU (README table); Diffusers: one pipeline on `cuda`; TensorRT-LLM: `trtllm-serve nvidia/Cosmos3-Nano --visual_gen_args …/cosmos3-nano-1gpu.yaml` ("Nano: one GPU"); vLLM-Omni and SGLang: the single-GPU Docker launches of their notebooks                                                                                                                         |
| `nvidia/Cosmos3-Super`                                | "Super (multi-GPU, 32B)" (README)                                                                                                                                                                               | Framework: `.venv/bin/torchrun --nproc-per-node=<N>`, "4+" GPUs; TensorRT-LLM: `torchrun --nproc_per_node=4 -m tensorrt_llm.commands.serve nvidia/Cosmos3-Super --visual_gen_args …/cosmos3-super-4gpu.yaml` ("Super: four GPUs"); vLLM-Omni: "For Super, add the upstream `--tensor-parallel-size 4` and `--enable-layerwise-offload` options if needed"; Diffusers: `Cosmos3OmniModularPipeline.from_pretrained("nvidia/Cosmos3-Super", ...)` |
| `nvidia/Cosmos3-Nano-FP8`, `nvidia/Cosmos3-Super-FP8` | "FP8 checkpoints are published for Edge, Nano, and Super. This video-to-video transfer example requires `Cosmos3-Nano`; base `Cosmos3-Edge-FP8` does not support video-to-video transfer." (vLLM-Omni notebook) | the FP8 routes of Variants 2–4 below                                                                                                                                                                                                                                                                                                                                                                                                            |

The sources give no memory figures for transfer and none are invented here (unknown until measured). All checkpoints are gated on Hugging Face:
"model access on Hugging Face, and either `uvx hf@latest auth login` or `HF_TOKEN` set" (diffusers notebook); the framework notebook takes
`HF_TOKEN` in its § 2 cell.

## Paths: running the notebook code from this skill directory

Skill note (not in the notebooks): the vLLM-Omni, SGLang and TensorRT-LLM notebooks start with `COSMOS_ROOT = find_repo_root(...)` (vLLM-Omni:
`FRAMEWORK_ROOT = find_framework_root(...)`) and derive `TRANSFER_ROOT = COSMOS_ROOT / "cookbooks" / "cosmos3" / "generator" / "transfer"`,
`SPECS_DIR` and `ASSETS_DIR`; the diffusers notebook does the same inside `configure_diffusers_environment()`. Replace those assignments — and
nothing else in the cells — with:

```python
# Skill note (not in the notebooks): run from this skill directory; every other line of the notebooks' cells runs unchanged
import os, sys
from pathlib import Path

COSMOS_ROOT = Path.cwd().resolve()       # this skill directory; the helpers' relative_to(COSMOS_ROOT) prints and vLLM-Omni's repo_relative_path() resolve against it
FRAMEWORK_ROOT = COSMOS_ROOT             # vLLM-Omni notebook only (its find_framework_root result is not used elsewhere)
TRANSFER_ROOT = COSMOS_ROOT              # the notebooks' cookbooks/cosmos3/generator/transfer
COSMOS3_TRANSFER_ROOT = TRANSFER_ROOT    # the diffusers notebook's name for it
SPECS_DIR = TRANSFER_ROOT / "specs"
ASSETS_DIR = TRANSFER_ROOT / "assets"
os.environ["COSMOS3_TRANSFER_ROOT"] = str(TRANSFER_ROOT)   # preview_helpers resolves specs/ and outputs from it
sys.path.insert(0, str(TRANSFER_ROOT / "scripts"))         # `from preview_helpers import ...`
```

The notebooks' output roots then land under this directory's `outputs/` unchanged: `outputs/vllm_omni/<control>/transfer_<control>.mp4` (vLLM-Omni),
`outputs/sglang/<control>/transfer_<control>.mp4` (SGLang), `outputs/notebooks/trt_llm/transfer_<control>.mp4` (TensorRT-LLM),
`outputs/notebooks/diffusers/<model>/transfer_<control>/vision.mp4` (Diffusers), `outputs/notebooks/<model>/transfer_<control>/vision.mp4` (framework
notebook). The framework notebook's `%%bash` cells read the environment its § 3 cell exports; outside Jupyter export it in every shell call (Variant 1).

## Variant 1 — Cosmos Framework (`cosmos_framework.scripts.inference`, in-repo)

Reference: [`references/cosmos-framework.md`](references/cosmos-framework.md) — read it before the first command (§ 1 Prerequisites, § 2 Configure,
§ 3 Confirm GPU access, § 4 Install system packages, § 5 Clone Cosmos Framework, § 6 Install, § 6.5 Authenticate with Hugging Face, § 7 Verify,
§ 8 Input Specs, § 9 Nano: Edge, § 19 Multi-Control), and `references/transfer-readme.md` "Run with Cosmos Framework" (Quickstart, launcher table,
output layout).

The README quickstart ("Run the commands below inside the **cosmos container** (e.g. `pytorch:25.09-py3`) — the same environment used to install the
venv and run the notebook. The commands mirror the notebook exactly: `cd` into the framework repo first, then invoke the venv's Python or torchrun
(the system Python does not have `cosmos_framework`)"):

```bash
# Set once — the cosmos-framework repo root (contains .venv/ and pyproject.toml).
# In this cosmos checkout: packages/cosmos3 (or packages/cosmos-framework).
export COSMOS_FRAMEWORK=/path/to/cosmos-framework   # e.g. <cosmos_root>/packages/cosmos3
export TRANSFER_ROOT=$(pwd)/cookbooks/cosmos3/generator/transfer

# NGC containers bundle libtorch in LD_LIBRARY_PATH which conflicts with Triton/CUDA.
unset LD_LIBRARY_PATH
```

Skill note (not in the source): from this skill directory, `export TRANSFER_ROOT="$PWD"` replaces the `$(pwd)/cookbooks/cosmos3/generator/transfer`
line — the specs and assets are here. Then the README's Nano block, verbatim:

```bash
cd "$COSMOS_FRAMEWORK"

# edge — replace edge.json with blur.json / depth.json / seg.json / wsm.json for other controls
CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$TRANSFER_ROOT/specs/edge.json" \
  -o "$TRANSFER_ROOT/outputs/Cosmos3-Nano/" \
  --checkpoint-path Cosmos3-Nano \
  --seed 2026
```

"Outputs are written under the directory passed to `-o`, with one subdirectory per sample name, e.g. `outputs/Cosmos3-Nano/transfer_edge/vision.mp4`."
Super: the README's `torchrun --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29500 … --checkpoint-path Cosmos3-Super` block
(`CUDA_VISIBLE_DEVICES=0,1,2,3`, "4+" GPUs). "The same spec files are used for both Nano and Super — model selection is controlled entirely by
`--checkpoint-path`." Guardrails: "Framework inference enables **guardrails by default**" and needs the system packages plus Hugging Face access to the
gated safety models; "To disable guardrails for a one-off run, pass `--no-guardrails` to `cosmos_framework.scripts.inference`" (setup guide) — skill
note: for a smoke run without access to the gated guardrail repository, append `--no-guardrails` to the command above.

The framework environment (`.venv` with `cosmos_framework`): the setup guide's "Cosmos Framework" section (`GIT_LFS_SKIP_SMUDGE=1`,
`uv sync --all-extras --group=cu130-train` or `cu128-train`, the system packages) with `uv 0.11.29` (the notebook's § 6: "Installs the
Framework-required `uv 0.11.29`"; the diffusers notebook's § 1 says the same), or the notebook's self-contained § 4–§ 6: run each `%%bash` cell of
the reference as ONE shell call, verbatim. Skill note (not in the notebook): outside Jupyter the cells do not share a process, so export in every
shell call what the notebook's § 3 cell puts into `os.environ`:

```bash
# Skill note (not in the notebook): the § 3 cell's environment for the %%bash cells of references/cosmos-framework.md
export COSMOS3_TRANSFER_ROOT="$PWD"                                                   # this skill directory (specs/, assets/)
export COSMOS3_TRANSFER_OUTPUT_ROOT="${COSMOS3_TRANSFER_OUTPUT_ROOT:-$PWD/outputs/notebooks}"
export COSMOS3_REPO="${COSMOS3_REPO:?set COSMOS3_REPO to the framework checkout (an existing tree skips the clone of § 5; a new path is cloned there)}"
export COSMOS3_GIT_URL="${COSMOS3_GIT_URL:-https://github.com/NVIDIA/cosmos-framework.git}"; export HF_TOKEN="${HF_TOKEN:-}"   # HF_TOKEN: the § 2 cell (empty when `hf auth login` is used)
export COSMOS3_UV_GROUP="${COSMOS3_UV_GROUP:-cu128-train}"   # the notebook's x86_64 default (aarch64: cu130-train); the setup guide pairs the tag with the driver's CUDA version
export UV_CACHE_DIR="${UV_CACHE_DIR:-$COSMOS3_TRANSFER_ROOT/.cache/uv}"; export HF_HOME="${HF_HOME:-$COSMOS3_TRANSFER_ROOT/.cache/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"; export COSMOS3_NUM_GPUS="${COSMOS3_NUM_GPUS:-1}"   # the Super cells use the GPU list and count
export COSMOS3_MASTER_ADDR="${COSMOS3_MASTER_ADDR:-127.0.0.1}"; export COSMOS3_MASTER_PORT="${COSMOS3_MASTER_PORT:-29500}"   # 29500 = the README's Super block (--master-port=29500); the notebook's § 3 cell picks a free port (free_local_port()) instead
unset LD_LIBRARY_PATH
```

The notebook's § 9 cell is the README command with `-i "$COSMOS3_TRANSFER_ROOT/specs/edge.json"`, `-o "$COSMOS3_TRANSFER_OUTPUT_ROOT/Cosmos3-Nano"`
and `cd "$COSMOS3_REPO"`; its output is `<COSMOS3_TRANSFER_OUTPUT_ROOT>/Cosmos3-Nano/transfer_edge/vision.mp4`. § 19 (Nano, `specs/multi_control.json`):
"Combines **edge** (Canny) and **blur** controls simultaneously from a single source video, computed on-the-fly via `vision_path`. No pre-computed
control files are needed." "The framework also saves the intermediate control signals alongside the output: `control_edge.mp4` and `control_blur.mp4`."

FP8: the framework notebook and README have no FP8 route of their own; the README's pointer is "For Nano/Super static FP8 checkpoints, see the
[quantization cookbook](../cookbooks/cosmos3/quantization/README.md) and the **Quantized Checkpoints** section in the vLLM-Omni notebooks" (Variant 3).
To quantize your own checkpoint: `../cosmos3-inference-quantization/SKILL.md` (the quantization cookbook's own migration; the path resolves once that row merges).

## Variant 2 — Diffusers (`Cosmos3OmniModularPipeline`, in-process)

Reference: [`references/diffusers.md`](references/diffusers.md) — read it before the first command (§ 1 Prerequisites, § 2 Configure Paths and
Environment, § 3 Install Diffusers Dependencies, § 5 Verify, § 7 Define the Transfer Runner — `get_pipe`, `run_transfer`,
`GUARDRAIL_DISABLED_CONTROLS` — "Quantized Checkpoints (FP8)", "Nano: Edge (Canny) Transfer"), and `references/transfer-readme.md` "Run with
Diffusers": its Quickstart is the standalone depth example (`Cosmos3OmniModularPipeline.from_pretrained("nvidia/Cosmos3-Nano", torch_dtype=torch.bfloat16)`,
`pipe.load_components(torch_dtype=torch.bfloat16)`, `pipe.enable_safety_checker()`, `pipe(prompt=…, negative_prompt=…, control_videos={"depth": load_video(…)},
control_guidance=1.5, num_frames=121, num_video_frames_per_chunk=121, num_conditional_frames=1, height=720, width=1280, fps=30.0, num_inference_steps=50,
guidance_scale=3.0, generator=torch.Generator(device="cuda").manual_seed(2026), output="videos")`, `export_to_video(videos, "/tmp/cosmos3_transfer_depth.mp4", fps=30, macro_block_size=1)`)
— run it verbatim from the reference if you want the README form; its `transfer_root` is the cookbook directory (skill note: here `Path(".")`).
"Transfer is reachable only through the modular pipeline, which takes precomputed control videos as `control_videos={hint: frames}`" (README).

The notebook's venv — the § 2 exports, verbatim ("use the CUDA 13 or 12.8 Torch backend depending on the CUDA version installed on your system
(`cu130` or `cu128`)"), then its § 3 install cell. § 1: "**uv version:** use `uv 0.11.29`, matching the version pinned by Cosmos Framework"; "**Headless
servers:** if you see an error like `libxcb.so.1: cannot open shared object file` … install the required system libraries:
`apt-get install -y libxcb1 libgl1 libglib2.0-0`":

```bash
export COSMOS3_DIFFUSERS_VENV=/path/to/.venv-cosmos3-diffusers
export COSMOS3_TORCH_BACKEND=cu130
export HF_HOME=/path/to/large/huggingface/cache
export UV_LINK_MODE=copy
export CUDA_VISIBLE_DEVICES=0
export COSMOS3_DIFFUSERS_PRECISION=bf16  # or fp8
```

Then the § 3 install cell — run it from `references/diffusers.md` as ONE shell call, verbatim, together with the exports above (it creates the
venv with `uv venv "$COSMOS3_DIFFUSERS_VENV" --python 3.13 --seed --managed-python --allow-existing`, activates it and installs
`diffusers @ git+https://github.com/huggingface/diffusers.git`, `accelerate`, `av`, `cosmos_guardrail`, `huggingface_hub`, `imageio`,
`imageio-ffmpeg`, `ipykernel`, `torch`, `torchvision`, `transformers` and `nvidia-modelopt==0.44.0` with `--torch-backend="$COSMOS3_TORCH_BACKEND"`,
then registers the Jupyter kernel); do not compose an install command from this summary.

Skill note (not in the notebook): the `export`s and the `source` do not survive a new shell call, hence ONE call; the kernel registration and the
"switch kernel" cells (§ 4) are Jupyter plumbing — outside Jupyter run the later cells with `"$COSMOS3_DIFFUSERS_VENV/bin/python"`.

Run: the notebook's § 2 `configure_diffusers_environment()` cell with, inside the function, `COSMOS_ROOT = Path.cwd().resolve()` and
`COSMOS3_TRANSFER_ROOT = COSMOS_ROOT` in place of its first two assignments (skill note: the path translation of "Paths"; everything else in the
function — venv, backend, the `…/outputs/notebooks/diffusers` output root, `HF_HOME`, `HF_HUB_DISABLE_XET`, the `COSMOS3_DIFFUSERS_PRECISION` default —
unchanged), then § 7 verbatim with `sys.path.insert(0, str(COSMOS3_TRANSFER_ROOT / "scripts"))` before its `from preview_helpers import …` line (skill
note), then the use-case cell `edge_nano_output = run_transfer("edge", model="Cosmos3-Nano")` — as ONE Python file run with
`"$COSMOS3_DIFFUSERS_VENV/bin/python"` from this directory. `run_transfer` "reads a spec, hands the control video to `Cosmos3OmniModularPipeline` as
`control_videos={hint: frames}`, and writes `vision.mp4` under `outputs/notebooks/diffusers/<model>/<spec name>/`"; "`guidance_scale` is the usual
text CFG and `control_guidance` amplifies the control signal on top of it. Both come from the spec, as do the step count, flow shift, seed, frame
count, and FPS." Guardrails: `GUARDRAIL_DISABLED_CONTROLS = frozenset({"edge", "depth"})` — `run_transfer` calls `pipe.disable_safety_checker()` for
those two controls and `pipe.enable_safety_checker()` for blur, seg and wsm (gated repository, "Read this first" 5). Output:
`outputs/notebooks/diffusers/Cosmos3-Nano/transfer_edge/vision.mp4`. "To run **Cosmos3-Super** instead, load the larger checkpoint:
`Cosmos3OmniModularPipeline.from_pretrained("nvidia/Cosmos3-Super", ...)`."

FP8 (the notebook's route): "FP8 checkpoints: set `COSMOS3_DIFFUSERS_PRECISION=fp8` before loading. Nano and Super use the dedicated Hub repos
`nvidia/Cosmos3-Nano-FP8` and `nvidia/Cosmos3-Super-FP8`. Use-case cells stay the same." "Default loads are BF16 (`COSMOS3_DIFFUSERS_PRECISION=bf16`).
Set the env var to `fp8` before running the helper cell's `get_pipe` (or re-run that cell after changing it)." "Do not pass `revision=fp8` on the BF16
cards." "Optional: point `COSMOS3_FP8_MODEL_PATH` at a local Diffusers checkpoint directory. Transfer preloads `transformer/` so ModelOpt 0.44 restores
`modelopt_state.pth`." The loader (`resolve_pretrained_source`, `materialize_checkpoint_dir`, `validate_fp8_transformer`, the ModelOpt imports in
`get_pipe`) is § 7 of the reference. To quantize your own checkpoint: `../cosmos3-inference-quantization/SKILL.md` (see Variant 1's note).

## Variant 3 — vLLM-Omni (Docker server, multipart `POST /v1/videos/sync`)

Reference: [`references/vllm-omni.md`](references/vllm-omni.md) — read it before the first command (§ 1 Prerequisites — server launch and
guardrails — "Quantized Checkpoints", § 2 Configure Paths and Endpoints, § 3 Verify Endpoint Configuration, § 4 Define Transfer Request Helpers,
§ 6 Edge (Canny) Transfer), and `references/transfer-readme.md` "Run with vLLM-Omni".

The notebook's server launch, verbatim ("Start a vLLM-Omni Cosmos3 server before running the request cells. The examples assume `.agents/skills` is
mounted at `/workspace` and the server runs from that directory inside the container, so repo-local `cookbooks/...` paths resolve correctly."
"Transfer controls are available from vLLM-Omni `main` and the released `vllm/vllm-omni:cosmos3` container."):

```bash
cd /path/to/cosmos-framework/.agents/skills
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export COSMOS3_WORKDIR="$PWD"
export COSMOS3_HOST_PORT=8000

docker run --runtime nvidia --gpus all \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${COSMOS3_WORKDIR}:/workspace" \
  -v "${COSMOS3_WORKDIR}:${COSMOS3_WORKDIR}" \
  -e HF_TOKEN \
  -p "${COSMOS3_HOST_PORT}:8000" --ipc=host \
  -w /workspace \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Nano \
    --omni \
    --model-class-name Cosmos3OmniDiffusersPipeline \
    --allowed-local-media-path / \
    --port 8000 \
    --init-timeout 1800
```

Skill note (not in the notebook): run it from this skill directory instead of `.agents/skills` (`cd` here, so `COSMOS3_WORKDIR="$PWD"` mounts this
directory at `/workspace`): the helper's `repo_relative_path()` sends `control_path` relative to `COSMOS_ROOT` — with the "Paths" block that is
`assets/edge/control_edge.mp4`, which the server, running in `/workspace`, reads from this mount. Guardrails (notebook): "Generator guardrails are on
by default and require access to the gated `nvidia/Cosmos-1.0-Guardrail` repository. When guardrails are enabled, also set and pass `HF_TOKEN` to the
container; it must be a Hugging Face token authorized for that repository. The edge and depth requests explicitly disable guardrails; blur,
segmentation, and WSM use `COSMOS3_VLLM_GUARDRAILS` (default `true`). To disable guardrails server-wide, pass `--no-guardrails` to `vllm serve`;
alternatively, set `COSMOS3_VLLM_GUARDRAILS=false` for these sample requests." Setup guide: without access to the gated repository "the server exits
before serving requests. Per-request `guardrails: false` in `extra_params` … does not fix this" — skill note: for a smoke run without that access,
add `--no-guardrails` to the `vllm serve` line above. Readiness: "Cosmos3 checkpoints can exceed the default server init timeout — always pass
`--init-timeout 1800`"; "vLLM-Omni prints `Application startup complete.` when the API is ready" (setup guide); the notebook's § 3 checks
`GET http://localhost:8000/v1/models`. Headless automation (skill note): `docker run -d --name cosmos3-vllm …` with the rest unchanged, poll
`/v1/models`, `docker logs`, then `docker stop && docker rm` — the pattern of `../cosmos3-generate-media/references/skill-notes.md` § 4.

Request (§ 4 `build_transfer_request` / `run_transfer`): `requests.post(VIDEOS_SYNC_URL, data=form, headers={"Accept": "video/mp4"}, timeout=3600)` to
`POST /v1/videos/sync` — a multipart form with `prompt` and `negative_prompt` (compact JSON of `assets/edge/prompt.json` and
`assets/negative_prompt.json`), `size` from the spec (`"1280x720"`), `num_frames "121"`, `fps "30"`, `num_inference_steps "50"`, `guidance_scale "3.0"`
(the spec's `guidance`), `flow_shift "10.0"`, `seed "2026"` and `extra_params` = compact JSON of `{"use_resolution_template": false,
"use_duration_template": false, "guardrails": <false for edge and depth>, "edge": {"control_path": "assets/edge/control_edge.mp4",
"preset_edge_threshold": "medium"}, "resolution": "720", "control_guidance": 1.5, "num_video_frames_per_chunk": 121, "num_conditional_frames": 1,
"num_first_chunk_conditional_frames": 0, "share_vision_temporal_positions": true, "max_frames": 121}`; the response body is the MP4, written to
`outputs/vllm_omni/edge/transfer_edge.mp4`. README: "Transfer requests should also pass the spec `resolution` inside `extra_params`. Cosmos3 transfer
bucket selection reads `extra_params.resolution` and the control/input aspect ratio; set the video API `size` field to the matching `<width>x<height>`
value for a consistent request." "vLLM-Omni accepts multiple control hints in one request, but does not support the per-hint `weight` field available
in Cosmos Framework."

FP8 ("Quantized Checkpoints"): "FP8 checkpoints are published for Edge, Nano, and Super. This video-to-video transfer example requires `Cosmos3-Nano`;
base `Cosmos3-Edge-FP8` does not support video-to-video transfer. Use `nvidia/Cosmos3-Nano-FP8` below." Download the published checkpoint to a local
directory (`FP8_MODEL_DIR`, `snapshot_download("nvidia/Cosmos3-Nano-FP8", local_dir=…)`) — "Or use an output from the quantization walkthrough" — and
"Serve that directory with the original `--served-model-name` so the request cells keep their existing model IDs. Do not select FP8 with `--revision`
alone: the Cosmos3 vLLM-Omni pipeline must receive a materialized local checkpoint." The launch is `vllm serve "$FP8_MODEL_DIR" --served-model-name
"$SERVED_MODEL" --omni --model-class-name Cosmos3OmniDiffusersPipeline --allowed-local-media-path / --port 8000 --init-timeout 1800` with `FP8_MODEL_DIR`
(and `FP8_SOURCE_DIR`: "The source mount preserves exported symlinks") mounted read-only at their absolute paths and `HF_HOME` "consistent with the
quantization run" — the full `add_mount` block is in the reference. Skill note: that block runs "from the **Framework checkout root**" and mounts
`$PWD/.agents/skills` at `/workspace`; from this skill directory use `add_mount "$PWD" /workspace` so the control paths under `assets/` resolve as above.
"Poll `curl http://localhost:8000/v1/models` until ready and verify the ID equals `SERVED_MODEL`." To quantize your own checkpoint:
`../cosmos3-inference-quantization/SKILL.md` (see Variant 1's note).

## Variant 4 — SGLang (Docker server, asynchronous `POST /v1/videos`)

Reference: [`references/sglang.md`](references/sglang.md) — read it before the first command (§ 1 Prerequisites — launch, guardrails, FP8 — § 2
Configure Paths and Endpoints, § 3 Verify Endpoint Configuration, § 4 Define Transfer Request Helpers, § 6 Edge (Canny) Transfer). The cookbook
README has no SGLang section; the notebook is the whole source.

The notebook's launch, verbatim ("Start a SGLang Cosmos3 server before running the request cells. The examples assume `${COSMOS3_WORKDIR}` points to
cosmos repo." "Transfer controls are available from SGLang `main` and the released `lmsysorg/sglang:dev` container."):

```bash
export SGLANG_HF_CACHE="${SGLANG_HF_CACHE:-$HOME/.cache/sglang-huggingface}"
mkdir -p "$SGLANG_HF_CACHE"
chmod 777 "$SGLANG_HF_CACHE"

export COSMOS3_WORKDIR="$PWD"

docker run --runtime nvidia --gpus all \
  -v $SGLANG_HF_CACHE:/root/.cache/huggingface \
  -v "${COSMOS3_WORKDIR}:${COSMOS3_WORKDIR}" \
  -p 30000:30000 \
  --ipc=host \
  lmsysorg/sglang:dev \
  sglang serve \
  --model-path nvidia/Cosmos3-Nano \
  --host 0.0.0.0
```

Skill note (not in the notebook): run it from this skill directory — `COSMOS3_WORKDIR="$PWD"` mounts it at the same absolute path inside the
container, and the helper sends the control's absolute host path (`hint["control_path"] = local_control_path.resolve().as_posix()`), which the server
reads through that mount. Guardrails (notebook): "Generator guardrails are on by default and require access to the gated `nvidia/Cosmos-1.0-Guardrail`
repository. When guardrails are enabled, also set and pass `HF_TOKEN` to the container; it must be a Hugging Face token authorized for that
repository. To disable guardrails server-wide, set `SGLANG_DISABLE_COSMOS3_GUARDRAILS=1` when launching server; alternatively, set
`"use_guardrails": false` in these sample requests." — skill note: for a Docker launch that is `-e SGLANG_DISABLE_COSMOS3_GUARDRAILS=1` (and
`-e HF_TOKEN` to pass the token) on the `docker run` above. The notebook's requests send `guardrails` from `COSMOS3_SGLANG_GUARDRAILS` (default
`"false"`). Readiness: the notebook's § 3 checks `GET http://localhost:30000/v1/models` (`requests.get(MODELS_URL, timeout=30)`, `raise_for_status()`).
Headless (skill note): `docker run -d --name cosmos3-sglang …`, poll `/v1/models`, `docker logs`, `docker stop && docker rm` — pattern:
`../cosmos3-generate-media/references/skill-notes.md` § 4.

Request (§ 4): `requests.post(VIDEOS_GENERATION_URL, json=form, headers={"Accept": "video/mp4"}, timeout=3600)` to `POST /v1/videos` with the same
`form` fields as Variant 3 (a JSON body here; `extra_params` additionally carries `"control_hint": "edge"`, and `control_path` is the absolute host
path), then poll `GET /v1/videos/{id}` every 5 s until `status == "completed"` (`failed` / `cancelled` raise) and download `GET /v1/videos/{id}/content`
→ `outputs/sglang/edge/transfer_edge.mp4` (`response.json` and `final.json` are written beside it). `COSMOS3_SGLANG_API_KEY`, when set, is sent as
`Authorization: Bearer`. "SGLang also accepts unweighted multi-control requests by passing a list of control videos in the API request. Per-hint
`weight` is supported only by Cosmos Framework."

FP8 (§ 1 "FP8"): "To serve Cosmos3-Nano in FP8, you can either: Download the prebuilt FP8 checkpoint from Hugging Face: `nvidia/Cosmos3-Nano-FP8`.
Quantize a checkpoint yourself by following the quantization guide." "To serve the prebuilt checkpoint, pass its Hugging Face ID to `--model-path`" —
`sglang serve --model-path nvidia/Cosmos3-Nano-FP8 --host 0.0.0.0`; "To serve a checkpoint you quantized yourself, pass its local directory to
`--model-path`. If the server runs in Docker, mount that directory and use its path inside the container." To quantize your own checkpoint:
`../cosmos3-inference-quantization/SKILL.md` (see Variant 1's note).

## Variant 5 — TensorRT-LLM (`trtllm-serve`, synchronous `POST /v1/videos/sync`)

Reference: [`references/trt-llm.md`](references/trt-llm.md) — read it before the first command ("Start the TensorRT-LLM server", "Request matrix",
the helper cell — `wait_for_server`, `CONTROL_CASES`, `run_transfer` — "Canny / Edge transfer"), `references/transfer-readme.md` "Run with
TensorRT-LLM", and the setup guide's "TensorRT-LLM Generator" section (install, `TRTLLM_ROOT`, Nano / Super launch, routes, `ffmpeg`, `use_guardrails`).

Install: "Follow the shared TensorRT-LLM setup, then launch one server from the TensorRT-LLM checkout" (notebook). Setup guide: "Install TensorRT-LLM
following its upstream documentation."; the source build (`git checkout cc63e66daf52644f1441e117bb7ba67d20566016`, the
`nvcr.io/nvidia/tensorrt-llm/devel:<tag>` container, `python3 scripts/build_wheel.py --use_ccache --skip_building_wheel --linking_install_binary`,
`pip install -e .`) "is the right path when you need a checkout that contains a recent Cosmos3 VisualGen change"; then `pip install cosmos_guardrail==0.3.2`
(headless hosts: swap in `opencv-python-headless==5.0.0.93`) "unless you explicitly disable guardrails before starting the server"; `TRTLLM_ROOT` must
hold `$TRTLLM_ROOT/examples/visual_gen/configs` (the setup guide's `test -d` preflight), and "Keep `ffmpeg` on the server `PATH`". Skill note
(unverified — not a route): `../cosmos3-generate-media/references/trt-llm.md` § 7.1–7.4 documents a prebuilt-wheel install (`tensorrt_llm==1.3.0rc27`)
that the generate-media eval verified for the audiovisual server; whether that wheel carries the transfer `extra_params` schema is unknown.

The notebook's launch, verbatim:

```bash
export TRTLLM_ROOT="${TRTLLM_ROOT:-$PWD}"

# Nano: one GPU
trtllm-serve nvidia/Cosmos3-Nano \
  --visual_gen_args "$TRTLLM_ROOT/examples/visual_gen/configs/cosmos3-nano-1gpu.yaml" \
  --port 8000

# Super: four GPUs (run instead of Nano)
torchrun --nproc_per_node=4 -m tensorrt_llm.commands.serve \
  nvidia/Cosmos3-Super \
  --visual_gen_args "$TRTLLM_ROOT/examples/visual_gen/configs/cosmos3-super-4gpu.yaml" \
  --port 8000
```

Skill note (not in the notebook): on one GPU run only the Nano lines (the block lists both launches — "run instead of Nano"); guardrails off for a
smoke run = `export TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1` before `trtllm-serve`, or `use_guardrails: false` in `extra_params` (setup guide; the notebook
sends `"use_guardrails": True`). Headless: `setsid trtllm-serve … > serve.log 2>&1 &`, poll `/health`, teardown by process group — pattern:
`../cosmos3-generate-media/references/skill-notes.md` § 4. Readiness: the notebook's `wait_for_server()` polls `GET <server root>/health` every 10 s
for up to 1800 s ("TensorRT-LLM server is ready"). "Install `requests` in this notebook kernel if needed, and install `ffmpeg` in the server environment."

Request (`run_transfer`): `POST /v1/videos/sync` with a JSON body — `prompt` / `negative_prompt` (compact JSON of `assets/edge/prompt.json` and
`assets/negative_prompt.json`), `size "1280x720"`, `num_frames 121`, `fps 30`, `num_inference_steps 50`, `guidance_scale 3.0`, `max_sequence_length 4096`,
`seed 2026`, `format "mp4"`, `response_format "file"`, `extra_params` = `{"use_duration_template": false, "use_resolution_template": false,
"use_system_prompt": false, "use_guardrails": true, "flow_shift": 10.0, "edge": {"preset_edge_threshold": "medium", "control": "<base64 of
assets/edge/control_edge.mp4>"}, "control_guidance": 1.5, "num_video_frames_per_chunk": 121, "num_conditional_frames": 1,
"num_first_chunk_conditional_frames": 0, "share_vision_temporal_positions": true, "emphasize_control_in_prompt": false, "max_frames": 121}` — headers
`Accept: video/mp4` and `Authorization: Bearer <COSMOS3_TRTLLM_API_KEY>` (default `tensorrt_llm`). "Every checked-in asset is already a precomputed
control, so the notebook sends it once as base64 media inside JSON `extra_params`; no top-level reference upload is needed." The response is checked —
`"video/mp4" in content_type` and `response.content[4:8] == b"ftyp"`, else `RuntimeError("Expected browser-compatible MP4 …")` — and written to
`outputs/notebooks/trt_llm/transfer_edge.mp4`. Request matrix: edge, blur and depth 121 frames / 30 fps with text / control guidance 3.0 / 1.5;
segmentation 3.0 / 2.0; WSM 100 frames / 10 fps, 1.0 / 3.0; "The blur control is 4:3 and explicitly selects `1104x832`; the other controls are 16:9 and
explicitly select `1280x720`." README: "Only edge and blur can be generated from a raw uploaded source (`"edge": true` or `"blur": true`); depth,
segmentation, and WSM always require precomputed control media. Precomputed edge and blur are also accepted. Use `use_guardrails`, not vLLM-Omni's
`guardrails`, and send encoded control bytes rather than a server-local `control_path`." "`/v1/videos/generations` remains only as a deprecated alias of
the canonical blocking `/v1/videos/sync` route."

FP8: the TensorRT-LLM transfer notebook and README have no FP8 route; to quantize your own checkpoint: `../cosmos3-inference-quantization/SKILL.md`
(see Variant 1's note).

## Verify the output

What the notebooks' viewer cells check, then verify the file yourself:

- Framework and Diffusers: `preview_transfer(control, model=…)` (`scripts/preview_helpers.py`) raises `missing output:
  <output_root>/<model>/transfer_<control>/vision.mp4 (run <control> inference first)` unless the file exists, then transcodes control and output with
  ffmpeg (`make_preview`: `libx264`, `-crf 28`, `-an`) and prints `<control> generated: vision.mp4 (<KB> KB -> <KB> KB preview)`. `preview_multi_control`
  also lists the framework's intermediate `control_edge.mp4` and `control_blur.mp4`.
- vLLM-Omni and SGLang: `view_transfer(control, output)` raises `missing output: … (run <control> transfer first)` unless `output_path.is_file()`, then
  previews control and output the same way.
- TensorRT-LLM: `run_transfer` itself verifies the response — non-empty, `video/mp4` content type, `ftyp` at bytes 4–8.
- Expected facts from the sources: edge, blur, depth and segmentation "121 frames @ 30 FPS", WSM "100 frames @ 10 FPS" (README table; `specs/wsm.json`
  carries `num_frames: 101`), multi-control "93 frames @ 16 FPS"; "`720` + `16,9` → 1280 × 720" (Spec field reference) — blur is 4:3 (`specs/blur.json`:
  `"aspect_ratio": "4,3"`; TensorRT-LLM `1104x832`). "Output frame count and geometry come from the control video" (README).
- Skill note (not in the source) — tool commands:
  `ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration -of default=nw=1 <file>`;
  where there is no ffprobe (the framework notebook's § 4 installs `ffmpeg` with apt; the diffusers venv has `av` and `imageio-ffmpeg`, which ships only
  an `ffmpeg` binary):
  `python -c "import av,sys; c=av.open(sys.argv[1]); s=c.streams.video[0]; n=sum(1 for _ in c.decode(s)); print(s.codec_context.name, s.width, s.height, n, float(s.average_rate))" <file>`
  (`pip install av` elsewhere). Record codec, size, frame count, fps and duration in `result_summary.md`; Variant 2: also the `cuda allocated … GiB` figure that the § 7 runner's `get_pipe` prints after `pipe.to("cuda")` (`loaded pipeline in …s (precision=…); cuda allocated … GiB`).

## Reference files

The complete cookbook content lives in `references/`, migrated 1:1 — read the matching file before the first command (see the top). Do not look for
the notebooks or the cookbook README; these files replace them.

| File                             | Content                                                                                                                                                                                                     |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `references/transfer-readme.md`  | The cookbook README, verbatim: introduction, Transfer Definition, the four backend quickstarts, Spec field reference, Cookbook entrypoints, Troubleshooting (NATTEN / Torch mismatch, `PYTHONNOUSERSITE=1`) |
| `references/cosmos-framework.md` | Variant 1 — `run_video_transfer_with_cosmos_framework.ipynb` (self-contained setup, Nano § 9–§ 13, Super § 14–§ 18, multi-control § 19)                                                                     |
| `references/diffusers.md`        | Variant 2 — `run_video_transfer_with_diffusers.ipynb` (venv, `run_transfer`, FP8 loader, five controls on Nano then Super)                                                                                  |
| `references/vllm-omni.md`        | Variant 3 — `run_video_transfer_with_vllm_omni.ipynb` (launch, Quantized Checkpoints, request helpers, five controls)                                                                                       |
| `references/sglang.md`           | Variant 4 — `run_video_transfer_with_sglang.ipynb` (launch, FP8, request helpers, five controls)                                                                                                            |
| `references/trt-llm.md`          | Variant 5 — `run_video_transfer_with_trt_llm.ipynb` (launch, request matrix, base64 control contract, five controls)                                                                                        |
| `scripts/preview_helpers.py`     | The cookbook's preview helpers (spec loading, ffmpeg previews), unchanged                                                                                                                                   |
| `specs/`                         | The cookbook's framework specs, unchanged (one per control plus `multi_control.json`)                                                                                                                       |

Origin: the `.agents/skills/cookbooks/cosmos3/generator/transfer` notebooks, README and `preview_helpers.py` of the public cosmos-framework repository
(`main` bbd4a67); superseded by this skill. The cookbook keeps `assets/` (LFS) and `specs/`: the framework tests import the specs and the NIM examples
reference the assets.

## Related skills

| Skill                                                                                                       | When to use                                                                                                                                                                         |
| ----------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `../cookbooks/cosmos3/README.md`                                                                            | The shared environment setup every variant points at (Cosmos Framework, Diffusers, TensorRT-LLM Generator, vLLM-Omni, NIM)                                                          |
| `../cosmos3-setup/SKILL.md`                                                                                 | Framework installation and checkpoint download (Variant 1)                                                                                                                          |
| `../cosmos3-inference/SKILL.md`                                                                             | Framework-native inference reference (`cosmos_framework.scripts.inference` arguments, parallelism presets)                                                                          |
| `../cosmos3-env-troubleshoot/SKILL.md`                                                                      | CUDA / import / server failures                                                                                                                                                     |
| `../cosmos3-generate-media/SKILL.md`                                                                        | Text-, image- and video-to-video generation without a control video on the same backends; `../cosmos3-generate-media/references/skill-notes.md` § 4 has the headless server pattern |
| `../cookbooks/cosmos3/nim/transfer.md`, `../cookbooks/cosmos3/nim/.agents/skills/cosmos3-nim-user/SKILL.md` | Transfer through the Cosmos3 NIM (`POST /v1/infer`, `model_mode=video2video`) — not covered by the cookbook sources of this skill                                                   |
| `../cosmos3-nano-transfer-auto/inference/SKILL.md`, `../cosmos3-nano-sim-transfer/inference/SKILL.md`       | The Transfer-Auto and Sim-Transfer world-simulation models (different checkpoints and inputs)                                                                                       |
| `../cosmos3-inference-quantization/SKILL.md`                                                                | Making your own FP8 checkpoint (the released FP8 routes are in the variant blocks)                                                                                                  |
| `../cosmos3-skill-authoring/SKILL.md`                                                                       | Editing this skill's eval cases                                                                                                                                                     |
