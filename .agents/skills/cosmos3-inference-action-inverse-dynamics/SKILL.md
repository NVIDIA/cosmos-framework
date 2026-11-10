---
name: cosmos3-inference-action-inverse-dynamics
description: >
  Predict the action trajectory behind an observed video with the Cosmos3 action models (inverse dynamics): an
  autonomous-vehicle ego clip or a LeRobot robot episode in, a per-step 9D pose-delta trajectory out, through diffusers
  in-process, the cosmos_framework PyTorch entrypoint, vLLM-Omni, SGLang or TensorRT-LLM. Use when the user asks
  "recover the ego-motion / end-effector trajectory from this video", "run Cosmos3 inverse dynamics", "predict the
  actions from a clip with Cosmos3-Nano", "action inverse dynamics with vLLM-Omni / SGLang / TensorRT-LLM / diffusers",
  or which backend to use for it. Not for forward dynamics (image + actions to video), policy inference or audiovisual
  generation, which have their own skills. The sample inputs ship in the action entry skill's assets/
  (cosmos3-inference-action).
---

# Cosmos3 Action Inverse Dynamics

One block of the Action skill family: the five `run_id_with_*` notebooks of the Action cookbook, migrated 1:1 into
`references/`, one serving-path variant each. "Inverse dynamics is the reverse of forward dynamics: given a video, it
predicts the ego-motion action trajectory that produced it" (`references/cosmos-framework.md`). The entry skill
`../cosmos3-inference-action/SKILL.md` holds the cookbook README (`../cosmos3-inference-action/references/action-readme.md`:
Action Definition, per-backend quickstarts) and every sample asset; forward dynamics and policy are sibling blocks.

## Read this first

Before the first command of any variant:

1. Read the variant's reference file in full (linked under its heading, with the sections the smoke path needs). The
   variant sections below only summarise the notebooks; the reference files are the notebooks migrated 1:1 (markdown
   and code cells in order, no outputs). After a failure, go back to the reference section that covers the failing
   step before improvising a fix. Do not look for the original notebooks; the reference files replace them.
2. Check the GPU with `nvidia-smi`: the driver's CUDA major version picks the torch backend / dependency group
   (`cu130` for 13.x, `cu128` for 12.x — the setup guide's table, `../cookbooks/cosmos3/README.md`); the memory picks
   the checkpoint (table below). Every notebook runs one GPU (`CUDA_VISIBLE_DEVICES=0`, `--gpus '"device=0"'`).
3. Assets. Skill note (not in the source): this skill has none of its own. The notebooks resolved `assets/` next to
   themselves (`cookbooks/cosmos3/generator/action/assets/…` against the cookbook root); the entry skill vendors those
   files at `../cosmos3-inference-action/assets/` (same layout), while the cookbook's own `assets/` directory holds only
   git-LFS pointers in a plain checkout. Each variant shows the one-line re-point that makes the notebook's code read the
   vendored copies. Run every block from this skill directory (the one with `references/` and `eval/`) with the whole
   `.agents/skills` tree beside it — the notebooks' `find_repo_root()` walks up to the directory that contains
   `cookbooks/cosmos3`.
4. Skill note (not in the source): blocks do not share shell state. The notebooks' Jupyter kernel carried `os.environ`
   from the Python configuration cells into the `%%bash` cells; here, run a Python spec block and its bash run block as
   one script, or re-export the variables the bash block reads (each variant lists them). An `export` exists only in the
   shell call that ran it.
5. Skill note (not in the source): a cosmos-framework checkout is needed by Variant 2 (it runs there) and by every
   viewer cell (they import `pose_rel_to_abs`, `denormalize_action` and the LeRobot readers from `cosmos_framework`):
   point `COSMOS3_REPO` at one (`../cosmos3-setup/SKILL.md`; this skill directory usually sits inside such a checkout,
   three levels up). Verify what you produced before reporting it (Verify the output below) and record the facts in
   `result_summary.md`.

## When to use this skill

- Use to predict the ego-motion trajectory of an autonomous-vehicle clip (the AV examples of every notebook) or the
  end-effector poses of a LeRobot robot episode (Bridge, AgiBotWorld-Beta, RoboMIND Franka / dual-arm / UR, UMI,
  Fractal — framework and diffusers notebooks) with `Cosmos3-Nano` (default), `Cosmos3-Edge` or `Cosmos3-Super`
- Use to choose a serving path for it: in-process (diffusers, cosmos_framework) or a server (vLLM-Omni, SGLang, TensorRT-LLM)
- For forward dynamics (image + trajectory -> video) use `../cosmos3-inference-action-forward-dynamics/SKILL.md`; for
  policy (image + instruction + state -> video + actions) `../cosmos3-inference-action-policy/SKILL.md`; the README,
  action definition and assets are in `../cosmos3-inference-action/SKILL.md`
- For audiovisual generation (t2i / t2v / i2v) this is the wrong skill (`../cosmos3-generate-media/SKILL.md`); for
  environment setup problems hand off to `../cosmos3-env-troubleshoot/SKILL.md`

## Bundled assets (in the entry skill)

The notebooks' inputs, as the entry skill vendors them (paths relative to this skill directory; the layout is the
cookbook's `assets/`). Skill note (not in the source): vendored = a plain file (the entry skill's `.gitattributes` exempts
it from LFS); not vendored = still a git-LFS object under `../cookbooks/cosmos3/generator/action/assets/` (too large).

| Example (notebook case)                                         | Path                                                                                 | Used by                                 | Vendored              |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------ | --------------------------------------- | --------------------- |
| AV clip 0 (`av_inverse_0`)                                      | `../cosmos3-inference-action/assets/videos/av_0.mp4`                                 | all five notebooks                      | yes (1.1 MB)          |
| AV clip 1 (`av_inverse_1`)                                      | `../cookbooks/cosmos3/generator/action/assets/videos/av_1.mp4`                       | framework, diffusers, SGLang, vLLM-Omni | no (1.6 MB)           |
| Bridge (`bridge_inverse`; framework § 7)                        | `../cosmos3-inference-action/assets/bridge_lerobot_example/`                         | framework, diffusers                    | yes                   |
| AgiBotWorld-Beta (`agibotworld_inverse`; framework § 8)         | `../cookbooks/cosmos3/generator/action/assets/agibotworld_beta_lerobot_example/`     | framework, diffusers                    | no (7–11.5 MB videos) |
| RoboMIND Franka (`robomind_franka_inverse`; framework § 9)      | `../cosmos3-inference-action/assets/robomind_lerobot_example/franka/`                | framework, diffusers                    | yes                   |
| RoboMIND Franka dual-arm (`robomind_franka_dual_inverse`; § 10) | `../cookbooks/cosmos3/generator/action/assets/robomind_lerobot_example/franka_dual/` | framework, diffusers                    | no (7–36 MB videos)   |
| RoboMIND UR (`robomind_ur_inverse`; framework § 11)             | `../cookbooks/cosmos3/generator/action/assets/robomind_lerobot_example/ur/`          | framework, diffusers                    | no (1.9 MB video)     |
| UMI (`umi_inverse`; framework § 12)                             | `../cosmos3-inference-action/assets/umi_lerobot_example/`                            | framework, diffusers                    | yes                   |
| Fractal (`fractal_inverse`; framework § 13)                     | `../cosmos3-inference-action/assets/fractal_lerobot_example/`                        | framework, diffusers                    | yes                   |

Skill note (not in the source): a non-vendored example runs from the cookbook's `assets/` once its LFS objects are
materialised as the entry skill's "Bundled sample assets" section describes (`git lfs pull`, or the public repository's
LFS objects at the pinned commit); then point `resolve_input(...)` / `asset_path(...)` / dataset `root` at
`../cookbooks/cosmos3/generator/action/assets/<example>`. The smoke cases of `eval/eval.config` use AV clip 0 only.

## Model sizes vs GPU

The sources name models and GPU counts, not memory figures (skill note, not in the source: the bf16 text-to-image
footprints in `../cosmos3-generate-media/references/skill-notes.md` § 2 are audiovisual measurements, not action ones):

| Checkpoint             | What the sources say                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `nvidia/Cosmos3-Nano`  | The default of every notebook ("Runs `Cosmos3-Nano` on every line of the spec"; README: inverse dynamics "using the Cosmos3-Nano"); one GPU in every launch (`--gpus '"device=0"'`, `cosmos3-nano-1gpu.yaml`, `CUDA_VISIBLE_DEVICES=0`)                                                                                                                                                                                                                                             |
| `nvidia/Cosmos3-Edge`  | Framework: "set `COSMOS3_CHECKPOINT_PATH=Cosmos3-Edge` or `Cosmos3-Super` before the configuration cell — nothing else changes"; vLLM-Omni: a Nano or an Edge server ("The commands are mutually exclusive"); setup guide: "Edge is the compact 4B checkpoint"; TensorRT-LLM (the generate-media notebook, `../cosmos3-generate-media/references/trt-llm.md`): "its action weights are not served by this pipeline"; the diffusers notebook's `MODEL_IDS` lists Nano and Super only |
| `nvidia/Cosmos3-Super` | Framework `COSMOS3_CHECKPOINT_PATH=Cosmos3-Super`; diffusers `MODEL_IDS["Cosmos3-Super"]`; vLLM-Omni FP8: "For Super, add the upstream `--tensor-parallel-size 4` and `--enable-layerwise-offload` options if needed"; setup guide: TensorRT-LLM Super on "four GPUs"                                                                                                                                                                                                               |
| FP8 checkpoints        | `nvidia/Cosmos3-Nano-FP8` and `nvidia/Cosmos3-Super-FP8` (diffusers, vLLM-Omni), `nvidia/Cosmos3-Edge-FP8` (vLLM-Omni), `nvidia/Cosmos3-Nano-FP8` (SGLang) — the FP8 line of each variant; to quantize your own checkpoint: `../cosmos3-quantization/SKILL.md`                                                                                                                                                                                                                      |

All checkpoints are gated on Hugging Face: the notebooks require model access and `uvx hf@latest auth login` or
`HF_TOKEN`. "Generator requires the Guardrail" (the gated `nvidia/Cosmos-1.0-Guardrail`); the README's switches to run
without it: `enable_safety_checker=False` (Diffusers; the notebook's `COSMOS3_DIFFUSERS_GUARDRAILS=false`),
`use_guardrails: false` (TensorRT-LLM `extra_params`), `guardrails: false` (vLLM-Omni `extra_params`/`extra_args`),
`--no-guardrails` (Cosmos Framework) — each variant names the one it uses.

## Variant 1 — diffusers (in-process, no server)

Reference: [`references/diffusers.md`](references/diffusers.md) — read it before the first command (§ 1 Prerequisites,
§ 2 Configure Paths and Environment, § 3 Install, § 7 the helper cell: `ACTION_SETS`, `LEROBOT_ID_SETS`, `get_pipe`,
`run_action`, `run_lerobot_action`, `view_action`, `view_lerobot_action`; "Quantized Checkpoints (FP8)"; Use Cases).

The notebook's environment (§ 2), with the torch backend matching the driver's CUDA major version, plus two skill-note lines at
the end. § 1: "use `uv 0.11.29`, matching the version pinned by Cosmos Framework" — § 1's installer, which the guard line runs when
`uv --version` differs (skill note, not in the source: `uv self update 0.11.29` is refused by a package-installed uv such as a host's 0.13.0, and the framework checkout this skill sits in applies the pin to any `uv` run under it):

```bash
export COSMOS3_DIFFUSERS_ACTION_VENV=/path/to/.venv-cosmos3-diffusers-action
export COSMOS3_REPO=/path/to/packages/cosmos3
export COSMOS3_TORCH_BACKEND=cu130
export HF_HOME=/path/to/large/huggingface/cache
export UV_LINK_MODE=copy
export CUDA_VISIBLE_DEVICES=0
export COSMOS3_DIFFUSERS_PRECISION=bf16  # or fp8
export COSMOS3_ACTION_OUTPUT_ROOT="$PWD/outputs"   # skill note (not in the source): the notebook's default is <cookbook>/outputs/notebooks/diffusers
export COSMOS3_DIFFUSERS_GUARDRAILS=false           # skill note (not in the source): the notebook's § 1 switch, for runs without Cosmos-1.0-Guardrail access
apt-get update && apt-get install -y libxcb1 libgl1 libglib2.0-0   # skill note (not in the source): § 1 "Headless servers" BEFORE the first run, not after the error — Cosmos3OmniPipeline imports cosmos_guardrail -> cv2, which needs libxcb.so.1 even with guardrails off (eval 2026-11-08: all four first runs died on it)
uv --version 2>/dev/null | grep -q '^uv 0\.11\.29' || { curl -LsSf https://astral.sh/uv/0.11.29/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh; export PATH="$HOME/.local/bin:$PATH"; hash -r; }   # skill note (not in the source): § 1's pin — the standalone installer, not `uv self update`, which a package-installed uv refuses
```

Install (skill note, not in the source): the block above — its last two lines are the headless pre-flight and the uv guard — and the
§ 3 cell verbatim, as ONE shell call: `uv venv "$COSMOS3_DIFFUSERS_ACTION_VENV" --python 3.13 --seed --managed-python --allow-existing`,
then its pinned `uv pip install --torch-backend="$COSMOS3_TORCH_BACKEND" …` list (its last lines register the Jupyter kernel, not needed here).

Run (skill note, not in the source): with `"$COSMOS3_DIFFUSERS_ACTION_VENV/bin/python"`, from this skill directory, ONE
file made of the § 4 restore cell (`configure_diffusers_environment()`), the re-point below, the § 7 helper cell verbatim
and the use-case lines, in the SAME shell call as the § 2 export block repeated (a new call has none of its exports: the
restore cell then defaults `COSMOS3_REPO` to `<cookbook root>/packages/cosmos3`, the output root to the cookbook's
`outputs/notebooks/diffusers` and guardrails to on). `COSMOS3_REPO` must hold a framework checkout (the helper cell
imports `denormalize_action` and `pose_rel_to_abs` from it); the restore cell derives `COSMOS3_ACTION_ROOT` as
`<cookbook root>/cookbooks/cosmos3/generator/action`, whose `assets/` are LFS pointers here (Read this first, 3):

```python
# Skill note (not in the source): insert after configure_diffusers_environment() and before the helper cell. Every
# asset_path("assets/...") of ACTION_SETS / LEROBOT_ID_SETS then resolves to the entry skill's vendored copies;
# COSMOS_ROOT stays as the cell set it (.agents/skills), which the helpers' relative_to(COSMOS_ROOT) prints need.
COSMOS3_ACTION_ROOT = Path("../cosmos3-inference-action").resolve()
```

```python
av_inverse_0_output = run_action("av_inverse_0")
```

```python
view_action("av_inverse_0")
```

`run_action` loads `nvidia/Cosmos3-Nano` (`model="Cosmos3-Nano"` default), uses `FIXED_SAMPLING` (30 steps, guidance
1.0, shift 10.0, seed 0) and `CosmosActionCondition(mode="inverse_dynamics", chunk_size=60, domain_name="av",
resolution_tier=480, video=<the clip>, view_point="ego_view")` at `fps=10`, and writes
`<COSMOS3_ACTION_OUTPUT_ROOT>/av_inverse_0/av_inverse_0_action.json` (`result.action[0]` as a list); `view_action`
integrates it with `pose_rel_to_abs(..., translation_scale=1.35)` and saves `av_inverse_0_trajectory.png` beside it.
Robot episodes: `run_lerobot_action("bridge_inverse")` / `view_lerobot_action("bridge_inverse")` (one 16-transition
window; `umi_inverse`, `fractal_inverse`, `robomind_franka_inverse` are vendored too, the other three are not).

FP8 (notebook, "Quantized Checkpoints (FP8)"): "Default loads are BF16 (`COSMOS3_DIFFUSERS_PRECISION=bf16`). Set the env var to
`fp8` before running the helper cell's `get_pipe` … Nano and Super load `nvidia/Cosmos3-Nano-FP8` and `nvidia/Cosmos3-Super-FP8`.
Do not pass `revision=fp8` on the BF16 cards." "Optional: point `COSMOS3_FP8_MODEL_PATH` at a local Diffusers checkpoint
directory." To quantize your own checkpoint: the `cosmos3-quantization` skill (FP8 row of "Model sizes vs GPU").

## Variant 2 — cosmos_framework (native PyTorch, in-repo)

Reference: [`references/cosmos-framework.md`](references/cosmos-framework.md) — read it before the first command
(§ 1–5: Prerequisites, Configure Paths, Clone or Reuse, Install, Verify; "Import dependencies and define helper
functions"; § 6 AV Inverse Dynamics: spec, run, visualize; § 7–13 the robot episodes).

"Tested path from the audit: … Install command: `uv sync --all-extras --group=cu130-train` … Models: `Cosmos3-Nano`
(default), `Cosmos3-Edge`, and `Cosmos3-Super`." Knobs (§ 2, exports): `COSMOS3_REPO`, `COSMOS3_GIT_URL`
(`https://github.com/NVIDIA/cosmos-framework.git`), `COSMOS3_UV_GROUP` (`cu130-train`; "use cu128-train for a CUDA
12.x driver"), `COSMOS3_OUTPUT_ROOT`, `COSMOS3_CHECKPOINT_PATH` ("defaults to Cosmos3-Nano"), `HF_HOME`,
`CUDA_VISIBLE_DEVICES`. Clone or reuse the checkout with the § 3 cell (`git clone "$COSMOS3_GIT_URL" "$COSMOS3_REPO"`
unless `$COSMOS3_REPO/.git` exists), then install with the § 4 cell (`export GIT_LFS_SKIP_SMUDGE=1`, `cd "$COSMOS3_REPO"`,
`uv sync --all-extras --group="$COSMOS3_UV_GROUP"` — "May take several minutes"; skill note, not in the source: it aborts with `Required uv
version ==0.11.29 does not match the running version` on any other uv — put Variant 1's uv guard first in the same call (the `uv --version … || { curl … install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh; …; }` line at the end of its export block), not `uv self update`, which a package-installed uv refuses), each as ONE shell call with those variables exported.

Spec (§ 6 "Create the AV Inverse-Dynamics Input Spec"): a JSONL, one record per run, `vision_path` absolute "because
the loader resolves relative paths against the spec file's own directory" — `action_chunk_size` 60, `domain_name`
`"av"`, `fps` 10, `image_size` 480, `view_point` `"ego_view"`, `model_mode` `"inverse_dynamics"`, `prompt`
"You are an autonomous vehicle planning system.", `seed` 0. Run, from this skill directory, the lines below followed
by the notebook's spec cell verbatim from `records = [` to its end (it writes
`<COSMOS3_INPUT_DIR>/action_inverse_dynamics_av_custom.jsonl` and sets `COSMOS3_ID_INPUT` / `COSMOS3_ID_OUTPUT`):

```python
# Skill note (not in the source): the notebook's configuration cell sets COSMOS_ROOT with find_repo_root() and resolves
# `cookbooks/cosmos3/generator/action/assets/...` against it (LFS pointers here); these lines replace it for the AV spec
# and point input_videos at the entry skill's vendored clip (the notebook lists av_0 and av_1; av_1 is not vendored).
from pathlib import Path
import json, os
COSMOS_ROOT = Path.cwd().resolve()
COSMOS3_OUTPUT_ROOT = Path(os.environ.get("COSMOS3_OUTPUT_ROOT", COSMOS_ROOT / "outputs")).resolve()
COSMOS3_INPUT_DIR = COSMOS3_OUTPUT_ROOT / "inputs"
def resolve_input(rel_path: str) -> str:      # the notebook's helper, unchanged
    path = (COSMOS_ROOT / rel_path).resolve()
    assert path.exists(), f"missing input: {path}"
    return str(path)
input_videos = {"av_inverse_0": "../cosmos3-inference-action/assets/videos/av_0.mp4"}
# --- the notebook's spec cell follows, verbatim, from `records = [` ---
```

Run (§ 6 "Run AV Inverse-Dynamics Inference"): "Runs `Cosmos3-Nano` on every line of the spec. Inverse dynamics
predicts an action rather than a video, so each run writes its result to `<output>/<name>/sample_outputs.json`. The
predicted action trajectory is stored under `outputs[0].content["action"]`." The cell reads its variables from the
kernel's environment; the exports before the cell text replace that, and the link line restores § 5's `LD_LIBRARY_PATH` (values = the notebook's defaults, except the output root, which this skill points at `./outputs` as the Python block above does):

```bash
# Skill note (not in the source): the environment the notebook's configuration and spec cells exported — same shell call as the cell.
export COSMOS3_REPO="${COSMOS3_REPO:-/path/to/cosmos-framework}"
export COSMOS3_OUTPUT_ROOT="${COSMOS3_OUTPUT_ROOT:-$PWD/outputs}"
export COSMOS3_ID_INPUT="$COSMOS3_OUTPUT_ROOT/inputs/action_inverse_dynamics_av_custom.jsonl"
export COSMOS3_ID_OUTPUT="$COSMOS3_OUTPUT_ROOT/action_inverse_dynamics_av_custom"
export COSMOS3_CHECKPOINT_PATH="${COSMOS3_CHECKPOINT_PATH:-Cosmos3-Nano}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export COSMOS3_MASTER_ADDR="${COSMOS3_MASTER_ADDR:-127.0.0.1}"
export COSMOS3_NANO_TEXT_MASTER_PORT="${COSMOS3_NANO_TEXT_MASTER_PORT:-$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')}"   # the notebook's free_local_port()
# Skill note (not in the source): § 5's configure_cosmos_framework_runtime_env() for this shell — torchcodec dlopens libavutil.so.60 & co., which the venv has only as PyAV's hashed av.libs files; without the seven links: "Could not load libtorchcodec" (eval 2026-11-08, three of four cells)
d="$COSMOS3_OUTPUT_ROOT/torchcodec_ffmpeg_links"; mkdir -p "$d"; for f in "$COSMOS3_REPO"/.venv/lib/python3.*/site-packages/av.libs/lib{avcodec,avdevice,avfilter,avformat,avutil,swresample,swscale}-*.so.*; do b=${f##*/}; v=${b##*.so.}; ln -sfn "$f" "$d/${b%%-*}.so.${v%%.*}"; done; export LD_LIBRARY_PATH="$d:${f%/*}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# --- the notebook's run cell from here on, verbatim ---
set -euo pipefail

cd "$COSMOS3_REPO"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
MASTER_ADDR="$COSMOS3_MASTER_ADDR" MASTER_PORT="$COSMOS3_NANO_TEXT_MASTER_PORT" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$COSMOS3_ID_INPUT" \
  -o "$COSMOS3_ID_OUTPUT" \
  --checkpoint-path "$COSMOS3_CHECKPOINT_PATH" \
  --seed 0 \
  --benchmark
```

Guardrails: the cell runs with the framework default (on — the gated `nvidia/Cosmos-1.0-Guardrail` plus the setup guide's system
packages: "`uv sync --all-extras` alone is not enough for guardrails"); the README's switch is `--no-guardrails` (setup guide: "pass
`--no-guardrails` to `cosmos_framework.scripts.inference`"). Skill note (not in the source): for the smoke cases of `eval/eval.config` (guardrails disabled) append it as the command's last option.

Visualize (§ 6): "The output action can be fed into `pose_rel_to_abs` to convert back to camera poses (camera-to-world
transformation, opencv convention, unit in meter)" — the "Import dependencies and define helper functions" cell plus
the "Visualize the AV Predicted Action" cell. Skill note (not in the source): run them with
`"$COSMOS3_REPO/.venv/bin/python"` (the `--all-extras` venv has `numpy`, `matplotlib`, `mediapy`, `scipy`, `torch`);
`media.show_video` / `plt.show()` are Jupyter display plumbing — pass `save_path=<file>.png` to `visualize_pose`. The
robot episodes (§ 7–13) hand `resolve_input("cookbooks/cosmos3/generator/action/assets/<example>")` to the LeRobot
readers: give them `"../cosmos3-inference-action/assets/<example>"` instead (vendored examples; others: Bundled assets). FP8: the notebook has no FP8 section.

## Variant 3 — vLLM-Omni (Docker server + OpenAI-style video API)

Reference: [`references/vllm-omni.md`](references/vllm-omni.md) — read it before the first command (Prerequisites;
Start vLLM-Omni Server — Cosmos3-Nano / Cosmos3-Edge and "Quantized Checkpoints"; Configure Notebook Variables; Create
the Inverse-Dynamics Input Spec; Run Inverse-Dynamics Inference; Visualize the Predicted Action).

The notebook's Nano launch ("start that server in a terminal from the cookbook root"; both launches "publish the
container's port `8000` to host port `8001`, which matches the notebook's default `http://localhost:8001` endpoint"),
plus two lines that are not in the notebook, marked: `-e HF_TOKEN` (Prerequisites: "also set and pass `HF_TOKEN` to
the container" — stated for guardrails, but the checkpoint download is gated too;
`../cosmos3-generate-media/references/vllm-omni.md` § 7.1: anonymous Hub downloads were rate-limited on a cold cache)
and the setup guide's cache mount `-v "${HF_HOME}:/root/.cache/huggingface"` (its vLLM-Omni Docker command; without it
every container start downloads the checkpoint again):

```bash
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"   # skill note (not in the notebook): the setup guide's export for the mount below
docker rm -f cosmos3-vllm-omni-notebook 2>/dev/null || true

docker run -d --name cosmos3-vllm-omni-notebook \
  --runtime nvidia --gpus '"device=0"' \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e HF_TOKEN \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "$PWD:/workspace" \
  -v "$PWD:$PWD" \
  -w /workspace \
  -p 8001:8000 --ipc=host \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Nano \
    --no-guardrails \
    --omni \
    --model-class-name Cosmos3OmniDiffusersPipeline \
    --allowed-local-media-path / \
    --port 8000 \
    --init-timeout 1800

# Wait until this returns model metadata before running the inference cells.
curl http://localhost:8001/v1/models
```

Readiness: the `curl` above "returns model metadata" when the server is up; setup guide: "vLLM-Omni prints
`Application startup complete.` when the API is ready" and "always pass `--init-timeout 1800`". Headless pattern (poll
`/v1/models`, keep `docker logs`, `docker stop && docker rm`): `../cosmos3-generate-media/references/skill-notes.md`
§ 4. Edge: the reference's "Cosmos3-Edge" block — the same command with `nvidia/Cosmos3-Edge`.

Client: the configuration cell verbatim (`find_framework_root()`: "Launch from the Framework checkout or a notebook
directory inside it"; it sets `COSMOS_ROOT` to the checkout's `.agents/skills`; `COSMOS3_VLLM_OUTPUT_ROOT` = output
root, `COSMOS3_VLLM_BASE_URL` default `http://localhost:8001`), the spec cell verbatim except its `input_videos` dict —
skill note (not in the source): write `input_videos = {"av_inverse_0": "cosmos3-inference-action/assets/videos/av_0.mp4"}`,
which the cell's `resolve_input()` resolves against that `COSMOS_ROOT` to the entry skill's vendored clip (`av_1` is not
vendored) — then the "Run Inverse-Dynamics Inference" cell — as ONE file with a python that has `requests` ("Install
requests in this notebook kernel: `pip install requests`"). Per record it posts multipart
`POST {VLLM_BASE_URL}/v1/videos` with `files={"input_reference": (video_path.name, video_file, "video/mp4")}`, the
form `prompt`, `num_frames` = `action_chunk_size + 1`, `fps`, `num_inference_steps` 30, `guidance_scale` 1.0,
`flow_shift` 10.0, `seed`, and `extra_params` = `json.dumps` of `{"action_mode": "inverse_dynamics", "domain_name":
record["domain_name"], "action_chunk_size": record["action_chunk_size"], "image_size": record["image_size"],
"view_point": record["view_point"], "raw_action_dim": 9, "guardrails": False}`; then polls `GET /v1/videos/{id}` every
2 s until `status == "completed"`. `final["action"]["data"]` is written as `<output>/<name>/sample_outputs.json`
(`{"outputs": [{"content": {"action": …}}]}`) with `response.json`, `final.json` and `action.json` beside it; the cell
prints `action shape:` and `dtype:`. Visualize: the notebook's last cell (Variant 2's, `COSMOS3_REPO` on `sys.path`).

FP8 (notebook, "Quantized Checkpoints"): "Published Edge FP8, Nano FP8 and Super FP8 checkpoints can be downloaded to
a local directory … Serve that directory with the original `--served-model-name` so the request cells keep their
existing model IDs. Do not select FP8 with `--revision` alone: the Cosmos3 vLLM-Omni pipeline must receive a
materialized local checkpoint." Its blocks run "from the **Framework checkout root**": a download
(`snapshot_download("nvidia/Cosmos3-Nano-FP8", local_dir=…)`, `SERVED_MODEL=nvidia/Cosmos3-Nano`; Edge likewise) and
a launch that mounts `FP8_MODEL_DIR` (and `FP8_SOURCE_DIR`: "The source mount preserves exported symlinks") read-only,
checks `test -f "$FP8_MODEL_DIR/hf_quant_config.json"` and serves `vllm serve "$FP8_MODEL_DIR" --served-model-name
"$SERVED_MODEL" --no-guardrails …`. To quantize your own checkpoint: the `cosmos3-quantization` skill (FP8 row above).

## Variant 4 — SGLang (Docker server + action generation API)

Reference: [`references/sglang.md`](references/sglang.md) — read it before the first command (Set up HF Cache; Start
SGLang Server; FP8; Configure Notebook Variables; Create the Inverse-Dynamics Input Spec; Run Inverse-Dynamics
Inference; Visualize the Predicted Action). The route is `POST /v1/actions/generations`.

The notebook's cache block and Nano launch ("Start the server in a terminal from the cookbook root"), plus two `-e`
lines that are not in the notebook, marked: `SGLANG_DISABLE_COSMOS3_GUARDRAILS=1` — the generate-media eval found that
`lmsysorg/sglang:dev` loads the gated guardrail at startup and exits with `GatedRepoError 401` without it
(`../cosmos3-generate-media/references/sglang.md` § 2; the request-level `guardrails: False` this notebook sends does
not prevent that exit) — and `HF_TOKEN` (anonymous Hub downloads are rate-limited on a cold cache; same source). Both
are skill additions not re-verified for the action route:

```bash
export SGLANG_HF_CACHE="${SGLANG_HF_CACHE:-$HOME/.cache/sglang-huggingface}"
mkdir -p "$SGLANG_HF_CACHE"
chmod 777 "$SGLANG_HF_CACHE"
```

```bash
docker rm -f cosmos3-sglang-notebook 2>/dev/null || true

docker run -d --init --name cosmos3-sglang-notebook \
  --runtime nvidia --gpus '"device=0"' \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e SGLANG_DISABLE_COSMOS3_GUARDRAILS=1 \
  -e HF_TOKEN \
  -v $SGLANG_HF_CACHE:/root/.cache/huggingface \
  -v "$PWD:/workspace" \
  -p 30000:30000 --ipc=host \
  lmsysorg/sglang:dev \
  sglang serve \
  --model-path nvidia/Cosmos3-Nano \
  --host 0.0.0.0

# Wait until this returns model metadata before running the inference cell.
curl http://localhost:30000/v1/models
```

Readiness: the `curl` above; the generate-media eval observed `GET /health` turning from 503 to 200
`{"status":"ok"}` after `The server is fired up and ready to roll!` (`../cosmos3-generate-media/references/sglang.md`
§ 2 skill note and § 7.1 — a text-to-image measurement, not this notebook's). Headless pattern:
`../cosmos3-generate-media/references/skill-notes.md` § 4.

Client: the configuration cell verbatim (prose: "set `COSMOS3_REPO` to the root of a local checkout of the
**cosmos-framework** repository"; the cell reads `COSMOS3_FRAMEWORK_PATH`, default two levels above the cookbook root
`COSMOS_ROOT` = `.agents/skills`, i.e. the checkout this skill sits in; `COSMOS3_SGLANG_OUTPUT_ROOT` = output root,
`COSMOS3_SGLANG_BASE_URL` default `http://localhost:30000`), the spec cell verbatim except its `input_videos` dict — skill
note (not in the source): write `input_videos = {"av_inverse_0": "cosmos3-inference-action/assets/videos/av_0.mp4"}`, which
`resolve_input()` resolves against that `COSMOS_ROOT` to the entry skill's vendored clip (`av_1` is not vendored) — then
the "Run Inverse-Dynamics Inference" cell — as ONE file with a python that has `requests`. Its form is Variant 3's, posted as multipart
`POST {SGLANG_BASE_URL}/v1/actions/generations` (`files={"input_reference": …}`, timeout 120; a non-OK response is
saved as `submit_error_response.txt`) with `extra_params` = `json.dumps` of `{"action_mode": "inverse_dynamics",
"domain_name": record["domain_name"], "action_view_point": record["view_point"], "raw_action_dim": 9, "guardrails":
False}`. The response's `data[0]["action"]` (`values`, `shape`, `dtype`) is written as `action.json` and, as `values`,
into `<output>/<name>/sample_outputs.json` (`{"outputs": [{"content": {"action": …}}]}`). Visualize: the last cell
(Variant 2's, with `COSMOS3_FRAMEWORK_PATH` on `sys.path`).

FP8 (notebook, "FP8"): "To serve Cosmos3-Nano in FP8, you can either: Download the prebuilt FP8 checkpoint from
Hugging Face: `nvidia/Cosmos3-Nano-FP8`. Quantize a checkpoint yourself …" — `sglang serve --model-path
nvidia/Cosmos3-Nano-FP8 --host 0.0.0.0`, or `--model-path /path/to/fp8/checkpoint` ("If the server runs in Docker,
mount that directory and use its path inside the container"). To quantize your own checkpoint: the
`cosmos3-quantization` skill (FP8 row of "Model sizes vs GPU").

## Variant 5 — TensorRT-LLM (trtllm-serve VisualGen + synchronous video API)

Reference: [`references/trt-llm.md`](references/trt-llm.md) — read it before the first command (Start the TensorRT-LLM
server; TensorRT-LLM response contract; the helper cell: `wait_for_server`, `submit_action`, `save_and_view_video`;
Prepare the checked-in AV observation; Run inverse dynamics; Inspect).

Install: "Follow the shared TensorRT-LLM setup" — the setup guide's source build at `cc63e66daf52644f1441e117bb7ba67d20566016` in `nvcr.io/nvidia/tensorrt-llm/devel:<tag>`
(`../cookbooks/cosmos3/README.md` "TensorRT-LLM Generator"; quoted in full in `references/trt-llm.md` "Skill notes", with the devel-tag cmake failure and the 17 + 33 min
the eval 2026-11-08 measured). README: "Both notebooks' current multipart fields were exercised against TensorRT-LLM source revision `cc63e66daf52644f1441e117bb7ba67d20566016`.
Follow the source pin in the shared setup." Skill note (not in the source): the install this skill runs, with or without Docker on the host, is the Route A block of `../cosmos3-generate-media/references/trt-llm.md`
§ 7.2 as ONE shell call, verbatim — the `tensorrt_llm==1.3.0rc27` wheel, its `export CUDA_HOME=…/nvidia/cu13` line (without it `trtllm-serve` exits at import: `AssertionError` in
`_find_cuda_home()`) and the pinned checkout `bca6761` as `TRTLLM_ROOT` — verified for this action route by the eval 2026-11-08 (action 60 × 9, ~16 s per request). Prerequisite (skill note, not in the source): host NVDEC/NVENC driver libraries available — `video_reference` is decoded on NVDEC, so the driver's `libnvcuvid.so.1` and `libnvidia-encode.so.1` must load where `trtllm-serve` runs
(check `/sbin/ldconfig -p | grep -E 'nvcuvid|nvidia-encode'` lists both; a container needs `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video` at creation), else the first `POST /v1/videos/sync` is HTTP 500 `Failed to load NVENC library` (eval 2026-11-08, two cells);
remedies and the HTTP 400 mismatch signature: `references/trt-llm.md` "Skill notes" § 2. Never `TRTLLM_USE_PRECOMPILED=1` (§ 7.5). Edge: "its action weights are not served by this pipeline" (`../cosmos3-generate-media/references/trt-llm.md`); the notebook serves Nano.

Serve (notebook: "run this from the TensorRT-LLM checkout. This notebook is a client; keep the server in a separate
terminal"; skill note, not in the source: with Route A, run the block from the directory § 7.2's block ran in — its first line is the skill's — or export `TRTLLM_VENV` and `TRTLLM_ROOT` yourself):

```bash
if [ -x "${TRTLLM_VENV:-$PWD/.venv-trtllm}/bin/python" ]; then export TRTLLM_VENV="${TRTLLM_VENV:-$PWD/.venv-trtllm}"; export TRTLLM_ROOT="${TRTLLM_ROOT:-$PWD/TensorRT-LLM}"; export PATH="$TRTLLM_VENV/bin:$PATH"; export CUDA_HOME="$("$TRTLLM_VENV/bin/python" -c 'import nvidia,os;print(os.path.join(nvidia.__path__[0],"cu13"))')"; fi   # skill note (not in the source): Route A only — § 7.2's venv, checkout and CUDA_HOME line again, because a new shell call has none of its exports (Read this first, 4); the venv first on PATH makes the notebook's `trtllm-serve` the venv's; without CUDA_HOME the server exits at import with AssertionError in _find_cuda_home(); TRTLLM_ROOT = the pinned checkout (holds examples/visual_gen/configs), not this skill directory; inside the devel container (no .venv-trtllm) the line is a no-op and the notebook's own TRTLLM_ROOT line applies
export TRTLLM_ROOT="${TRTLLM_ROOT:-$PWD}"
trtllm-serve nvidia/Cosmos3-Nano \
  --visual_gen_args "$TRTLLM_ROOT/examples/visual_gen/configs/cosmos3-nano-1gpu.yaml" \
  --port 8000
```

Guardrails: the notebook's request sends `"use_guardrails": True`; the README's switch is `use_guardrails: false`
(TensorRT-LLM `extra_params`), the setup guide's server-side one `TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1` — use one of
them for a run without `nvidia/Cosmos-1.0-Guardrail` access. Headless launch / teardown of `trtllm-serve` (`setsid`,
process group): `../cosmos3-generate-media/references/skill-notes.md` § 4.

Client ("Install client-side decoding dependencies in this notebook kernel if needed"):

```bash
uv venv --no-config .venv-client --python 3.12 --seed --managed-python && uv pip install --no-config --python .venv-client/bin/python requests safetensors torch imageio imageio-ffmpeg pillow ipython av==18.0.0   # skill note (not in the source): the notebook's `pip install requests safetensors torch imageio imageio-ffmpeg pillow` as a headless client venv (--no-config: the framework checkout's uv pin does not apply to it), plus ipython — the helper cell imports IPython.display; display() then prints a repr — and av for "Verify the output"; run the client cells with .venv-client/bin/python
export COSMOS3_TRTLLM_ACTION_OUTPUT_ROOT="$PWD/outputs"   # skill note (not in the source): the notebook's default is <cookbook>/outputs/notebooks/trt_llm; repeat it in the shell call that runs the client python
```

Readiness: `wait_for_server()` polls `GET <server root>/health` every 10 s for up to 1800 s. Run the configuration
cell, the helper cell, the "Prepare" cell, `wait_for_server()` + `submit_action(...)` and the "Inspect" cell as ONE
file; the configuration cell derives `ACTION_ROOT` as `<cookbook root>/cookbooks/cosmos3/generator/action` and reads
`COSMOS3_TRTLLM_ACTION_OUTPUT_ROOT` (output root), `COSMOS3_TRTLLM_BASE_URL` (`http://localhost:8000`) and
`COSMOS3_TRTLLM_API_KEY` (`tensorrt_llm`, sent as `Authorization: Bearer`):

```python
# Skill note (not in the source): insert after the configuration cell; the "Prepare" cell's
# `video_path = ACTION_ROOT / "assets" / "videos" / "av_0.mp4"` then resolves to the entry skill's vendored clip.
ACTION_ROOT = Path("../cosmos3-inference-action").resolve()
```

The request (`submit_action`): multipart `POST /v1/videos/sync` — `form = {"prompt": prompt, "format": "safetensors",
"response_format": "file", "seed": "0", "extra_params": json.dumps(extra_params, separators=(",", ":"))}`,
`files={"video_reference": (reference_path.name, reference_file, media_type)}`, `Accept: application/octet-stream`,
timeout 3600 — with the "Prepare" cell's `prompt = "You are an autonomous vehicle planning system."` and
`extra_params = {"action_mode": "inverse_dynamics", "domain_name": "av", "use_guardrails": True}`. "The `av` domain
preset selects the 60-step, 9D trajectory shape and the trained 480p/10 fps recipe." The response "is
`application/octet-stream` with named `video`, `action`, and `frame_rate` tensors" (safetensors; `submit_action` raises
if `video` or `action` is missing) and is saved as `OUTPUT_ROOT / "inverse_dynamics_av.safetensors"`;
`save_and_view_video` writes `OUTPUT_ROOT / "inverse_dynamics_av.mp4"` at the payload's `frame_rate` (`imageio`), and
the last cell prints `predicted action shape` and the first five rows. The asynchronous `POST /v1/videos` (HTTP 202,
poll `GET /v1/videos/{id}`, download `GET /v1/videos/{id}/content`) "serves the same tensor payload". FP8: the
notebook has no FP8 section.

## Verify the output

Check what you produced before reporting it — the facts the notebooks' viewer cells print (tool commands are skill
notes; the notebooks only plotted inline in Jupyter):

- Framework / vLLM-Omni / SGLang: `<output>/<name>/sample_outputs.json` → `outputs[0].content["action"]`; the
  visualize cells read it as `poses_rel = np.array(…)` (comment: `[T-1, 9] = [translation(3), rot6d(6)]`), convert
  with `pose_rel_to_abs(poses_rel, rotation_format="rot6d", pose_convention="backward_framewise", translation_scale=1.35)`
  (comment: `[T, 4, 4] camera-to-world`) and print `name, poses_rel.shape, poses_abs.shape`. For the AV record
  (`action_chunk_size: 60`; README: Autonomous vehicle "Ego pose (9D)", "60 frames @ 10FPS") expect 60 rows of 9 (skill
  note, not in the source: the comments imply 61 absolute poses). SGLang / vLLM-Omni also print `action shape:` / `dtype:` and keep `action.json`. Skill note:
  `python3 -c "import json, sys; a = json.load(open(sys.argv[1]))['outputs'][0]['content']['action']; print(len(a), len(a[0]))" <sample_outputs.json>`.
- Diffusers: `run_action` prints `wrote <path> (<shape> predicted actions)`; `view_action` prints
  `predicted actions: <steps> steps x <D>D -> <path>`, `first step: […]` and `saved <case>_trajectory.png` (same 60 × 9
  expectation for `av_inverse_0`). Skill note: `python3 -c "import json, sys; a = json.load(open(sys.argv[1])); print(len(a), len(a[0]))" <case>_action.json`;
  the PNG with Pillow (`Image.open(p).verify()`).
- TensorRT-LLM: `submit_action` prints `video: (shape) dtype` and `action: (shape) dtype`; `save_and_view_video`
  expects `video` as `[T,H,W,3]` (a leading batch dimension of 1 is squeezed) and writes the MP4 at `frame_rate`; the
  last cell prints `predicted action shape:` and `first five predicted action rows`. Skill note: with the client
  python, `python -c "import sys; from safetensors.torch import load; p = load(open(sys.argv[1], 'rb').read()); print({k: (tuple(v.shape), str(v.dtype)) for k, v in p.items()})" <file>.safetensors`;
  MP4 facts with `ffprobe` or the `av` package as in `../cosmos3-generate-media/SKILL.md` "Verify the output"
  (`ffprobe` is on neither a bare host nor in these venvs).
- Record in `result_summary.md`: backend and checkpoint, input clip, the action shape and dtype, wall-clock time and
  GPU memory observed, and for a server its launch command, time-to-ready and request latency.

## Reference files

The full notebook content for a backend (every embodiment, helper code, request contracts, FP8 routes) lives in
`references/`, migrated 1:1 — read the matching file before the first command; these files replace the notebooks. The
cookbook README is `../cosmos3-inference-action/references/action-readme.md`; the setup guide `../cookbooks/cosmos3/README.md`.

| File                             | Backend                                                             |
| -------------------------------- | ------------------------------------------------------------------- |
| `references/diffusers.md`        | Variant 1 — diffusers (`run_id_with_diffusers.ipynb`)               |
| `references/cosmos-framework.md` | Variant 2 — cosmos_framework (`run_id_with_cosmos_framework.ipynb`) |
| `references/vllm-omni.md`        | Variant 3 — vLLM-Omni (`run_id_with_vllm_omni.ipynb`)               |
| `references/sglang.md`           | Variant 4 — SGLang (`run_id_with_sglang.ipynb`)                     |
| `references/trt-llm.md`          | Variant 5 — TensorRT-LLM (`run_id_with_trt_llm.ipynb`)              |

Origin: the `run_id_with_*` notebooks of `.agents/skills/cookbooks/cosmos3/generator/action/` in the public
cosmos-framework repository; superseded by this skill (the cookbook keeps its `assets` and `finetune` subdirectories and the NIM pages).

## Related skills

| Skill                                                   | When to use                                                                        |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `../cosmos3-inference-action/SKILL.md`                  | Entry: cookbook README, action definition, the sample assets                       |
| `../cosmos3-inference-action-forward-dynamics/SKILL.md` | Forward dynamics (image + trajectory -> video)                                     |
| `../cosmos3-inference-action-policy/SKILL.md`           | Policy (image + instruction + state -> video + action chunk)                       |
| `../cosmos3-setup/SKILL.md`                             | Framework environment for Variant 2 and the viewer imports                         |
| `../cosmos3-env-troubleshoot/SKILL.md`                  | CUDA / import / server failures                                                    |
| `../cosmos3-generate-media/SKILL.md`                    | Audiovisual generation; shared server patterns and TensorRT-LLM install notes      |
| `../cosmos3-quantization/SKILL.md`                      | Quantizing your own FP8 checkpoint (the released FP8 checkpoints: FP8 lines above) |
| `../cosmos3-skill-authoring/SKILL.md`                   | Editing this skill's eval cases                                                    |
