---
name: cosmos3-inference-action-policy
description: >
  Run Cosmos3 action-policy inference for the DROID robot: one conditioning
  observation (the 640x540 wrist + two-exterior-camera concat_view) plus a task
  instruction in, a 16-step action chunk out, optionally with a 17-frame
  predicted rollout video, through the Cosmos3-Nano-Policy-DROID /
  Cosmos3-Edge-Policy-DROID checkpoints on diffusers in-process, vLLM-Omni,
  SGLang, TensorRT-Edge-LLM, or the Cosmos Framework policy server with a
  RoboLab client. Use when the user asks "predict the robot's next actions from
  this observation", "run the Cosmos3 DROID policy", "serve
  Cosmos3-Nano-Policy-DROID / Cosmos3-Edge-Policy-DROID", "roll out a policy
  with vLLM-Omni / SGLang", "build the policy engines with TensorRT-Edge-LLM"
  or "start the Cosmos3 policy server for RoboLab". Not for forward dynamics
  (video from an action trajectory: ../cosmos3-inference-action-forward-dynamics),
  inverse dynamics (actions from a video: ../cosmos3-inference-action-inverse-dynamics),
  audiovisual generation (../cosmos3-generate-media) or post-training the
  policy. The entry point for the whole Action row is ../cosmos3-inference-action.
---

# Cosmos3 Action Policy Inference (DROID)

The policy block of the Cosmos3 Action row: the four policy notebooks and the policy-server guide of the action
cookbook, migrated 1:1 into `references/`. "Policy (`policy`) — predict future observations and action trajectories from
a start image, task instruction, and state" (cookbook README). Input: the first frame of the three DROID camera clips
composed into one 640x540 `concat_view` image plus the instruction `"Pick up the object and place it in the target
container."`; output: a 16-step action chunk (plus a 17-frame, 15 fps rollout video on three of the five paths: Variants 1–3). Everything below quotes the sources; text that is not in a source is marked "skill note".

## Read this first

Before the first command of any variant:

1. Read the variant's reference file in full (linked under its heading, with the sections the run needs). The variant sections below only
   summarise the notebooks; the reference files are the notebooks migrated 1:1 (every markdown and code cell, including the Jupyter
   `display(...)` calls). Do not look for the original notebooks; the reference files replace them.
2. Check the GPU with `nvidia-smi`. Variants 1–4 run on one GPU (Variant 1 `CUDA_VISIBLE_DEVICES=0`, Variants 2–3 `--gpus '"device=0"'`,
   Variant 4 "on the target GPU"); Variant 5 runs as the guide launches it (`--runtime nvidia`, no device restriction; the docs page adds a
   two-rank `--cfg-parallel` launch). The driver's CUDA major version picks Variant 1's torch backend (`COSMOS3_TORCH_BACKEND=cu130` for
   13.x, `cu128` for 12.x — setup guide `../cookbooks/cosmos3/README.md`, "CUDA driver and the `cuXXX` backend").
3. Where the assets live: this block has no assets of its own. The row's vendored samples are in `../cosmos3-inference-action/assets/` (same
   layout as the cookbook's assets directory; none of them is read by the policy notebooks). Every policy variant reads the DROID LeRobot
   example, which is NOT vendored (three git-LFS clips of 4.0–5.3 MB) and stays in `../cookbooks/cosmos3/generator/action/assets/droid_lerobot_example/`
   — materialise it first ("Bundled assets" below). Skill note: the notebooks resolved their assets next to themselves through `find_repo_root()`,
   which walks up from the working directory to the first directory that has a cookbooks/cosmos3 subdirectory and reads the action cookbook's
   assets under it; run the cells from this skill directory inside the `.agents/skills` tree and that directory is `.agents/skills`.
4. `COSMOS3_REPO` (Variants 1–3) must be "the framework checkout that contains the `cosmos_framework/` Python package—not this `cosmos`
   cookbook repository, a model directory, or the `packages/cosmos3` model sources" (vLLM-Omni / SGLang notebooks). Skill note: when this
   skill directory sits inside a cosmos-framework checkout, that checkout's root is `../../..` from here.
5. Run each notebook's Python cells as ONE script, in notebook order, with one interpreter: the later cells use variables of the earlier ones
   (`policy_image_path`, `policy_prompt`, `DROID_ASSET_ROOT`, …). Skill note: the cells import `IPython.display` and call `display(...)`; outside
   Jupyter, install `ipython` into the client environment (the Variant 1 install list already carries `ipykernel`) so the import succeeds — `display()` then prints a representation.
6. Verify what you produced before reporting it ("Verify the output" below) and record the facts in `result_summary.md`. After a failure, go
   back to the reference section that covers the failing step before improvising a fix.

## When to use this skill

- Use to predict a DROID action chunk (Variants 1–3 also render a 17-frame rollout video) from one observation and an
  instruction with a released Cosmos3 policy checkpoint, on any of the five paths below
- Use to serve `Cosmos3-Nano-Policy-DROID` / `Cosmos3-Edge-Policy-DROID` over HTTP (vLLM-Omni, SGLang) or to a
  simulator client (Cosmos Framework policy server + RoboLab)
- Use to build the Edge policy TensorRT engines and run `cosmos3_policy_inference` (TensorRT-Edge-LLM)
- Forward dynamics (video from a given action trajectory) → `../cosmos3-inference-action-forward-dynamics/SKILL.md`;
  inverse dynamics (actions from a video) → `../cosmos3-inference-action-inverse-dynamics/SKILL.md`; the row's README,
  Action Definition and shared assets → `../cosmos3-inference-action/SKILL.md`
- Audiovisual generation is `../cosmos3-generate-media/SKILL.md`; environment problems → `../cosmos3-env-troubleshoot/SKILL.md`

## Bundled assets

| Asset                                                                        | Path                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Used by                                                 |
| ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| DROID LeRobot example (`meta` and `data` parquet + JSON; three camera clips) | `../cookbooks/cosmos3/generator/action/assets/droid_lerobot_example/` — kept in the cookbook; the clips `../cookbooks/cosmos3/generator/action/assets/droid_lerobot_example/videos/observation.image.wrist_image_left/chunk-000/file-000.mp4`, `../cookbooks/cosmos3/generator/action/assets/droid_lerobot_example/videos/observation.image.exterior_image_1_left/chunk-000/file-000.mp4` and `../cookbooks/cosmos3/generator/action/assets/droid_lerobot_example/videos/observation.image.exterior_image_2_left/chunk-000/file-000.mp4` are git-LFS objects, NOT vendored | every variant (the notebooks read only the three clips) |
| DROID normalisation stats `droid_lerobot_stats.json`                         | `../../../cosmos_framework/data/generator/action/normalizer_stats/droid_lerobot_stats.json` — the framework checkout's `cosmos_framework/data/generator/action/normalizer_stats/droid_lerobot_stats.json` (`COSMOS3_REPO`)                                                                                                                                                                                                                                                                                                                                                 | Variant 1 `view_policy` (denormalisation)               |
| Row samples (AV, UMI, Bridge, Fractal, RoboMIND Franka, hand-pose, prompts)  | `../cosmos3-inference-action/assets/` (vendored by the entry skill)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | not read by the policy notebooks                        |

Skill note (not in the sources) — materialising the DROID clips. In a checkout without git-lfs the three clips are
132-byte LFS pointers. Either `git lfs pull --include='.agents/skills/cookbooks/cosmos3/generator/action/assets/droid_lerobot_example/**'`
in a cosmos-framework checkout with git-lfs installed, or download each object from the public repository at the pinned commit and
check it against the pointer's `size` and `oid sha256` (embedded below, read from the pointers at bbd4a67) — run this from this skill directory as ONE shell call:

```bash
# Skill note (not in the source). COSMOS_ROOT = the directory the notebooks' find_repo_root() resolves (the first ancestor of the
# working directory that contains cookbooks/cosmos3; from this skill directory inside the .agents/skills tree that is .agents/skills).
# The loop creates the tree under COSMOS_ROOT when it is missing; a read-only checkout: export COSMOS_ROOT=<writable dir> first (see below).
set -eu; COSMOS_ROOT="${COSMOS_ROOT:-$(cd "$(pwd -P)/.." && pwd)}"; SHA=bbd4a67c3cf54cc6699464ad1259f3a0f6537da9   # commit the values below were read from
while read -r cam size oid; do   # <camera> <pointer size> <pointer oid sha256>
  f="cookbooks/cosmos3/generator/action/assets/droid_lerobot_example/videos/$cam/chunk-000/file-000.mp4"
  if [ ! -s "$COSMOS_ROOT/$f" ] || [ "$(head -c 7 "$COSMOS_ROOT/$f" | tr -d '\0')" = version ]; then   # missing, or still a pointer: fetch the object
    mkdir -p "$(dirname "$COSMOS_ROOT/$f")" && : > "$COSMOS_ROOT/$f.tmp" || { echo "$COSMOS_ROOT is not writable: export COSMOS_ROOT=<writable dir> and re-run"; exit 1; }
    curl -fsSL -o "$COSMOS_ROOT/$f.tmp" "https://media.githubusercontent.com/media/NVIDIA/cosmos-framework/$SHA/.agents/skills/$f"
    [ "$(wc -c < "$COSMOS_ROOT/$f.tmp" | tr -d ' ')" = "$size" ] && [ "$(sha256sum "$COSMOS_ROOT/$f.tmp" | cut -d' ' -f1)" = "$oid" ] \
      || { echo "download of $f does not match its pointer"; exit 1; }
    mv "$COSMOS_ROOT/$f.tmp" "$COSMOS_ROOT/$f"
  fi
  ls -l "$COSMOS_ROOT/$f"   # expected sizes: wrist 5,337,025 B, exterior_1 4,694,851 B, exterior_2 3,982,548 B
done <<'EOF'
observation.image.wrist_image_left 5337025 e9d641240f9efc344924c31715a44be1f66207f667dd63c292bbd943dfa816d0
observation.image.exterior_image_1_left 4694851 6aac49551ff9b9fc7b8e9899df4fc050f5cd31164ed4ee3c014f332bcef3b9f1
observation.image.exterior_image_2_left 3982548 34f6e5dae8591809324f1df9bf166c6e304bd2cbdad643595a991f641cbb4fd7
EOF
```

Read-only checkout (skill note): `export COSMOS_ROOT=<writable directory>` (for example the `.agents/skills` directory of your fresh framework
clone, which holds the same pointers) and run the block — it recreates `cookbooks/cosmos3/generator/action/assets/droid_lerobot_example/videos/<camera>/chunk-000/`
there and verifies every download against the embedded values — then copy this skill directory directly under that directory and run the cells
from the copy (`find_repo_root()` picks the nearest ancestor with a cookbooks/cosmos3 subdirectory). The same applies when this skill directory is
delivered on its own, outside a checkout: the block then creates the tree beside it (the launcher of `../cosmos3-skill-authoring/references/running-the-eval.md`
§ 2 names the skill inside the mounted checkout, `/workspace/skill-source/.agents/skills/<skill>`, so `..` is that tree). The row's other non-vendored
examples (listed in `../cosmos3-inference-action/SKILL.md`) are not read by the policy notebooks.

## Checkpoints and GPUs, as the sources state them

| Checkpoint                         | What the sources say                                                                                                                                                                                                                                                                                                                          | GPUs, as the sources run it                                                                                                                         |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `nvidia/Cosmos3-Nano-Policy-DROID` | "the post-trained … checkpoint, conditioned on the DROID sample" (diffusers.md § 1); the only entry of Variant 1's `MODEL_IDS`; default of Variants 3 and 5, the first of Variant 2's two launch blocks; vLLM-Omni: "also requires an up-to-date NumPy installation … `pip install --upgrade numpy`" in the environment that starts the model | one GPU: `CUDA_VISIBLE_DEVICES=0` (Variant 1), `--gpus '"device=0"'` (2, 3); Variant 5: one process, or two ranks with `--cfg-parallel` (docs page) |
| `nvidia/Cosmos3-Edge-Policy-DROID` | Variant 2: "The Edge checkpoint receives the structured JSON prompt format it was trained on"; Variant 3: "use `nvidia/Cosmos3-Edge-Policy-DROID` as `model-path`"; Variant 4: "On-device TensorRT path"; Variant 5: `--format-prompt-as-json True --guidance-interval 960 1001`                                                              | one GPU as above; Variant 4 builds the engines "on the target GPU"                                                                                  |

Skill note: the sources state no checkpoint sizes or memory figures; none was measured for this skill. FP8: none of the
five policy sources has a quantized-checkpoint route; the row's FP8 pointer ("For Edge/Nano/Super static FP8
checkpoints, see the quantization cookbook and the **Quantized Checkpoints** section in the vLLM-Omni notebooks") is
the first paragraph of the README, `../cosmos3-inference-action/references/action-readme.md`.

Action spaces differ between the sources — keep each as written: Variant 1 writes `result.action` as "predicted
actions, model-normalized space" and `view_policy` denormalises it (`method="quantile"`) into `actions[:, :9]` poses plus
the `actions[:, 9]` gripper channel (the README's DROID row: "End-effector pose (9D) + gripper grasp state (1D)");
Variants 2 and 3 expect "a `[16, 8]` DROID joint-position action chunk" ("7 joints + 1 gripper"); Variant 4 takes a
`--state` of "eight comma-separated model-space values" and writes `action.json` as `[batch, chunk, action_dimension]`.

Guardrails: "Generator requires the Guardrail. Request access to the gated nvidia/Cosmos-1.0-Guardrail HF repository
before running these examples." Variant 1: `COSMOS3_DIFFUSERS_GUARDRAILS=false`; Variant 2: `--no-guardrails` (in the
notebook's launch) and `"guardrails": False` per request; Variant 3: `"guardrails": False` per request — the launch has no
server-wide switch (skill note: for the same `lmsysorg/sglang:dev` image the generate-media eval runs needed
`-e SGLANG_DISABLE_COSMOS3_GUARDRAILS=1` on `docker run`, otherwise the server loaded the gated guardrail at startup and
exited with `GatedRepoError 401`, `../cosmos3-generate-media/references/skill-notes.md` § 3; unverified for the policy checkpoint).

## Variant 1 — diffusers (in-process, Cosmos3-Nano-Policy-DROID)

Reference: [`references/diffusers.md`](references/diffusers.md) — read it before the first command (§ 1 Prerequisites,
§ 2 Configure Paths and Environment, § 3 Install, § 5 Verify, § 7 helpers, "Policy: DROID Pick and Place").

"Use a Linux machine with NVIDIA GPU access, model access on Hugging Face, and either `uvx hf@latest auth login` or
`HF_TOKEN` set." "Point `COSMOS3_REPO` at your framework checkout (it defaults to `packages/cosmos3` beside this repo)."
"uv version: use `uv 0.11.29`, matching the version pinned by Cosmos Framework." Headless servers: "if you see an error
like `libxcb.so.1: cannot open shared object file` … `apt-get install -y libxcb1 libgl1 libglib2.0-0`". The notebook's
§ 2 exports (replace the placeholders; skill note: `COSMOS3_ACTION_OUTPUT_ROOT` is the notebook's output root — it
defaults to `outputs/notebooks/diffusers` under the cookbook's action directory; point it at your output directory):

```bash
export COSMOS3_DIFFUSERS_ACTION_VENV=/path/to/.venv-cosmos3-diffusers-action
export COSMOS3_REPO=/path/to/packages/cosmos3
export COSMOS3_TORCH_BACKEND=cu130
export HF_HOME=/path/to/large/huggingface/cache
export UV_LINK_MODE=copy
export CUDA_VISIBLE_DEVICES=0
```

Install (§ 3, the notebook's `%%bash` cell, verbatim — run it as ONE shell call with the exports above in the same call;
skill note: the `ipykernel install` lines register the Jupyter kernel and are harmless outside Jupyter):

```bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
uv venv "$COSMOS3_DIFFUSERS_ACTION_VENV" --python 3.13 --seed --managed-python --allow-existing
source "$COSMOS3_DIFFUSERS_ACTION_VENV/bin/activate"

# The LeRobot readers, pose helpers, and trajectory plots add parquet support and plotting on top
# of the diffusers stack.
uv pip install --torch-backend="$COSMOS3_TORCH_BACKEND" \
  "diffusers @ git+https://github.com/huggingface/diffusers.git" \
  "lerobot @ https://github.com/huggingface/lerobot/archive/1a4316c6845330bc552fb982dbc44bdb4f66f2f1.tar.gz" \
  accelerate \
  av \
  cosmos_guardrail \
  datasets \
  draccus \
  huggingface_hub \
  imageio \
  imageio-ffmpeg \
  ipykernel \
  loguru \
  matplotlib \
  mujoco \
  pandas \
  pyarrow \
  scipy \
  torch \
  torchvision \
  transformers

# The LeRobot video decoder needs a torchcodec build matching the Torch backend, which is
# published only on the PyTorch index. Keep that index scoped to this requirement so the rest of
# the tree still resolves against PyPI.
uv pip install --torch-backend="$COSMOS3_TORCH_BACKEND" \
  --extra-index-url "https://download.pytorch.org/whl/$COSMOS3_TORCH_BACKEND" \
  torchcodec

"$COSMOS3_DIFFUSERS_ACTION_VENV/bin/python" -m ipykernel install --user \
  --name cosmos3-diffusers-action \
  --display-name "Cosmos3 Diffusers Action (Python 3.13)"

echo
echo "Installed dependencies into: $COSMOS3_DIFFUSERS_ACTION_VENV"
echo "Next: switch this notebook kernel to: Cosmos3 Diffusers Action (Python 3.13)"
echo "After switching kernels, run the Restore Environment cell below, then continue with Verify."
```

Skill note: this skill directory sits inside a cosmos-framework checkout whose `pyproject.toml` pins `[tool.uv] required-version = "==0.11.29"`,
and uv applies that pin from any cwd under the checkout — the notebook's "uv 0.11.29" prerequisite satisfies it (the generate-media
skill handles the pin differently for its standalone venv: `../cosmos3-generate-media/SKILL.md`, "Read this first" item 3).

Run: with the exports of § 2 set in the same shell call, execute § 2's Python cell, § 5 (Verify GPU and Python
Environment), § 7 (helpers) and the two cells of "Policy: DROID Pick and Place", in that order, as ONE file with
`"$COSMOS3_DIFFUSERS_ACTION_VENV/bin/python"` from this skill directory (§ 4's "restore" cell repeats § 2 for a kernel
switch and is not needed in a script). Do not compose the pipeline call from the prose. The run cells:

```python
droid_policy_output = run_policy("droid_policy")
```

```python
view_policy("droid_policy")
```

What § 7 does: `build_concat_frame` composes the first frame of each clip onto a 640x540 canvas (wrist on top, the two
exterior views below); `run_policy` loads the checkpoint in `torch.bfloat16` with `enable_safety_checker=GUARDRAILS`, swaps
in `UniPCMultistepScheduler` (`flow_shift=5.0`) and calls the pipeline with `CosmosActionCondition(mode="policy",
chunk_size=16, domain_name="droid_lerobot", resolution_tier=480, image=image, view_point="concat_view")`, `fps=15`, 30
steps, `guidance_scale=1.0`, `use_system_prompt=False`, seed 0 — "it passes no `raw_actions`. `height`, `width`, and
`num_frames` stay unset". Outputs under `<COSMOS3_ACTION_OUTPUT_ROOT>/droid_policy/`: `droid_policy_input.png`,
`droid_policy.mp4` (15 fps), `droid_policy_action.json` ("model-normalized space"); `view_policy` adds the two plots.

## Variant 2 — vLLM-Omni (Docker server, Nano or Edge; one-shot `/v1/videos`)

Reference: [`references/vllm-omni.md`](references/vllm-omni.md) — read it before the first command (Prerequisites,
"Start vLLM-Omni for One-Shot Policy Rollout", "Prepare a DROID Policy Input", "Run One-Shot Policy Inference").

"Choose either the Cosmos3-Nano or Cosmos3-Edge DROID checkpoint and start it in a terminal from the cookbook root
(`.agents/skills`). Before running it, set `COSMOS3_REPO` to the root of a local checkout of the **cosmos-framework**
repository" (`export COSMOS3_REPO=/path/to/cosmos-framework`). "The server container mounts this checkout at
`/workspace/cosmos-framework`." "They publish the container's port `8000` to host port `8001`, matching the default
`http://localhost:8001` endpoint." The Edge launch (the Nano block differs only in the model name,
`nvidia/Cosmos3-Nano-Policy-DROID`):

```bash
docker rm -f cosmos3-vllm-omni-policy-notebook 2>/dev/null || true

docker run -d --name cosmos3-vllm-omni-policy-notebook \
  --runtime nvidia --gpus '"device=0"' \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e PYTHONPATH=/workspace/cosmos-framework \
  -v "$COSMOS3_REPO:/workspace/cosmos-framework" \
  -p 8001:8000 --ipc=host \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Edge-Policy-DROID \
    --no-guardrails \
    --omni \
    --model-class-name Cosmos3OmniDiffusersPipeline \
    --allowed-local-media-path / \
    --port 8000 \
    --init-timeout 1800

# Wait until this returns model metadata before running the inference cells.
curl http://localhost:8001/v1/models
```

Readiness: the notebook's `check_vllm_server()` polls `GET /v1/models` every 10 s for up to 600 s and returns the first
model id ("Check `docker logs -f cosmos3-vllm-omni-policy-notebook`" on timeout). Skill notes: the launch passes no
`HF_TOKEN` and mounts no Hugging Face cache (the setup guide's Prerequisites require Hugging Face access to the gated
checkpoints); when done, keep the logs before removing the container — `docker logs cosmos3-vllm-omni-policy-notebook > <OUT>/vllm-server.log 2>&1; docker stop cosmos3-vllm-omni-policy-notebook; docker rm cosmos3-vllm-omni-policy-notebook`.

Client: run the notebook's three Python cells (paths; "Prepare a DROID Policy Input"; "Run One-Shot Policy Inference
Through `/v1/videos`") as ONE script from this skill directory; the "Inspect" cell is the viewer. The cells name
`pip install imageio-ffmpeg` and `pip install requests`; skill note: they also import `numpy`, `PIL` (Pillow) and `IPython`
— e.g. `python3 -m venv .venv-policy-client && .venv-policy-client/bin/pip install numpy pillow requests imageio-ffmpeg ipython`,
then run with that venv's `python`. `COSMOS3_VLLM_OUTPUT_ROOT` sets the output root. `submit_policy_video` sends multipart
`POST /v1/videos` with `input_reference` = the 640x540 composite (`policy_input.png`) and the form

```json
{"prompt": "<instruction; for Cosmos3-Edge-Policy-DROID the make_edge_policy_prompt() structured JSON>",
 "num_frames": 17, "fps": 15, "size": "736x544", "num_inference_steps": 30, "guidance_scale": 1.0,
 "flow_shift": 5.0, "seed": 0,
 "extra_params": "{\"action_mode\": \"policy\", \"domain_name\": \"droid_lerobot\", \"raw_action_dim\": 8, \"action_chunk_size\": 16, \"image_size\": 480, \"guardrails\": false}"}
```

(skill note: `"size"` is `closest_action_size(540, 640)` → `(736, 544)` from the notebook's `ACTION_VIDEO_RES_SIZE_INFO` table; the Edge
prompt is `make_edge_policy_prompt(policy_prompt, 736, 544, num_frames=17, fps=15)`). The cell polls `GET /v1/videos/{id}` every 2 s
until `completed`, requires `final["action"]["data"]` of shape `(16, 8)` and finite, writes `action.json` and `sample_outputs.json`, then
downloads `GET /v1/videos/{id}/content` into `policy_rollout.mp4` ("video content endpoint returned an empty body" leaves it out). Outputs under `<COSMOS3_VLLM_OUTPUT_ROOT>/action_policy_droid/video_api/`.

## Variant 3 — SGLang (Docker server, Nano by default; `/v1/actions/generations` + forward-dynamics `/v1/videos`)

Reference: [`references/sglang.md`](references/sglang.md) — read it before the first command (Overview, "Set up HF
Cache", "Start SGLang Server", "Prepare a DROID Policy Input", "Run Policy Inference and Render Rollout").

"It first sends a `POST /v1/actions/generations` request with a first frame and instruction to predict a DROID policy
action vector. It can then send that action vector to the forward-dynamics `POST /v1/videos` endpoint to render a rollout
video." "The notebook uses this same server for both the policy action request and the rollout-video request by
default." Set `COSMOS3_REPO` as in Variant 2 ("Start the server in a terminal from the cookbook root (`.agents/skills`)"),
then the notebook's two launch blocks:

```bash
export SGLANG_HF_CACHE="${SGLANG_HF_CACHE:-$HOME/.cache/sglang-huggingface}"
mkdir -p "$SGLANG_HF_CACHE"
chmod 777 "$SGLANG_HF_CACHE"
```

```bash
docker rm -f cosmos3-sglang-policy-notebook 2>/dev/null || true

docker run -d --init --name cosmos3-sglang-policy-notebook \
  --runtime nvidia --gpus '"device=0"' \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e PYTHONPATH=/workspace/cosmos-framework \
  -v $SGLANG_HF_CACHE:/root/.cache/huggingface \
  -v "$PWD:/workspace" \
  -p 30000:30000 --ipc=host \
  lmsysorg/sglang:dev \
  sglang serve \
    --model-path nvidia/Cosmos3-Nano-Policy-DROID \
    --host 0.0.0.0

# Wait until this returns model metadata before running the inference cells.
curl http://localhost:30000/v1/models
```

"When serving Cosmos3 Edge, use `nvidia/Cosmos3-Edge-Policy-DROID` as `model-path`." Logs: `docker logs -f cosmos3-sglang-policy-notebook`.
Readiness: `check_sglang_server()` polls `GET /v1/models` every 10 s for up to 600 s. Skill notes (observations, not
changes to the command): this launch mounts `$PWD` at `/workspace` and sets `PYTHONPATH=/workspace/cosmos-framework`, so
the framework package is visible to the server only when the directory you launch from contains the checkout as a
`cosmos-framework` directory (the command does not reference `COSMOS3_REPO`); it passes no `HF_TOKEN` and no guardrail
switch (see "Guardrails" above). Teardown as in Variant 2 with the name `cosmos3-sglang-policy-notebook`.

Client: the notebook's three Python cells as ONE script from this skill directory (same client packages as Variant 2;
`COSMOS3_SGLANG_OUTPUT_ROOT` sets the output root, `COSMOS3_SGLANG_FD_BASE_URL` a separate forward-dynamics server). First
request, `submit_policy_action`: multipart `POST /v1/actions/generations`, `input_reference` = the 640x540 composite, form
`prompt` = the plain instruction ("The SGLang server applies ActionPromptJsonFormatter to the incoming prompt itself"),
`num_frames` 17, `fps` 15, `height` 544 / `width` 736 ("`/v1/actions/generations` reads height/width options; it does not
parse a `"size"` string"), `num_inference_steps` 30, `guidance_scale` 1.0, `flow_shift` 5.0, `seed` 0, `extra_params` =
`{"action_mode": "policy", "domain_name": "droid_lerobot", "raw_action_dim": 8, "action_view_point": "concat_view", "action_cinematography_framing": CONCAT_VIEW_DESCRIPTION, "guardrails": False}`;
the response's `data[0].action` must be a finite `(16, 8)` chunk → `policy_response.json`, `action.json`, `policy_action_for_fd.json`, `sample_outputs.json`.
Second request, `submit_forward_dynamics_rollout`: `POST /v1/videos` with `extra_params` `action_mode` `forward_dynamics`, the chunk as `action`,
`num_frames` = chunk length + 1 (17), `size` `640x528` (skill note: `(640 // 16) * 16` x `(540 // 16) * 16`), `flow_shift` 10.0; poll
`GET /v1/videos/{id}` every 2 s, download `/content` → `policy_rollout.mp4`. Outputs under `<COSMOS3_SGLANG_OUTPUT_ROOT>/action_policy_droid/video_api/`.

## Variant 4 — TensorRT-Edge-LLM (Cosmos3-Edge-Policy-DROID; direct engine builder + `cosmos3_policy_inference`)

Reference: [`references/trt-edge-llm.md`](references/trt-edge-llm.md) — read it before the first command (intro, § 1
Setup paths, § 2 Build the policy engines, § 3 Prepare a DROID observation, § 4 Run the policy); setup guide
`../cookbooks/cosmos3/README.md`, section "TensorRT-Edge-LLM".

Setup guide: "On-device **TensorRT** path … builds engines on the target (Jetson Thor/Orin, DRIVE, DGX Spark, or an x86
developer build). Policy on v0.11.0 builds engines with the direct builder." "Use TensorRT-Edge-LLM **v0.11.0**. Complete
the upstream Installation guide first (Python package on the host; C++ runtime on the device). Policy examples need
`-DBUILD_EXPERIMENTAL_MODELS=ON` so `cosmos3_policy_inference` exists, and `-DENABLE_CUTE_DSL=ALL` for `tensorrt-edgellm-build`."
Notebook: a v0.11.0 checkout built with those flags "and a Python environment with `pip install -e ".[builder]"`"; "Run this
on the target GPU". Skill note: the C++ runtime build needs a CUDA toolkit and TensorRT on the build host — without them
(the eval's task container has no CUDA toolkit) build inside a CUDA development container.

§ 1 defaults (the setup cell exports them): `EDGELLM_ROOT=~/TensorRT-Edge-LLM` ("It must contain
`build/experimental_models/cosmos3/examples/` after a CMake build"), `ENGINE_DIR=~/tensorrt-edgellm-workspace/Cosmos3-Edge/engines`,
`POLICY_CHECKPOINT=nvidia/Cosmos3-Edge-Policy-DROID` (a local directory or the Hugging Face repo id), `POLICY_PROMPT`, `POLICY_STATE=0,0,0,0,0,0,0,0`
("a zero placeholder; replace it with the robot's current state"), `POLICY_ENGINE_DIR=$ENGINE_DIR/policy`, `WORK_DIR=$POLICY_ENGINE_DIR/cookbook_work`, `OUTPUT_JSON=$WORK_DIR/action.json`.
Run § 1's Python cell first (it also asserts the DROID example exists), then § 2's `%%bash` cell verbatim as ONE call
with those variables exported ("`--components` keeps the build on the policy towers … `--max-input-len 512` matches the
v0.11.0 policy prompt budget. For DROID, the builder sets `content_height=540` and `content_width=640` from
`checkpoint.json`"). Its core (quoted as text — do not run this excerpt on its own: `MODEL_DIR` is set by the cell's own
if/else, which the excerpt omits) is `cd "$EDGELLM_ROOT"`, `mkdir -p "$POLICY_ENGINE_DIR"`, then `tensorrt-edgellm-build --model-dir "$MODEL_DIR"
--engine-dir "$POLICY_ENGINE_DIR" --components und-prefill,gen,vae-encoder --max-input-len 512 --max-batch-size 1`,
followed by the cell's contract check (`content_height == 540`, `content_width == 640`, `attention_position_ids` in the
UND profile, `embed_tokens.safetensors` and the tokenizer JSON under `text_tokenizer`). § 3 composes the 640x540 `concat_view`
from the three DROID clips ("Same geometry as run_policy_with_vllm_omni.ipynb / run_policy_with_sglang.ipynb") into
`$WORK_DIR/droid_policy_first_frame.png` and exports `OBSERVATION_IMAGE`. § 4, verbatim:

```bash
set -euo pipefail
: "${EDGELLM_ROOT:?run the setup cell first}"
: "${POLICY_ENGINE_DIR:?run the setup cell first}"
: "${OBSERVATION_IMAGE:?run the DROID observation cell first}"
: "${POLICY_PROMPT:?run the setup cell first}"
: "${POLICY_STATE:?run the setup cell first}"
: "${OUTPUT_JSON:?run the setup cell first}"

cd "$EDGELLM_ROOT"
./build/experimental_models/cosmos3/examples/cosmos3_policy_inference \
  --engineDir "$POLICY_ENGINE_DIR" \
  --image "$OBSERVATION_IMAGE" \
  --prompt "$POLICY_PROMPT" \
  --state "$POLICY_STATE" \
  --viewPoint concat_view \
  --output "$OUTPUT_JSON"

echo "Wrote $OUTPUT_JSON"
python3 - <<'PY'
import json
import os
from pathlib import Path
data = json.loads(Path(os.environ["OUTPUT_JSON"]).read_text())
print(json.dumps(data, indent=2)[:4000])
PY
```

"Pass `--viewPoint concat_view` so the policy prompt matches that multiview layout (the CLI default is `ego_view`)."
"For an ordered frame list, replace `--image` with `--video frame_00.png,frame_01.png,frame_02.png`. The current policy
conditions on the most recent frame. Optional: `--steps` (denoise steps) and `--seed`." Output: `action.json`
(`[batch, chunk, action_dimension]`, README) — skill note: copy it from `$OUTPUT_JSON` into your output directory.

## Variant 5 — Cosmos Framework policy server + RoboLab client (guide; no eval case)

Reference: [`references/cosmos-framework.md`](references/cosmos-framework.md) — the cookbook guide
`run_policy_with_cosmos_framework.md`, copied verbatim. The framework docs page `../../../docs/action_policy_droid_server.md`
is a near-duplicate that adds "Two-rank CFG parallelism" (`OMP_NUM_THREADS=16 torchrun --nproc-per-node=2 … --cfg-parallel`);
it is linked, not copied.

"Each of these policy models is served by a policy **Server** that streams actions to a **Client** driving a simulated
or real robot. This example uses `RoboLab`, a simulation benchmark for task-generalist policies, as the client. Start the
server first, then connect the client." Server: clone `https://github.com/NVIDIA/cosmos-framework.git`, build
`docker build --build-arg INSTALL_APEX=0 -t cosmos-framework:latest .`, launch the container with `HF_TOKEN`, `--net host`
and `uv sync --all-extras --group=cu130-torch213-train --group=policy-server` (the guide's `docker run` block), then
inside it:

```
python -m cosmos_framework.scripts.action_policy_server_robolab \
  --port 8000
```

for Nano; for Edge the same module with `--checkpoint-path nvidia/Cosmos3-Edge-Policy-DROID --port 8000 --format-prompt-as-json True --guidance-interval 960 1001`
("The guidance interval applies classifier-free guidance only to denoising timesteps in the inclusive range `[960, 1001]`.
Omit `--guidance-interval` to apply guidance at every denoising step."). Client: clone `https://github.com/NVlabs/RoboLab.git`,
`./docker/build_docker.sh latest`, `./docker/run_docker.sh latest`, then `python policies/cosmos3/run.py --task BananaInBowlTask`
(viewer window) or `--num-envs 10 --headless`. Skill note: this variant needs a second container and a simulator and has
no bundled input, so it has no eval case; the NIM's policy client for the same DROID sample is the cookbook's
`../cookbooks/cosmos3/nim/action.md` ("Do not copy its payload" into the vLLM-Omni flow).

## Verify the output

What the notebooks' viewer cells check (quote them in `result_summary.md`): Variant 1 `view_policy` prints "generated:
`<case>.mp4` (… KB)", "predicted actions: `<steps>` steps x `<dim>`D" (the shape it measured; skill note: 16 steps = `chunk_size`, 10D = the
README's DROID row and the 10-D `droid_lerobot_stats.json`) and "first denormalized step: …", then saves the trajectory and gripper plots; Variants 2 and 3 "Inspect" print `action array: (16, 8) float32` and the first five rows and re-encode the
rollout into `policy_rollout_preview.mp4` with `imageio_ffmpeg` (`libx264`, `-crf 28`); Variant 4's § 4 cell prints
`action.json` (`[batch, chunk, action_dimension]`).

Skill note — tool commands (`ffprobe` is absent on a bare host and in the eval's task container; `imageio-ffmpeg`, which
every client environment above installs, ships an `ffmpeg` binary):

- Frame count / size / fps of a rollout: `FF="$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"; "$FF" -i <file>.mp4 -map 0:v:0 -f null - 2>&1 | grep -E 'Stream #0:0|frame='`
  (17 frames at 15 fps per the requests; the frame size is unverified — expected to follow the request size, `736x544` for Variant 2 and `640x528` for Variant 3) —
  or, in the Variant 1 venv, `python -c "import av,sys; c=av.open(sys.argv[1]); s=c.streams.video[0]; print(s.codec_context.name, s.width, s.height, sum(1 for _ in c.decode(s)), float(s.average_rate))" <file>.mp4`.
- Action chunk: `python -c "import json,sys,numpy as np; d=json.load(open(sys.argv[1])); a=np.asarray((d.get('values') or d.get('data')) if isinstance(d,dict) else d, dtype=float); print(a.shape, np.isfinite(a).all())" <action.json>`
  (Variant 1: `(16, 10)` normalised — inferred from `chunk_size=16` and the 10-D stats file, not printed by the notebook; Variants 2–3: `(16, 8)`, the notebooks' own check — Variant 3's file is keyed `values` or `data`, as its `action_values_from_response` reads it; Variant 4: the file's `[batch, chunk, action_dimension]`).
- Record in `result_summary.md`: checkpoint, server command or script, time-to-ready, latency or generation time, the shapes above and the output paths; never report a file you did not inspect.

## Reference files

The complete notebook content (environment, helpers, requests, viewer cells) lives in `references/`, migrated 1:1
from the cookbook notebooks — read the matching file before the first command (see the top). Do not look for the
notebooks; these files replace them. Their relative links are the notebooks' own: a link to "assets" means the cookbook's
`../cookbooks/cosmos3/generator/action/assets/` (kept); the links to the forward/inverse-dynamics notebooks were
re-pointed at the sibling block skills.

| File                                                      | Variant                                                                                                                                       |
| --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `references/diffusers.md`                                 | Variant 1 — `run_policy_with_diffusers.ipynb`                                                                                                 |
| `references/vllm-omni.md`                                 | Variant 2 — `run_policy_with_vllm_omni.ipynb`                                                                                                 |
| `references/sglang.md`                                    | Variant 3 — `run_policy_with_sglang.ipynb`                                                                                                    |
| `references/trt-edge-llm.md`                              | Variant 4 — `run_policy_with_trt_edge_llm.ipynb`                                                                                              |
| `references/cosmos-framework.md`                          | Variant 5 — `run_policy_with_cosmos_framework.md` (guide); near-duplicate with CFG parallelism: `../../../docs/action_policy_droid_server.md` |
| `../cosmos3-inference-action/references/action-readme.md` | The row's README (Overview, Action Definition, per-backend quickstarts, Post-Train pointer) — entry skill, not part of this block             |
| `../cookbooks/cosmos3/README.md`                          | The shared environment setup guide (Diffusers, vLLM-Omni, TensorRT-Edge-LLM, Cosmos Framework sections)                                       |

Origin: the `run_policy_with_*` notebooks and `run_policy_with_cosmos_framework.md` of `.agents/skills/cookbooks/cosmos3/generator/action`
in the public cosmos-framework repository (`main` bbd4a67); superseded by this skill. Eval cases: `eval/eval.config` (one per notebook; the guide has none).

## Related skills

| Skill                                                   | When to use                                                                                                                                                                |
| ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `../cosmos3-inference-action/SKILL.md`                  | Entry point of the Action row: README, Action Definition, vendored sample assets                                                                                           |
| `../cosmos3-inference-action-forward-dynamics/SKILL.md` | Forward dynamics: future video from a start image plus an action trajectory                                                                                                |
| `../cosmos3-inference-action-inverse-dynamics/SKILL.md` | Inverse dynamics: ego-motion / end-effector trajectories from a video                                                                                                      |
| `../cosmos3-setup/SKILL.md`                             | Framework checkout and venv (`COSMOS3_REPO`, Variant 5's `uv sync`)                                                                                                        |
| `../cosmos3-env-troubleshoot/SKILL.md`                  | CUDA / import / container failures                                                                                                                                         |
| `../cosmos3-generate-media/SKILL.md`                    | Same server images (vLLM-Omni, SGLang) for audiovisual generation; headless-server pattern and guardrail switches in `../cosmos3-generate-media/references/skill-notes.md` |
| `../cosmos3-inference-quantization/SKILL.md`            | Quantizing your own checkpoint (no policy source documents an FP8 route; the row's FP8 pointer is in the entry skill)                                                      |
| `../cosmos3-skill-authoring/SKILL.md`                   | Editing this skill's eval cases and running the eval                                                                                                                       |
