# Cosmos3-Edge Reasoner with TensorRT-Edge-LLM

This notebook runs **Cosmos3-Edge** multimodal reasoning with
[TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) (v0.11.0):
export the checkpoint to ONNX, build TensorRT engines, then run the C++
`llm_inference` binary.

It is **not** the datacenter [TensorRT-LLM](./tensorrt-llm.md)
OpenAI-compatible server path.

1. Points at a TensorRT-Edge-LLM checkout and a local workspace for ONNX/engines.
2. Exports `nvidia/Cosmos3-Edge` with `--task reasoning`.
3. Builds the LLM and visual engines.
4. Runs an image caption request against the cookbook asset `robot_153.jpg`.

Complete the upstream
[Installation](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/installation.html)
first (Python export on an x86 host; C++ runtime on Jetson Thor/Orin, DRIVE,
DGX Spark, or an x86 developer build). The authoritative command sequence is
the upstream
[Cosmos3-Edge VLA guide](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/examples/vla/cosmos3.html).

> **Skill note (not in the source):** this file is the cookbook notebook `run_with_trt_edge_llm.ipynb` (retired into this skill from `.agents/skills/cookbooks/cosmos3/reasoner/run_with_trt_edge_llm.ipynb`, cosmos-framework `main` bbd4a67) migrated 1:1: every markdown cell and every code cell in the notebook's order, code cells fenced (`python`; `%%bash` cells as `bash` with the magic line removed), outputs omitted (the committed notebook has none). Paths and links are the notebook's own; SKILL.md says where this skill keeps the assets and what replaces the notebook's working directory.

## 1. Setup paths

Set `EDGELLM_ROOT` to your TensorRT-Edge-LLM checkout (the directory that
contains `build/` after you compile the runtime). Optional: override
`ONNX_DIR` / `ENGINE_DIR`.

> **Skill note (not in the source):** from this skill directory `find_repo_root()` raises `FileNotFoundError` (no `cookbooks/cosmos3` tree here); replace the assignments that call it with `Path.cwd()` and `Path.cwd() / "assets"` (SKILL.md Read this first 3).

```python
from pathlib import Path
import os
import json


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / ".agents" / "skills" / "cookbooks" / "cosmos3").is_dir():
            return candidate
        if (candidate / "cookbooks" / "cosmos3").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the cosmos-framework repository root")


def cookbook_root(repo: Path) -> Path:
    for relative in (Path(".agents/skills/cookbooks/cosmos3"), Path("cookbooks/cosmos3")):
        candidate = repo / relative
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(repo / ".agents/skills/cookbooks/cosmos3")


COSMOS_ROOT = find_repo_root(Path.cwd().resolve())
REASONER_ASSETS = cookbook_root(COSMOS_ROOT) / "reasoner" / "assets"
assert REASONER_ASSETS.exists(), REASONER_ASSETS

EDGELLM_ROOT = Path(
    os.environ.get("EDGELLM_ROOT", Path.home() / "TensorRT-Edge-LLM")
).expanduser().resolve()
ONNX_DIR = Path(
    os.environ.get(
        "ONNX_DIR",
        Path.home() / "tensorrt-edgellm-workspace" / "Cosmos3-Edge" / "onnx",
    )
).expanduser().resolve()
ENGINE_DIR = Path(
    os.environ.get(
        "ENGINE_DIR",
        Path.home() / "tensorrt-edgellm-workspace" / "Cosmos3-Edge" / "engines",
    )
).expanduser().resolve()
REASONING_CHECKPOINT = os.environ.get("REASONING_CHECKPOINT", "nvidia/Cosmos3-Edge")
WORK_DIR = ENGINE_DIR / "reasoning" / "cookbook_work"
WORK_DIR.mkdir(parents=True, exist_ok=True)

os.environ["COSMOS_ROOT"] = str(COSMOS_ROOT)
os.environ["REASONER_ASSETS"] = str(REASONER_ASSETS)
os.environ["EDGELLM_ROOT"] = str(EDGELLM_ROOT)
os.environ["ONNX_DIR"] = str(ONNX_DIR)
os.environ["ENGINE_DIR"] = str(ENGINE_DIR)
os.environ["REASONING_CHECKPOINT"] = REASONING_CHECKPOINT
os.environ["WORK_DIR"] = str(WORK_DIR)

print("cosmos root:", COSMOS_ROOT)
print("EDGELLM_ROOT:", EDGELLM_ROOT, "(exists:", EDGELLM_ROOT.exists(), ")")
print("ONNX_DIR:", ONNX_DIR)
print("ENGINE_DIR:", ENGINE_DIR)
print("REASONING_CHECKPOINT:", REASONING_CHECKPOINT)
```

## 2. Export the Reasoner on CPU

Run this on the export host (after `pip install -e .` in the TensorRT-Edge-LLM
checkout). It downloads `nvidia/Cosmos3-Edge` on first use.

> **Skill note (not in the source):** at v0.11.0 `pip install -e .` installs only `cuda-python` and `numpy`, and the
> `[export]` extra lacks `nvidia-modelopt`, which the visual export imports (`ModuleNotFoundError: No module named 'modelopt'`
> after the LLM half is written). Host install that runs this cell unchanged:
> `git clone --branch v0.11.0 https://github.com/NVIDIA/TensorRT-Edge-LLM.git "$EDGELLM_ROOT" && cd "$EDGELLM_ROOT" && python3 -m venv venv && . venv/bin/activate && uv pip install --python venv/bin/python -e ".[tools]"` (`uv pip`, parallel cached downloads — the notebook's `pip install` of the same extras exceeded a 10-minute tool limit on the CUDA wheels in the eval; uv 0.11.29 is installed per SKILL.md Read this first 5)
> (the upstream Installation page tracks `main`; the v0.11.0 text is `docs/source/user_guide/getting_started/installation.md` in the checkout).

```bash
set -euo pipefail
: "${REASONING_CHECKPOINT:?run the setup cell first}"
: "${ONNX_DIR:?run the setup cell first}"

mkdir -p "$ONNX_DIR/reasoning"
tensorrt-edgellm-export \
  "$REASONING_CHECKPOINT" \
  "$ONNX_DIR/reasoning" \
  --task reasoning

echo "Exported reasoning ONNX under $ONNX_DIR/reasoning"
```

## 3. Build the LLM and visual engines

Run these from the TensorRT-Edge-LLM tree after the C++ runtime is built
(`./build/...` binaries exist). Use the platform CMake flags from the upstream
installation guide; reasoning does **not** require
`-DBUILD_EXPERIMENTAL_MODELS=ON`.

```bash
set -euo pipefail
: "${EDGELLM_ROOT:?run the setup cell first}"
: "${ONNX_DIR:?run the setup cell first}"
: "${ENGINE_DIR:?run the setup cell first}"

cd "$EDGELLM_ROOT"
mkdir -p "$ENGINE_DIR/reasoning"

./build/examples/llm/llm_build \
  --onnxDir "$ONNX_DIR/reasoning/llm" \
  --engineDir "$ENGINE_DIR/reasoning" \
  --maxInputLen 2048 \
  --maxKVCacheCapacity 4096

./build/examples/multimodal/visual_build \
  --onnxDir "$ONNX_DIR/reasoning/visual" \
  --engineDir "$ENGINE_DIR/reasoning"

echo "Built reasoning engines under $ENGINE_DIR/reasoning"
```

## 4. Write an image reasoning request

The C++ `llm_inference` CLI accepts the standard
[image message format](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/format/input-format.html#message-fields).
This cell writes `input.json` pointing at the cookbook asset.

```python
image_path = (REASONER_ASSETS / "robot_153.jpg").resolve()
assert image_path.exists(), image_path

input_path = WORK_DIR / "input.json"
output_path = WORK_DIR / "output.json"

payload = {
    "batch_size": 1,
    "max_generate_length": 256,
    "requests": [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": "Caption the image in detail."},
                    ],
                }
            ]
        }
    ],
}
input_path.write_text(json.dumps(payload, indent=2) + "\n")
os.environ["INPUT_JSON"] = str(input_path)
os.environ["OUTPUT_JSON"] = str(output_path)

print("wrote", input_path)
print("image:", image_path)
```

## 5. Run reasoning

Inference reads the engines under `ENGINE_DIR/reasoning` and writes
`output.json`.

```bash
set -euo pipefail
: "${EDGELLM_ROOT:?run the setup cell first}"
: "${ENGINE_DIR:?run the setup cell first}"
: "${INPUT_JSON:?run the input-json cell first}"
: "${OUTPUT_JSON:?run the input-json cell first}"

cd "$EDGELLM_ROOT"
./build/examples/llm/llm_inference \
  --engineDir "$ENGINE_DIR/reasoning" \
  --multimodalEngineDir "$ENGINE_DIR/reasoning" \
  --inputFile "$INPUT_JSON" \
  --outputFile "$OUTPUT_JSON"

echo "Wrote $OUTPUT_JSON"
python3 - <<'PY'
import os
from pathlib import Path
out = Path(os.environ["OUTPUT_JSON"])
print(out.read_text()[:4000])
PY
```

## 6. Next steps

- Change the prompt or image path in the input-json cell and re-run inference.
- For **Cosmos3-Edge-Policy-DROID** on TensorRT-Edge-LLM, see
  [`../generator/action/run_policy_with_trt_edge_llm.ipynb`](../../cosmos3-inference-action-policy/references/trt-edge-llm.md)
  (build the runtime with `-DBUILD_EXPERIMENTAL_MODELS=ON`).
- Prefer an OpenAI-compatible server on workstation GPUs? Use
  [`run_with_vllm.ipynb`](./vllm.md) or
  [`run_with_tensorrt_llm.ipynb`](./tensorrt-llm.md) (Nano/Super).
- Python-first Edge Reasoner without TensorRT engines:
  [`run_with_transformers.ipynb`](./transformers.md).
