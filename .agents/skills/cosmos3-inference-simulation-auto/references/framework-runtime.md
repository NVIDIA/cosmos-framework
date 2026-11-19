# Environment and model assets

> **Skill note (not in the source):** the training-only parts of this page (the paragraph on the training allocation and output root, training memory, checkpoint memory, training components) are in the [training half of this page](../../cosmos3-post-train-simulation-auto/references/runtime.md) in the Auto post-train skill (`cosmos3-post-train-simulation-auto`).

Run from a Cosmos Framework checkout containing `docs/training_transfer_auto.md`,
`docs/training_sim_auto.md` and `cosmos_framework/scripts/train_experiment.py`.
Follow its `docs/setup.md` for the container, system packages and GPU driver.
The commands use the repository's Python 3.13 training environment. Install the
required `uv` version (`0.11.29`) and system packages from the setup guide first.
Choose CUDA 13 on a compatible driver, or the CUDA 12.8 group on older drivers
without a supported forward-compatibility setup.

> **Skill note (not in the source):** the first two commands of the block below are skill additions; the first stands in for the setup guide's two `uv` lines. `uv sync` stops with an error that starts with `error: Required uv version` on any uv other than 0.11.29, and `uv self update 0.11.29` is refused for a uv that pip or a package manager installed. When `uv --version` reports another version or no uv, the first command installs uv 0.11.29 into `$HOME/.local/bin`, where it replaces a uv already in that directory and leaves a uv elsewhere in place, and puts that directory first on `PATH` in the current shell: run it in the shell that runs `uv sync`. The second installs the FFmpeg shared libraries named in the skill's [Prerequisites](../SKILL.md#prerequisites) when the loader cache (`/sbin/ldconfig -p`) lists none.

```bash
# Skill note (not in the source): installs uv 0.11.29 into $HOME/.local/bin and puts it first on PATH in this shell when uv --version reports another version or no uv; run it in the shell that runs the uv sync line below.
uv --version 2>/dev/null | grep -q '^uv 0\.11\.29' || { curl -LsSf https://astral.sh/uv/0.11.29/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh; export PATH="$HOME/.local/bin:$PATH"; hash -r; }
# Skill note (not in the source): installs the FFmpeg shared libraries for TorchCodec when the loader cache lists none (Debian or Ubuntu, as root).
[ -n "$(/sbin/ldconfig -p 2>/dev/null | grep 'libavutil\.so')" ] || { apt-get update && apt-get install -y --no-install-recommends ffmpeg; }
uv sync --locked --python 3.13 --group cu130-train --extra train --extra guardrail
source .venv/bin/activate
export COSMOS_INTERNAL=0
```

The training extra is also needed by imported inference utilities. The separate
guardrail extra supplies `nltk` and the other dependencies used by default
inference; `--extra train` alone does not install them. Keep the checkout's
lockfile: it already specifies FlashAttention-4. Do not layer an
unrelated attention/CUTLASS upgrade on top and assume numerical parity.
Verify PyTorch, TorchCodec/FFmpeg, CUDA and attention imports. Runtime compatibility
does not establish memory capacity for a particular training or inference request.
Also run the setup guide's GPU-allocation and video-decoding checks. Preserve
required CUDA compatibility library paths; do not reset `LD_LIBRARY_PATH` merely
because the environment is a container.

> **Skill note (not in the source):** the block below is the Python part of the setup guide's [GPU and media check](../../../../docs/setup.md#verify-the-gpu-and-media-runtime); use it to check TorchCodec instead of a bare `import torchcodec`. In the `cu130-train` environment a bare import can fail with `RuntimeError: Could not load libtorchcodec`, its traceback naming `libnppicc.so.13: cannot open shared object file`, also when FFmpeg is installed: TorchCodec's CUDA build needs the NPP library, which comes in a wheel that is not on the loader path, and Framework loads it before it imports TorchCodec ([CUDA Issue](../../../../docs/setup.md#cuda-issue)). Inference uses the same loader.

```bash
# Skill note (not in the source): the Python part of the setup guide's GPU and media check. Run it from the Framework root; .venv/bin/python is the environment's Python, so no activated shell is needed.
.venv/bin/python - <<'PY'
import cv2
import torch
from cosmos_framework.utils.generator.torchcodec_dependencies import load_torchcodec

print(f"torch={torch.__version__}, CUDA build={torch.version.cuda}, OpenCV={cv2.__version__}")
torch.cuda.init()
torch.ones(1, device="cuda").add_(1)
torch.cuda.synchronize()
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"TorchCodec: {load_torchcodec().__version__}")
PY
```

## Model snapshots

| Workflow                                  | Model repository                    | Local variable     |
| ----------------------------------------- | ----------------------------------- | ------------------ |
| Curriculum base                           | `nvidia/Cosmos3-Nano`               | `NANO_HF`          |
| Transfer inference, fine-tuning, Sim base | `nvidia/Cosmos3-Nano-Transfer-Auto` | `TRANSFER_AUTO_HF` |
| Released Sim inference                    | `nvidia/Cosmos3-Nano-Sim-Auto`      | `SIM_AUTO_HF`      |

Use an authorized complete snapshot. Obtain its immutable revision from the model
distribution and set the corresponding `*_REVISION` variable before downloading;
do not invent a revision or treat `main` as immutable. For example:

```bash
# Skill note (not in the source): when TRANSFER_AUTO_REVISION is unset, takes the commit that main of the model repository names now, as the native reference does. Run it from the Framework root.
: "${TRANSFER_AUTO_REVISION:=$(.venv/bin/python -c 'from huggingface_hub import HfApi; print(HfApi().model_info("nvidia/Cosmos3-Nano-Transfer-Auto", revision="main").sha)')}"
hf download nvidia/Cosmos3-Nano-Transfer-Auto \
  --revision "${TRANSFER_AUTO_REVISION:?set the chosen snapshot revision}" \
  --local-dir "${TRANSFER_AUTO_HF:?set a local snapshot directory}"
```

Use the same pattern for Nano or Sim-Auto when needed. Accept applicable licenses
and authenticate through `hf auth login` or `HF_TOKEN` without printing the token.
A repository ID is not a local checkpoint directory. Keep config, weights and
tokenizers from the same snapshot.

```bash
# Skill note (not in the source): the same pattern written out for Sim-Auto, with the revision taken the same way when SIM_AUTO_REVISION is unset. Run it from the Framework root.
: "${SIM_AUTO_REVISION:=$(.venv/bin/python -c 'from huggingface_hub import HfApi; print(HfApi().model_info("nvidia/Cosmos3-Nano-Sim-Auto", revision="main").sha)')}"
hf download nvidia/Cosmos3-Nano-Sim-Auto \
  --revision "${SIM_AUTO_REVISION:?set the chosen snapshot revision}" \
  --local-dir "${SIM_AUTO_HF:?set a local snapshot directory}"
```

## Execution modes

Set `COSMOS_TRAINING=1` for training/config resolution and `COSMOS_TRAINING=0`
for inference. Use `cosmos_framework.scripts.train_experiment` for Auto training;
generic single-video SFT examples do not implement these sensor recipes.
Use `cosmos_framework.scripts.inference` for both Auto inference paths.

Retain the Framework revision, lockfile and snapshot revision with results so
you can reproduce the environment when comparing outputs or resuming training.
