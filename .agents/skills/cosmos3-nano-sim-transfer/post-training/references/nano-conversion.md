# Convert public Nano weights for Stage 0

Use the pinned [Cosmos3-Nano model](https://huggingface.co/nvidia/Cosmos3-Nano/tree/e59a53c25979a090fa8706c9acc0c254a6e89b92)
as the initializer for Stage 0. The conversion produces regular model
weights in the native DCP layout. Stage 0 initializes EMA from its first update;
subsequent stages use your selected EMA checkpoints.

## Prepare the converter

Use the clean Framework checkout and environment from
[setup](../../shared/setup.md), installed with that checkout's
lockfile. No separate historical checkout is needed. The helper checks the
converter's SHA-256 and import location, then records the actual Framework
commit and package versions in the conversion receipt. Keep the checkout
unchanged until conversion finishes.

```bash
set -euo pipefail
: "${FRAMEWORK_DIR:?Set the absolute Framework checkout root}"
: "${SIM_TRANSFER_SKILL:?Set the absolute skill bundle location}"
cd "$FRAMEWORK_DIR"
source .venv/bin/activate
```

Preserve the [native-library environment](../../shared/environment.md).
Before downloading model weights, check the converter and installed runtime:

```bash
python "$SIM_TRANSFER_SKILL/post-training/scripts/prepare_nano_dcp.py" check-env \
  --framework-dir "$FRAMEWORK_DIR"
```

## Download and convert

Set `NANO_HF_DIR` and `NANO_DCP_DIR` to separate new absolute paths. The helper
creates both directories. It downloads the Nano model and tokenizer from
Hugging Face and runs Framework's native
`cosmos_framework.scripts.convert_model_to_dcp` with `--checkpoint-path` and
`-o`. It stages processor metadata from the same pinned Nano revision in an
isolated cache because the pinned architecture otherwise resolves that metadata
through `main`.

```bash
export NANO_HF_DIR="${NANO_HF_DIR:?Set a new absolute path for the Nano HF download}"
export NANO_DCP_DIR="${NANO_DCP_DIR:?Set a new absolute path for the converted DCP}"
python "$SIM_TRANSFER_SKILL/post-training/scripts/prepare_nano_dcp.py" convert \
  --framework-dir "$FRAMEWORK_DIR" \
  --source-dir "$NANO_HF_DIR" \
  --output-dir "$NANO_DCP_DIR"
export BASE_CKPT_ROOT="$NANO_DCP_DIR"
```

The auxiliary video VAE comes from
[`Wan-AI/Wan2.2-TI2V-5B`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/blob/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth)
at revision `921dbaf3f1674a56f47e83fb80a34bac8a8f203e`. Bind that downloaded
file for training:

```bash
export VIDEO_VAE_PATH="$(python - <<'PY'
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

print(hf_hub_download(
    repo_id="Wan-AI/Wan2.2-TI2V-5B",
    filename="Wan2.2_VAE.pth",
    revision="921dbaf3f1674a56f47e83fb80a34bac8a8f203e",
    cache_dir=Path(os.environ["NANO_HF_DIR"]) / ".cache/conversion-hub",
))
PY
)"
```

Conversion runs on CPU. Allocate 160 GiB of host memory and disk space for the
Nano source, auxiliary cache and converted output. The DCP root contains
`model/.metadata`, model shards and `model/config.json`. The native converter
also derives the audio checkpoint and configuration under
`NANO_HF_DIR/sound_tokenizer/` for Stage 0. Its text tokenizer is available at
`NANO_HF_DIR/text_tokenizer/`.

Pass `BASE_CKPT_ROOT` to [Stage 0](local-training.md). Keep the entire DCP root;
training initializes from that root, while model-only loaders use its `model/`
child. Use the native full-state checkpoint path for later stage transitions
and same-run resume.

Receipt verification checks provenance and file integrity, not training
compatibility. Complete the Stage 0 short training and save/resume checks before
a longer run. Existing conversion receipts remain verifiable without the old
converter checkout.
