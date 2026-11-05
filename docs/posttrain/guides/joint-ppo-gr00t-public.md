# GR00T-N1.7 Joint PPO on LIBERO-Spatial

Reuse the Python 3.12 model environment, vLLM-Omni installation, LIBERO
simulator venv, assets, manifest directory, EGL variables, and short runtime directories
from [the π0.5 guide](joint-ppo-pi05-public.md).

Install the pinned Isaac-GR00T source without changing the locked dependency
closure:

```bash
uv pip install --python "$POSTTRAIN_PY" --no-deps \
  'git+https://github.com/NVIDIA/Isaac-GR00T@51d4c89f72fda44cbf77285c6a8114b52676b8a1'
```

Download the exact public model snapshot. Keep the revision directory and
`libero_spatial` subfolder because the config binds both into model identity.
The processor also references the gated `nvidia/Cosmos-Reason2-2B` backbone;
accept its license and export a Hugging Face read token before downloading or
launching.

```bash
export GR00T_REVISION=2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21
export HF_TOKEN="<your-read-token>"
mkdir -p "$POSTTRAIN_RESOURCE_ROOT/checkpoints/$GR00T_REVISION"
uvx hf@latest download nvidia/GR00T-N1.7-LIBERO \
  --revision "$GR00T_REVISION" \
  --include 'libero_spatial/*' \
  --exclude 'libero_spatial/global_step20000/*' \
  --local-dir "$POSTTRAIN_RESOURCE_ROOT/checkpoints/$GR00T_REVISION"
```

Generate the `legacy_100` manifests expected by the GR00T config, then validate
and run:

```bash
"$POSTTRAIN_PY" - <<'PY'
import os
from pathlib import Path

from cosmos_framework.posttrain.embodied.sims.libero.protocol import fixed_manifest

root = Path(os.environ["LIBERO_MANIFEST_ROOT"])
root.mkdir(parents=True, exist_ok=True)
for split in ("train", "eval"):
    fixed_manifest(split, "libero_spatial", "legacy_100").write(
        root / f"libero_spatial-{split}.json"
    )
PY

export GR00T_DIT_SDPA_MODE=math

"$POSTTRAIN_PY" -m cosmos_framework.posttrain.entrypoints.joint_ppo \
  --config cosmos_framework/posttrain/configs/gr00t_n1d7_libero_public_4gpu.py \
  --validate-manifest

"$POSTTRAIN_PY" -m cosmos_framework.posttrain.entrypoints.joint_ppo \
  --config cosmos_framework/posttrain/configs/gr00t_n1d7_libero_public_4gpu.py
```

The recipe runs 800 steps and evaluates the policy on the 100 `legacy_100` cases before the first
step (about 5 minutes on the reference host), every 80 steps, and after the last step. For a short
check, append `-- rl.max_steps=5 rl.validation.enabled=false` to the launch command; see the
[run guide Q&A](../howto/run-posttrain-qa.md).

Do not mix a GR00T checkpoint, processor, or execution horizon with the π0.5
recipe. They share the Joint PPO controller and simulator protocol, not model
weights or observation processing.
