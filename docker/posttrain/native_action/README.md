# Native action source runtime

This isolated **Linux x86_64, Python 3.12.13, CUDA 12.8** profile is shared by native
Alpamayo-R1 trajectory training and GR00T-N1.7 action training. It preserves the
version closure recorded by `runtime-flow-857f597-attempt02`, removes its four
private Cosmos/profile distributions, and adds bitsandbytes 0.49.2. Important pins
are Torch 2.10.0+cu128, torchvision 0.25.0+cu128, Transformers 4.57.6, Diffusers
0.35.1, NumPy 1.26.4, safetensors 0.8.0 and PyAV 18.0.0.

From the selected Cosmos checkout, with a bootstrap `uv==0.11.29` executable:

```bash
python docker/posttrain/native_action/install.py \
  --uv /path/to/uv-0.11.29 --venv /durable/venvs/cosmos-native-action
source /durable/venvs/cosmos-native-action/native-action-env.sh
python docker/posttrain/native_action/install.py \
  --venv /durable/venvs/cosmos-native-action --verify-only
```

The installer refuses an existing environment, creates a new managed Python
environment, installs exact package versions from PyPI and the official PyTorch
CUDA index, and runs `uv pip check`. It clones public Isaac-GR00T at
`51d4c89f72fda44cbf77285c6a8114b52676b8a1` and NVlabs/Alpamayo at
`11a0e01c13a5622377c45ee37d653351453ec43b`, then checks clean revisions and the same
source hashes required by the native loaders. Activation sets `PYTHONPATH` to
those sources and this Cosmos checkout. It does not modify Slurm routing, GPU
placement, `LD_LIBRARY_PATH`, the main project dependencies or Qwen's runtime.
The environment records versions, source hashes and the checkout revision in
`native-action-runtime.json`; rerun verification after moving or changing it.

This is intentionally a **source-checkout installation**. Do not subsequently
`pip install cosmos-framework`, Isaac-GR00T or Alpamayo into it and let their
independent dependency ranges resolve: current Cosmos metadata requires
Diffusers >=0.39, upstream GR00T pins Torch 2.9/Transformers 4.57.3, and Alpamayo
pins Torch 2.8/Transformers 4.57.1. Those metadata contracts do not describe this
measured compatibility profile. Public model Python code is unchanged and its
bytes are verified; this does not claim every upstream feature works with the
selected dependency overrides. The version-pinned requirements are not a
wheel-byte lock. A fresh installation still needs import/kernel/task checks
before its artifacts can inherit qualification from a retained container.

After activation use the normal [Alpamayo](../../../docs/post_training/native_qlora_alpamayo.md)
or [GR00T](../../../docs/post_training/native_qlora_gr00t.md) `torchrun` recipe.
Checkpoints and datasets are separate inputs. The profile contains no LIBERO or
AlpaSim simulator and makes no rollout, serving-latency or distributed claim.
Dense export additionally records `COSMOS_TRAINING_CODE_REVISION` and
`COSMOS_TRAINING_WHEEL_SHA256`. Build a real wheel from the exact clean checkout
used by the activated runtime; do not use a version label or invent a wheel hash.
The wheel supplies build provenance while the profile continues importing the
verified checkout. Building it does not install Cosmos dependencies or change
`PYTHONPATH`.

From a fresh clean checkout at the desired commit, after activating its source
profile, choose a **new external** build directory. The separate build environment
below pins the existing Hatchling backend and leaves the training environment
unchanged. Use the same `uv` 0.11.29 executable as the runtime installer:

```bash
set -euo pipefail
export ACTION_SOURCE="$(git rev-parse --show-toplevel)"
export ACTION_BUILD=/durable/new-native-action-build
ACTION_UV=/path/to/uv-0.11.29
ACTION_PYTHON=/durable/venvs/cosmos-native-action/bin/python

test -z "$(git -C "$ACTION_SOURCE" status --porcelain --untracked-files=all)"
mkdir "$ACTION_BUILD"
"$ACTION_UV" --no-config venv "$ACTION_BUILD/tools" --python 3.12.13 --managed-python
"$ACTION_UV" --no-config pip install --python "$ACTION_BUILD/tools/bin/python" hatchling==1.29.0
"$ACTION_UV" --no-config build --wheel --no-build-isolation \
  --python "$ACTION_BUILD/tools/bin/python" \
  --out-dir "$ACTION_BUILD/wheels" "$ACTION_SOURCE"
```

Verify the active import path and every package member against both the checkout
and its committed bytes. This also checks the package inventory, respecting the
existing wheel exclusion of `*_test.py`; it does not infer equivalence from the
package version. The output records the actual whole-wheel digest:

```bash
"$ACTION_PYTHON" - <<'PY' > "$ACTION_BUILD/export-provenance.json"
import fnmatch, hashlib, importlib.util, json, os, pathlib, subprocess, zipfile
source = pathlib.Path(os.environ["ACTION_SOURCE"]).resolve()
build = pathlib.Path(os.environ["ACTION_BUILD"]).resolve()
def git(*args):
    return subprocess.check_output(["git", "-C", str(source), *args])
assert not git("status", "--porcelain", "--untracked-files=all").strip(), "checkout changed"
revision = git("rev-parse", "HEAD").decode().strip()
origin = pathlib.Path(importlib.util.find_spec("cosmos_framework").origin).resolve()
assert origin == source / "cosmos_framework/__init__.py", "wrong active framework source"
tracked = git("ls-files", "-z", "cosmos_framework").decode().split("\0")
expected = {p for p in tracked if p and not fnmatch.fnmatch(pathlib.Path(p).name, "*_test.py")}
wheels = list((build / "wheels").glob("*.whl"))
assert len(wheels) == 1, "expected one wheel in the new build directory"
wheel = wheels[0]
with zipfile.ZipFile(wheel) as archive:
    members = [p for p in archive.namelist() if p.startswith("cosmos_framework/") and not p.endswith("/")]
    assert len(members) == len(set(members)) and set(members) == expected, "package inventory differs"
    for name in members:
        assert archive.read(name) == (source / name).read_bytes() == git("show", f"{revision}:{name}"), name
record = {"code_revision": revision, "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
          "wheel": str(wheel), "active_source": str(source), "verified_package_files": len(members)}
print(json.dumps(record, indent=2, sort_keys=True))
PY
export COSMOS_TRAINING_CODE_REVISION="$("$ACTION_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["code_revision"])' "$ACTION_BUILD/export-provenance.json")"
export COSMOS_TRAINING_WHEEL_SHA256="$("$ACTION_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["wheel_sha256"])' "$ACTION_BUILD/export-provenance.json")"
```

Retain the wheel and provenance JSON with the exported artifact. Repeat this
procedure after changing the checkout; an older wheel digest must not label new
source. This verifies build provenance, not accelerator/task qualification of a
new build. Keep the current source profile activated for the normal native export
command; no wheel installation or dependency resolution is required in that runtime.
