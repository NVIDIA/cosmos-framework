# Native action source runtime

This isolated **Linux x86_64, Python 3.12.13, CUDA 12.8** environment is shared by native
Alpamayo-R1 trajectory training and GR00T-N1.7 action training. `requirements.txt` pins
every package version, including bitsandbytes 0.49.2. Important pins are Torch
2.10.0+cu128, torchvision 0.25.0+cu128, Transformers 4.57.6, Diffusers 0.35.1, NumPy
1.26.4, safetensors 0.8.0 and PyAV 18.0.0.

From a Cosmos checkout, with a bootstrap `uv==0.11.29` executable:

```bash
python docker/posttrain/native_action/install.py \
  --uv /path/to/uv-0.11.29 --venv /path/to/venvs/cosmos-native-action
source /path/to/venvs/cosmos-native-action/native-action-env.sh
python docker/posttrain/native_action/install.py \
  --venv /path/to/venvs/cosmos-native-action --verify-only
```

The installer refuses an existing environment, builds a new relocatable managed Python
environment next to it and renames it into place only after every check passes (a failed
installation leaves nothing behind and can be rerun), installs the exact package versions from PyPI and the official PyTorch
CUDA index, and runs `uv pip check`. It clones public Isaac-GR00T and NVlabs/Alpamayo
at the revisions pinned in `cosmos_framework/posttrain/model/gr00t/adaptation.py` and
`cosmos_framework/posttrain/model/alpamayo/sft.py`, then checks that each checkout is
clean and at its pinned commit. The native loaders compare that checked-out commit before
loading a model. Activation sets `PYTHONPATH` to
those sources and this Cosmos checkout; it does not change GPU placement,
`LD_LIBRARY_PATH`, the main project dependencies or the Qwen runtime. The environment
records versions, source revisions and the checkout revision in
`native-action-runtime.json`; rerun verification after moving or changing it.

This is a **source-checkout installation**. Do not `pip install cosmos-framework`,
Isaac-GR00T or Alpamayo into it: their own dependency ranges conflict (current Cosmos
metadata requires Diffusers >=0.39, upstream GR00T pins Torch 2.9/Transformers 4.57.3,
and Alpamayo pins Torch 2.8/Transformers 4.57.1). The upstream model code is used
unmodified. The requirements pin versions, not wheel hashes.

After activation, use the Alpamayo-R1 or GR00T N1.7 `torchrun` recipe in the
[native QLoRA guide](../../../docs/post_training/native_qlora.md). Checkpoints and
datasets are separate inputs. The environment does not include the LIBERO or AlpaSim
simulators.
