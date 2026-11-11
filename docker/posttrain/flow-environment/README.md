
## Resolver-managed native flow runtime

The historical `install.sh` preserves the old image contract; it uses prepared
closures and is not a clean dependency-resolution claim. The two-node campaign
uses the new `packages/flow-runtime` distribution and source wheels built by
`build_source_wheels.py` from the exact upstream revisions. The derived wheels
retain upstream module bytes and licenses, use distinct distribution names, and
support only Cosmos's native flow integration. They do not advertise the upstream
SDK training/deployment CLIs or their incompatible Torch/Transformers environments.
Do not install the upstream `gr00t` or `alpamayo_r1` distributions into that runtime.

Framework Diffusers dependencies now belong to the backend: native/generator
training still requires `diffusers>=0.39.0`, reasoner-runtime pins 0.39.0, and the
physical runtime pins its existing model dependency 0.35.1. The native and
Transformers-5 split remains intact. Install `[posttrain]` with the selected runtime;
`[train]` remains the native generator dependency set. Physical data decoding uses
PyAV18, consistent with the framework's existing Python3.12 contract; historical
PyAV19 evidence retains its original identity.

Build the three framework/runtime wheels from an immutable checkout, then the two
source wheels into the same fresh wheel directory. Use uv 0.11.29 with
`--no-config`: this standalone runtime must not inherit the root workspace's
native-training NumPy override, which conflicts with GR00T's NumPy1 contract:

```bash
python docker/posttrain/flow-environment/build_source_wheels.py \
  --gr00t /path/to/Isaac-GR00T --alpamayo /path/to/alpamayo --out /new/wheels
uv --no-config venv /new/runtime --python 3.12.13
uv --no-config pip install --python /new/runtime/bin/python \
  --find-links /new/wheels --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  'cosmos-framework-flow-runtime==1.2.2' \
  'torch==2.10.0+cu128' 'torchvision==0.25.0+cu128'
uv --no-config pip check --python /new/runtime/bin/python
```

Resolve and freeze the entire installed closure, attest framework and external
source bytes, and run the affected physical-model tests before claiming this new
environment qualified. The new route is under campaign validation; it does not
retroactively relabel historical `--no-deps` installations.
