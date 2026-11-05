# Run Posttrain in a container

This guide starts one standard NVIDIA PyTorch container, installs Cosmos
Framework plus the locked Posttrain runtime, and runs the public recipes
without Slurm-specific launchers. Commands run from the repository root inside
the container.

The commands use one four-GPU host as a concrete, runtime-qualified showcase.
Four GPUs are not an algorithm or API requirement: users can adapt the recipe
topology, model parallelism, and batch sizes to their available hardware while
preserving the validation constraints described below.

## Reference host used for qualification

The commands in this guide were qualified on one Linux NVIDIA GB200 node with
four B200 GPUs. For a standalone host, the equivalent reference envelope is:

- a Docker-compatible NVIDIA container runtime configured so `--gpus all`
  exposes all four GPUs; with Docker, this means Docker Engine plus NVIDIA
  Container Toolkit;
- a host NVIDIA driver compatible with the container's CUDA 13.x user-space
  stack;
- `nvcr.io/nvidia/pytorch:26.06-py3` as the base container;
- 64 GiB of container shared memory and unlimited locked memory, as set by the
  command below;
- plan for about 150 GiB of free disk for the repository, Python environments,
  Hugging Face cache, model checkpoints, simulator assets, and initial outputs;
  longer runs need additional checkpoint and output capacity.

This is a tested reference configuration, not a minimum hardware or host
version requirement. Other GPU hosts, driver versions, container engines, and
storage layouts may work; adapt the image and architecture-specific dependency
group where necessary. Before continuing, verify that the chosen runtime can
see the expected GPUs, for example:

```bash
nvidia-smi
docker run --rm --gpus all nvcr.io/nvidia/pytorch:26.06-py3 nvidia-smi
```

## 1. Start the container

On the host:

```bash
export COSMOS_REPO=/absolute/path/to/cosmos-framework
export COSMOS_RUNS=/absolute/path/to/cosmos-runs
export COSMOS_DATA=/absolute/path/to/posttrain-data
mkdir -p "$COSMOS_RUNS" "$COSMOS_DATA"

docker run --rm -it \
  --gpus all \
  --ipc=host \
  --shm-size=64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$COSMOS_REPO:/workspace/cosmos-framework" \
  -v "$COSMOS_RUNS:/outputs" \
  -v "$COSMOS_DATA:/data:ro" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -w /workspace/cosmos-framework \
  nvcr.io/nvidia/pytorch:26.06-py3 bash
```

## 2. Install the environment

Inside the container, first install the Framework base, then add the locked
Posttrain runtime to the same active environment. Posttrain intentionally owns
its Torch 2.11, vLLM 0.23, CUDA, and NATTEN compatibility matrix.

```bash
apt-get update
apt-get install -y --no-install-recommends curl ffmpeg git git-lfs
python3 -m pip install --quiet uv==0.11.29

export UV_PROJECT_ENVIRONMENT="$PWD/.venv-posttrain"
uv sync --frozen --no-default-groups
source .venv-posttrain/bin/activate
uv sync --project docker/posttrain/environment \
  --frozen --group cu130-runtime --active --inexact

export POSTTRAIN_PY="$PWD/.venv-posttrain/bin/python"
export PYTHONPATH="$PWD"
export IMAGINAIRE_OUTPUT_ROOT=/outputs
export WANDB_MODE=disabled
export LD_LIBRARY_PATH=
export RAY_TMPDIR=/tmp/posttrain-ray
mkdir -p "$RAY_TMPDIR"
```

Verify the runtime:

```bash
"$POSTTRAIN_PY" - <<'PY'
import natten, ray, torch, vllm
import cosmos_framework.posttrain
assert torch.cuda.device_count() == 4
print(torch.__version__, torch.version.cuda)
print(vllm.__version__, ray.__version__, natten.__version__)
PY
```

The GPU-count assertion above checks the showcased setup. When adapting the
examples, update that assertion and keep these coupled settings consistent:

- DPO `torchrun --nproc-per-node`, model data-parallel width, and visible GPUs;
- `grid.ranks`/role shapes, rollout tensor parallelism, and model parallelism;
- environment count, global minibatch, microbatch, and rank divisibility for
  Joint PPO.

Run `--validate-manifest` after every topology change and before model loading.
Use the [configuration reference](../reference/config.md) and
[batching guide](batching.md) when changing the showcased resource shape.

Qwen models and public datasets below are read directly from Hugging Face.
Set `HF_TOKEN` only when the selected repository is gated. Cosmos generator
checkpoints follow the download/conversion flow in
[Framework training](../../training.md#step-2--prepare-checkpoint).

### Run length

The launch commands below run each recipe to its full `rl.max_steps`. On the reference host that is
about four minutes for OPD, 30–45 minutes for NFT and DPO, two to three hours for GRPO and PPO, and
longer for Joint PPO. To check that a recipe works end to end, append `rl.max_steps=5` to the
overrides after `--`. For Joint PPO, also append `rl.validation.enabled=false`, which skips the
policy evaluation that otherwise runs before the first step and after the last one. The
[run guide Q&A](run-posttrain-qa.md) lists the measured time per recipe, the checkpoint intervals,
and fixes for the errors first runs hit.

## 3. GRPO

This uses Qwen2.5-3B and GSM8K directly from Hugging Face. Policy FSDP and
rollout TP are colocated across all four GPUs.

```bash
"$POSTTRAIN_PY" -m cosmos_framework.posttrain.entrypoints.grpo \
  --config cosmos_framework/posttrain/configs/grpo_qwen3b_gsm8k_public_4gpu.py \
  --validate-manifest

"$POSTTRAIN_PY" -m cosmos_framework.posttrain.entrypoints.grpo \
  --config cosmos_framework/posttrain/configs/grpo_qwen3b_gsm8k_public_4gpu.py -- \
  job.name=grpo_public_4gpu
```

## 4. DPO

The public recipe creates deterministic image preference pairs in memory. It
needs the public Wan2.2 VAE and starts from Cosmos3-Nano: convert the Nano
checkpoint to DCP with the Framework converter and point `DPO_BASE_CHECKPOINT` at
it. Section 7 uses the same DCP, so this conversion runs once.

```bash
mkdir -p artifacts/wan22_vae
uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
  --local-dir artifacts/wan22_vae
export DPO_TOKENIZER_PATH="$PWD/artifacts/wan22_vae/Wan2.2_VAE.pth"

mkdir -p artifacts/checkpoints
"$POSTTRAIN_PY" -m cosmos_framework.scripts.convert_model_to_dcp \
  --checkpoint-path Cosmos3-Nano \
  -o artifacts/checkpoints/Cosmos3-Nano
export DPO_BASE_CHECKPOINT="$PWD/artifacts/checkpoints/Cosmos3-Nano"

"$POSTTRAIN_PY" -m torch.distributed.run \
  --standalone --nproc-per-node=4 \
  -m cosmos_framework.posttrain.entrypoints.dpo \
  --config cosmos_framework/posttrain/configs/dpo_t2i_public.py -- \
  job.name=dpo_public_4gpu
```

For real preference data, keep the same model and entrypoint but replace the
synthetic source with a JSONL dataset:

```bash
"$POSTTRAIN_PY" -m torch.distributed.run \
  --standalone --nproc-per-node=4 \
  -m cosmos_framework.posttrain.entrypoints.dpo \
  --config cosmos_framework/posttrain/configs/dpo_t2i_public.py -- \
  rl.data.synthetic_pairs=0 \
  rl.data.jsonl_path=/data/preferences/train.jsonl \
  rl.data.shuffle=true \
  job.name=dpo_public_4gpu
```

The JSONL and tar-backed preference contracts are described in the
[DPO data contract](../guides/dpo.md#3-data-contract).

## 5. PPO

The public profile uses Qwen3-0.6B from Hugging Face: two colocated
policy/rollout GPUs, one frozen-reference GPU, and one critic GPU.

```bash
"$POSTTRAIN_PY" -m cosmos_framework.posttrain.entrypoints.ppo \
  --config cosmos_framework/posttrain/configs/ppo_qwen06b_public_4gpu.py \
  --validate-manifest

"$POSTTRAIN_PY" -m cosmos_framework.posttrain.entrypoints.ppo \
  --config cosmos_framework/posttrain/configs/ppo_qwen06b_public_4gpu.py -- \
  job.name=ppo_public_4gpu
```

## 6. OPD

The public profile uses a two-GPU Qwen2.5-1.5B policy, a one-GPU rollout, and
a one-GPU Qwen2.5-3B teacher. Cross-device policy publication uses NIXL. The
CUDA 13 NIXL wheels below are available for both `x86_64` and `aarch64`; uv
uses the qualified dependency group on `aarch64` and selects the matching
standalone wheels on `x86_64`.

```bash
case "$(uname -m)" in
  aarch64)
    uv sync --project docker/posttrain/environment \
      --frozen --group nixl-cu130-aarch64 --active --inexact
    ;;
  x86_64)
    uv pip install --python "$POSTTRAIN_PY" --no-deps \
      "nixl-cu13==1.3.2" "nixl==1.3.2"
    ;;
  *)
    echo "Unsupported architecture: $(uname -m)" >&2
    exit 1
    ;;
esac

export POSTTRAIN_NIXL_CUDA_FLAVOR=cu130
export UCX_TLS=cuda_copy,cuda_ipc,sm,tcp,self
export UCX_NET_DEVICES=lo
export NCCL_NET_PLUGIN=none
export UCX_MODULE_DIR="$($POSTTRAIN_PY -c \
  'import sysconfig; print(sysconfig.get_path("purelib") + "/nixl_cu13.libs/ucx")')"

"$POSTTRAIN_PY" -m cosmos_framework.posttrain.utils.nixl_preflight
"$POSTTRAIN_PY" -m cosmos_framework.posttrain.entrypoints.opd \
  --config cosmos_framework/posttrain/configs/opd_qwen1p5b_countdown_public_4gpu.py -- \
  job.name=opd_public_4gpu
```

For multi-node NIXL, replace `UCX_NET_DEVICES=lo` with the qualified NICs and
run an actor-to-actor transfer probe before training.

## 7. NFT

The public NFT profile uses Cosmos3-Nano, four-rank FSDP, four prompt rows, and
an HTTP image-reward service. It uses the VAE and the Nano DCP from section 4;
skip the converter command below if you already ran it there.

```bash
mkdir -p artifacts/checkpoints
"$POSTTRAIN_PY" -m cosmos_framework.scripts.convert_model_to_dcp \
  --checkpoint-path Cosmos3-Nano \
  -o artifacts/checkpoints/Cosmos3-Nano

mkdir -p artifacts/nft
printf '%s\n' \
  '{"prompt":"A red cube on a white table."}' \
  '{"prompt":"A blue sphere beside a wooden block."}' \
  '{"prompt":"A robot gripper above a green bowl."}' \
  '{"prompt":"A yellow mug on a clean workbench."}' \
  > artifacts/nft/prompts.jsonl

export NFT_BASE_CHECKPOINT="$PWD/artifacts/checkpoints/Cosmos3-Nano"
export NFT_TOKENIZER_PATH="$PWD/artifacts/wan22_vae/Wan2.2_VAE.pth"
export NFT_T2I_PROMPTS_JSONL="$PWD/artifacts/nft/prompts.jsonl"
export NFT_T2I_PROMPTS_SHA256="$(sha256sum "$NFT_T2I_PROMPTS_JSONL" | cut -d' ' -f1)"
export NFT_REWARD_REVISION=toy-image-hash-v1
export REMOTE_REWARD_ENQUEUE_URL=http://127.0.0.1:8765/enqueue
export REMOTE_REWARD_FETCH_URL=http://127.0.0.1:8765/fetch
export REMOTE_REWARD_TOKEN=local-toy-token
```

The enqueue endpoint accepts an octet stream containing one JSON metadata line
followed by an NPY image and returns `{"uuid":"..."}`. The fetch endpoint
accepts form fields `uuid` and `type` and returns
`{"scores":{"toy_image_hash":[<score>]}}`. A production run should use an
immutable, authenticated reward deployment; a local deterministic toy service
is sufficient only for connectivity testing. Create and start that local toy
service in the same container:

```bash
cat > artifacts/nft/toy_reward_server.py <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from urllib.parse import parse_qs


class RewardHandler(BaseHTTPRequestHandler):
    """Serve the minimal synchronous NFT image-reward protocol."""

    protocol_version: str = "HTTP/1.1"
    token: str = os.environ["REMOTE_REWARD_TOKEN"]
    submitted: set[str] = set()
    submitted_lock: Lock = Lock()

    def _send_json(self, status: int, value: dict[str, object]) -> None:
        body = json.dumps(value, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        if self.headers.get("Authorization") == f"Bearer {self.token}":
            return True
        self._send_json(401, {"error": "unauthorized"})
        return False

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(length)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._authorized():
            return
        body = self._body()
        if self.path == "/enqueue":
            metadata_raw, separator, array_raw = body.partition(b"\n")
            try:
                metadata = json.loads(metadata_raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(400, {"error": "invalid metadata"})
                return
            if (
                not separator
                or not array_raw
                or not isinstance(metadata, dict)
                or metadata.get("media_type") != "image"
            ):
                self._send_json(400, {"error": "invalid image payload"})
                return
            service_id = hashlib.sha256(body).hexdigest()
            with self.submitted_lock:
                self.submitted.add(service_id)
            self._send_json(200, {"uuid": service_id})
            return
        if self.path == "/fetch":
            fields = parse_qs(body.decode("utf-8"))
            service_id = fields.get("uuid", [""])[0]
            provider = fields.get("type", [""])[0]
            with self.submitted_lock:
                known = service_id in self.submitted
            if not known or provider != "toy_image_hash":
                self._send_json(404, {"error": "unknown request"})
                return
            score = int(service_id[:8], 16) / 0xFFFFFFFF
            self._send_json(200, {"scores": {"toy_image_hash": [score]}})
            return
        self._send_json(404, {"error": "not found"})

    def log_message(self, format: str, *args: object) -> None:
        return


server: ThreadingHTTPServer = ThreadingHTTPServer(("127.0.0.1", 8765), RewardHandler)
print("Toy NFT reward service listening on http://127.0.0.1:8765", flush=True)
server.serve_forever()
PY

"$POSTTRAIN_PY" artifacts/nft/toy_reward_server.py \
  > artifacts/nft/toy_reward_server.log 2>&1 &
export NFT_REWARD_SERVER_PID=$!
trap 'kill "$NFT_REWARD_SERVER_PID" 2>/dev/null || true' EXIT

until curl --silent --fail http://127.0.0.1:8765/healthz >/dev/null; do
  sleep 1
done
```

Keep that process running while executing the NFT commands below. The toy score
is a deterministic hash of the submitted payload and is not a quality metric.

```bash
"$POSTTRAIN_PY" -m cosmos_framework.posttrain.entrypoints.nft \
  --validate-manifest -- experiment=cosmos3_nft_t2i_public_4gpu

"$POSTTRAIN_PY" -m cosmos_framework.posttrain.entrypoints.nft -- \
  experiment=cosmos3_nft_t2i_public_4gpu \
  job.name=nft_public_4gpu
```

## 8. Embodied PPO

The implementation uses the name Joint PPO because the policy/action backbone and value head form
one shared actor-value model and are optimized together in one learner, rather than using a
separate critic model. “Joint” describes this model and update topology, not a different PPO
objective.

Joint PPO needs an independently pinned LIBERO simulator interpreter and extra
model assets. Follow the complete public recipes:

- [π0.5 on LIBERO-Spatial](../guides/joint-ppo-pi05-public.md)
- [GR00T-N1.7 on LIBERO-Spatial](../guides/joint-ppo-gr00t-public.md)

## 9. Outputs and resume

Runs write their resolved configuration, logs, metrics, and checkpoints under:

```text
$IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>/<run_ts>/
```

Inspect `config_resolved.yaml` first: it is the fully merged configuration the
entrypoint actually used. When checkpointing is enabled, complete checkpoints
appear under `checkpoints/iter_*`, and `latest_checkpoint.txt` advances only
after a complete commit. `rl.resume=auto` resumes the latest complete checkpoint
for the same job identity; `rl.resume=off` starts a fresh attempt. Check the
selected recipe's resolved value before relaunching. See the
[checkpoint reference](../reference/checkpoint.md) and
[metrics reference](../reference/metrics.md) for the complete contracts.
