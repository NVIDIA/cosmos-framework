<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Deploy the Cosmos3 NIM

Use this page to authenticate to NVIDIA NGC, choose a model, launch Generator or
Reasoner, and verify the selected service. The commands pin the current 2.1.0
release image. Use the [Support Matrix](support-matrix.md) to choose a
compatible configuration.

## How Selection Works

You normally select a runtime and model, not a profile ID:

1. Choose Generator or Reasoner.
2. Choose a model variant.
3. Optionally pin precision. Generator prefers FP8 when compatible. Reasoner
   derives its preference from GPU compute capability.
4. For Generator, choose latency or throughput.
5. The NIM selects the best compatible profile for those choices and the
   visible GPUs.

Selection prefers a resident profile that uses the visible GPUs well and can
fall back to an equivalent lower-memory offload or tensor-parallel layout when
current free GPU memory is short. On unified-memory systems, selection counts
70% of the shared pool as GPU memory and uses only resident Generator profiles.
If no profile fits, startup fails. Free GPU memory and restart. Refer to
[GPU Architecture and Topology](support-matrix.md#gpu-architecture-and-topology)
for the memory rules.

### Select a Generator Model

| `NIM_MODEL_VARIANT` | Supported Use |
| --- | --- |
| `nano` | General-purpose Nano Generator |
| `edge-droid` | Edge-DROID policy with an action-only response; BF16 only; [experimental](support-matrix.md#choose-a-model) |
| `nano-droid` | Nano-DROID policy with an action-only response; BF16 only |
| `super` | General-purpose Super Generator |
| `super-t2i` | Full-step T2I specialist |
| `super-t2i-4step` | Four-step T2I specialist |
| `super-i2v` | Full-step I2V specialist |
| `super-i2v-4step` | Four-step I2V specialist |

Specialist variants accept only their task. There is no general-purpose Edge
Generator. Set the workload objective explicitly:

| `NIM_PERF_PROFILE` | Optimize For |
| --- | --- |
| `latency` | Lower latency for an individual request |
| `throughput` | Higher aggregate request rate |

The default is `latency`.

### Select a Reasoner Model

Set `NIM_MODEL_TYPE=reasoner` and choose `NIM_MODEL_VARIANT=edge`, `nano`, or
`super`. Reasoner rejects `NIM_PERF_PROFILE` at startup. DFlash speculative
decoding is on by default for Nano and Super. Set `NIM_USE_DFLASH=0` to
disable it. Edge has no DFlash draft: its profiles disable DFlash and reject
`NIM_USE_DFLASH=1` (refer to [Configuration](configuration.md#speculative-decoding)).

### Precision Selection

Omit `NIM_PRECISION` normally. Generator prefers FP8 when compatible. Reasoner
prefers BF16 on compute capability 8.0 through 8.8, FP8 on 8.9 through 9.x, and
NVFP4 on 10.0 or newer, falling back to another compatible precision.
Set `NIM_PRECISION=bf16`, `fp8`, or another available value only when the
workload requires it. Edge Reasoner is BF16 or FP8 (no NVFP4), and Edge-DROID
and Nano-DROID are BF16 only.

## Before You Deploy

Verify the host against [Prerequisites](prerequisites.md) and choose a
compatible configuration from the [Support Matrix](support-matrix.md). For a
Kubernetes deployment, follow [Deploy with Helm](helm.md).

## Authenticate to NGC

Export `NGC_API_KEY` (not `NGC_TOKEN`) as described under
[Network and NGC Access](prerequisites.md#network-and-ngc-access). Never put
the key in source control. Docker saves the registry login after
`docker login`. Docker Desktop uses a native credential store, while standalone
Docker Engine without a helper stores the login base64-encoded in
`$HOME/.docker/config.json` and prints a warning.

For stronger at-rest protection, configure a
[supported credential helper](https://docs.docker.com/reference/cli/docker/login/#credential-stores)
whose `docker-credential-*` program is on `PATH`, for example
`docker-credential-pass` with this entry in `$HOME/.docker/config.json`:

```json
{
  "credHelpers": {
    "nvcr.io": "pass"
  }
}
```

To migrate an existing saved login, run `docker logout nvcr.io` first. Then log
in with the literal username `$oauthtoken`:

```bash
echo "$NGC_API_KEY" \
  | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

`--password-stdin` keeps the key out of shell history but does not encrypt the
saved credential. Without a helper, log out when you finish evaluating.

Pin and pull the current release image:

```bash
export NIM_IMAGE='nvcr.io/nim/nvidia/cosmos3:2.1.0'
docker pull "$NIM_IMAGE"
```

Do not replace the versioned tag with `latest` or another unvalidated image.

### Set the Reasoner Memory Share on Unified-Memory Systems

On NVIDIA DGX Spark/GB10 and Jetson AGX Thor, the NIM gives the model 70% of the
unified host/device memory pool. It applies that share to profile floors and
uses it as the Reasoner `NIM_GPU_MEMORY_UTILIZATION` default. No variable is
required. Lower the share only when host measurements show that media
processing or other host workloads need more memory, for example:

```bash
export NIM_GPU_MEMORY_UTILIZATION=0.60
```

If you set a value, pass `-e NIM_GPU_MEMORY_UTILIZATION` in every Reasoner
launch command on that system. Profile selection and preflight always use the
70% share, so the value does not change the preflight result. Do not set it
for Generator. Refer to
[Reasoner Context and Scheduling](configuration.md#context-and-scheduling).

## Run the Pre-Download Profile Preflight

After the image is present locally, run profile selection without starting the
server or downloading model artifacts. This example checks the same Nano
Generator selectors used by the launch example:

```bash
docker run --rm \
  --gpus '"device=0"' \
  -e NIM_MODEL_TYPE=generator \
  -e NIM_MODEL_VARIANT=nano \
  -e NIM_PERF_PROFILE=latency \
  --entrypoint /bin/bash \
  "$NIM_IMAGE" \
  -lc '
    set -e
    output=/tmp/cosmos3-preflight.env
    /opt/nim/.venv/bin/python -m profile_selection.startup --output "$output"
    printf "Profile preflight passed.\n"
  '
```

Use the selectors, GPU exposure, and container memory limit of the real
deployment. For Reasoner, set `NIM_MODEL_TYPE=reasoner`, select `edge`, `nano`,
or `super`, and omit `NIM_PERF_PROFILE`. Preflight ignores
`NIM_GPU_MEMORY_UTILIZATION`, so leave it out here even if you
[lowered the memory share](#set-the-reasoner-memory-share-on-unified-memory-systems).
This example checks Nano Reasoner:

```bash
docker run --rm \
  --gpus '"device=0"' \
  -e NIM_MODEL_TYPE=reasoner \
  -e NIM_MODEL_VARIANT=nano \
  --entrypoint /bin/bash \
  "$NIM_IMAGE" \
  -lc '
    set -e
    output=/tmp/cosmos3-preflight.env
    /opt/nim/.venv/bin/python -m profile_selection.startup --output "$output"
    printf "Profile preflight passed.\n"
  '
```

Do not pass `NGC_API_KEY`, `NIM_MODEL_PATH`, or `NIM_DFLASH_MODEL_PATH` to this
bundled-profile check. Checkpoint-source validation is a separate bring your own
checkpoint (BYOC) step, and an `hf://` source can download files.

Preflight success is necessary, not sufficient. It confirms that the image has
a profile matching the selectors, compute capability, system-memory floor, and
total and free video memory (VRAM). It also logs Generator Transfer admission.
It does not load the model. Only a cold start and representative requests
confirm host RAM, disk, shared memory, the driver stack, media, and inference.
Preflight writes nothing outside its removed container. Resolve any reported
failure before a cold download.

## Prepare the Model Cache

```bash
export LOCAL_NIM_CACHE="${LOCAL_NIM_CACHE:-$HOME/.cache/nim/cosmos3}"
mkdir -p "$LOCAL_NIM_CACHE"
chmod -R a+rwX "$LOCAL_NIM_CACHE"
```

These permissions suit local development. The container runs as UID/GID 1000
(`nvs`). In production, grant that UID/GID write access through a group or
ACL.
Keep the cache between runs to avoid repeated downloads.

## Launch Generator

This example runs general-purpose Nano with latency on host port 8000:

```bash
docker run -d --name cosmos3-generator \
  --gpus '"device=0"' \
  --shm-size 16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=65536:65536 \
  -p 8000:8000 \
  -e NGC_API_KEY \
  -e NIM_MODEL_TYPE=generator \
  -e NIM_MODEL_VARIANT=nano \
  -e NIM_PERF_PROFILE=latency \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  "$NIM_IMAGE"
```

The container starts in the background. Follow startup logs with
`docker logs -f <container>` (here `cosmos3-generator`). Ctrl+C stops
following without stopping the container.

On a discrete-GPU host, a non-offloading Nano profile needs at least 40 GiB of
system memory for BF16 or 35 GiB for FP8. Selection enforces these
[floors](support-matrix.md#generator-profiles).

Change the model and objective as needed, expose every GPU a multi-GPU
configuration requires, and add `-e NIM_PRECISION=fp8` only to pin precision.
The Generator uses as many exposed GPUs as a compatible profile supports. To
limit them, refer to [GPU Selection](configuration.md#gpu-selection).

## Launch Reasoner

Reasoner uses the same host port. Remove a running Generator first:

```bash
docker rm -f cosmos3-generator
```

### DGX Spark and Jetson AGX Thor

Use Nano. The NIM applies the unified-memory default. Add
`-e NIM_GPU_MEMORY_UTILIZATION="$NIM_GPU_MEMORY_UTILIZATION"` only if you
exported a lower measured value:

```bash
docker run -d --name cosmos3-reasoner \
  --gpus '"device=0"' \
  --shm-size 16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=65536:65536 \
  -p 8000:8000 \
  -e NGC_API_KEY \
  -e NIM_MODEL_TYPE=reasoner \
  -e NIM_MODEL_VARIANT=nano \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  "$NIM_IMAGE"
```

### Discrete GPU Example

This example launches Super FP8 Reasoner. It needs compute capability 8.9+,
67 GiB of VRAM usable after the 4-GiB Reasoner reserve, and 43 GiB of system
memory (refer to
[Reasoner Profiles](support-matrix.md#reasoner-profiles)):

```bash
docker run -d --name cosmos3-reasoner \
  --gpus '"device=0"' \
  --shm-size 16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=65536:65536 \
  -p 8000:8000 \
  -e NGC_API_KEY \
  -e NIM_MODEL_TYPE=reasoner \
  -e NIM_MODEL_VARIANT=super \
  -e NIM_PRECISION=fp8 \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  "$NIM_IMAGE"
```

It uses the DFlash and video-token pruning defaults. To use another
configuration, change the model and precision selectors together and expose
the GPUs it needs.

To run both runtimes concurrently, publish one on another host port and set
that client's `NIM_URL` accordingly.

## Deploy Edge Models

For Edge Reasoner, replace the model selector lines in the command in
[Launch Reasoner](#launch-reasoner) with these, using `fp8` or `bf16`:

```text
-e NIM_MODEL_TYPE=reasoner \
-e NIM_MODEL_VARIANT=edge \
-e NIM_PRECISION=fp8
```

The Responses API stays enabled by default (`NIM_DISABLE_RESPONSES_ROUTE=false`).

> **Known issue:** With `NIM_ENABLE_STREAMING=true`, Edge Reasoner fails at
> startup with the default `NIM_STREAMING_SPEC_METHOD=dflash`, because Edge
> does not support DFlash. To stream with Edge, set both
> `NIM_ENABLE_STREAMING=true` and `NIM_STREAMING_SPEC_METHOD=none`, which turns
> off speculative decoding for streaming (refer to
> [Advanced Streaming Controls](configuration.md#advanced-streaming-controls)).

For Edge-DROID, which is
[experimental](support-matrix.md#choose-a-model), replace the model selector
lines in the command in [Launch Generator](#launch-generator) with these:

```text
-e NIM_MODEL_TYPE=generator \
-e NIM_MODEL_VARIANT=edge-droid \
-e NIM_PRECISION=bf16 \
-e NIM_PERF_PROFILE=latency \
-e NIM_CACHE_BACKEND=none
```

Edge-DROID disables SeaCache by default. The explicit `NIM_CACHE_BACKEND=none`
keeps that visible, so leave `NIM_CACHE_CONFIG` unset. Guardrails stay enabled. After readiness, follow
[Verify the Selection](#verify-the-selection)
and run the [Reasoner catalog](reasoning.md#run-the-task-catalog) or the
[DROID policy client](action.md#edge-droid-and-nano-droid-policy). Edge-DROID
serves policy only, not image/video generation or forward/inverse dynamics.

## Wait for Readiness

Liveness means the HTTP process exists. Readiness means the selected model can
serve requests:

```bash
export NIM_URL=${NIM_URL:-http://localhost:8000}
curl -f "$NIM_URL/v1/health/live"

until curl -fsS "$NIM_URL/v1/health/ready" >/dev/null; do
  sleep 10
done
```

A cold start (download, materialization, model load, warmup) takes much longer
than HTTP startup. Send inference only after readiness succeeds.

## Verify the Selection

```bash
curl -fsS "$NIM_URL/v1/metadata" | python3 -m json.tool
curl -fsS "$NIM_URL/v1/manifest" | python3 -m json.tool
```

Metadata reports the selected model and profile. Confirm that `model_type` and
`inference_endpoint` are `generator` and `/v1/infer`, or `reasoner` and
`/v1/chat/completions`, before sending requests.

## Offline and Air-Gapped Deployment

The NIM can start offline from a cache filled by an earlier online start. The
cache holds only the selected profile's artifacts, so fill it for every
configuration you run offline.

1. Make the container image available on the offline host. Push it to a
   registry that the host can reach, or move it as an archive:

   ```bash
   docker save "$NIM_IMAGE" -o cosmos3-nim.tar
   # On the offline host:
   docker load -i cosmos3-nim.tar
   ```

   Set `NIM_IMAGE` on the offline host to the loaded or mirrored image.
2. On a host with NGC access, launch each configuration as shown in
   [Launch Generator](#launch-generator) or [Launch Reasoner](#launch-reasoner),
   with the offline host's selectors and GPU exposure, and wait for readiness.
   Several configurations can share one cache.
3. Copy the cache with a tool that preserves symbolic links:

   ```bash
   rsync -a "$LOCAL_NIM_CACHE/" /path/to/offline-cache/
   ```

   Snapshot files are relative symbolic links into `blobs/`, so tools that
   dereference links break the cache. On the offline host,
   `find /path/to/offline-cache -xtype l` must print nothing. Apply the
   ownership from [Prepare the Model Cache](#prepare-the-model-cache).
4. On the offline host, set `LOCAL_NIM_CACHE` to the copy and run the same
   launch command. Generator can omit `-e NGC_API_KEY`. Keep it for Reasoner,
   whose keyless offline start has not been validated.
5. For Generator, do not set `NIM_DISABLE_MODEL_DOWNLOAD=1`. Startup rejects it
   because guardrail artifacts must be materialized from the cached files.
6. [Wait for Readiness](#wait-for-readiness). With a complete cache, every
   artifact logs "Skipping download, using cached copy of file" instead of
   downloading it. The "Downloading manifest profile" line does not indicate
   network access.

To test a cache on a connected host before moving it, replace `-p 8000:8000`
with `--network none` and check readiness inside the container:

```bash
docker exec cosmos3-generator python3 -c \
  'import urllib.request as u; print(u.urlopen("http://localhost:8000/v1/health/ready").status)'
```

## Advanced Profile Controls

Tag selectors, offload modes, and profile-ID pins are for validated needs
only. Refer to [Advanced Profile Controls](configuration.md#advanced-profile-controls)
for `NIM_TAGS_SELECTOR`, and to [Advanced Settings](configuration.md#advanced-settings)
for `NIM_OFFLOAD_MODE` and `NIM_MODEL_PROFILE`.

## Docker Flag Summary

| Flag | Purpose |
| --- | --- |
| `--gpus` | Expose the GPUs available to automatic profile selection |
| `--shm-size` | Provide shared memory for media and intermediate buffers |
| `--ulimit` | Raise memory-lock, stack, and open-file limits |
| `-p HOST:8000` | Publish the selected host port |
| `-e NGC_API_KEY` | Pass the exported NGC credential |
| `-e NIM_GPU_MEMORY_UTILIZATION=...` | Optionally lower the Reasoner memory share on DGX Spark or Jetson AGX Thor |
| `-v ...:/opt/nim/.cache` | Persist model artifacts |

GPU counts, VRAM floors, and host RAM are in the
[Support Matrix](support-matrix.md).

## Next Steps

- [Configuration](configuration.md) lists user-facing and advanced variables.
- [Bring Your Own Checkpoint](bring-your-own-checkpoint.md) covers local and
  Hugging Face model sources.
- [API Reference](api-reference.md) routes requests to the right task guide.
- [Operations](operations.md) covers health, logs, metrics, and failures.

Remove the example containers when finished:

```bash
docker rm -f cosmos3-generator
docker rm -f cosmos3-reasoner
docker logout nvcr.io
```

Logout does not revoke the NGC key. Deactivate or delete an evaluation key in
the NGC user interface. The model cache remains.
