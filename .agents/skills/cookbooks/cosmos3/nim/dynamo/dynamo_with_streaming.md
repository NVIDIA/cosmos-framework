<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Dynamo With Streaming

Use [`reasoner_with_streaming.yaml`](reasoner_with_streaming.yaml) for Chat
Completions plus persistent video sessions over REST on NVIDIA Dynamo. The frontend and workers
use the same Cosmos3 NIM image, with different launch commands. The frontend
runs the bundled streaming adapter. Workers run the NIM server with streaming
enabled.

> **WebSocket is not supported in Dynamo mode.** The
> `WS /v1/streaming/ws` endpoint is unavailable on both the shared frontend
> and worker NIM endpoints. Use the REST session API below.

First complete the [prerequisites](dynamo_deployment.md#prerequisites) and
[namespace and cache setup](dynamo_deployment.md#prepare-the-namespace-and-cache).
Complete the shared [model-download step](dynamo_deployment.md#download-model-artifacts)
before starting workers. If reusing a cache, set `spec.pvcs[0].name` and
`ReasonerWorker.volumeMounts[0].name` in the streaming YAML, plus the download
Job's PVC `claimName`, to `<existing-cache-pvc>`. Keep `create: false`.
Replace all `<angle-bracket>` values before applying.

This manifest adapts the regular deployment's `v1alpha1` services schema,
resources, shared cache, and operator-managed worker health probes. Validate
the adapter and session requests below on your deployment.

## Set the NIM Image

[`reasoner_with_streaming.yaml`](reasoner_with_streaming.yaml) uses
`nvcr.io/nim/nvidia/cosmos3:2.1.0` for both the frontend and the workers. If you change it,
change both occurrences and the download Job together. The image must include
`/opt/nim/streaming/dynamo/`.

The frontend runs `/opt/nim/.venv/bin/python -m streaming.dynamo.frontend`
from `/opt/nim`. It serves streaming configuration and REST sessions. It
forwards ordinary Chat Completions to Dynamo without loading a model or
requesting a GPU. The stock `python3 -m dynamo.frontend` command does not
expose the session routes.

Workers start `/opt/nim/start_server.sh`: `NIM_DYNAMO_WORKER=true` and
`NIM_ENABLE_STREAMING=1` select the streaming worker and its private frontend.
The worker's `DYN_NAMESPACE_PREFIX` comes from the operator's pod label,
allowing the private frontend to discover models in suffixed worker
namespaces. The
streaming pool still uses the separate `NIM_DYNAMO_STREAMING_ENDPOINT` setting.
Refer to [Dynamo Worker Configuration](../configuration.md#run-nim-as-a-dynamo-worker)
and [Dynamo Streaming Configuration](../configuration.md#dynamo-streaming)
for environment variables, defaults, precedence, and timeouts.
The tested image bundled Dynamo `1.5.0.dev20260902`. The graph's operator
runtime override remains `1.5.0`. Verify compatibility for your
image/operator pairing.

Both manifests use the same Bash launch style (`set -euo pipefail` followed
by `exec`), explicit Kubernetes discovery, scheduling tolerations, and shared
worker environment values. The streaming manifest differs from the regular
one in the following ways:

- Uses the NIM image and its Python environment for the frontend in place of
  the regular pinned Dynamo image
- Changes the frontend module
- Enables `NIM_ENABLE_STREAMING`
- Disables asynchronous scheduling
- Adds the shared streaming endpoint to the frontend and workers

The YAML sets synchronous scheduling explicitly and uses the same
`NIM_DYNAMO_STREAMING_ENDPOINT` on frontend and workers. Keep that value
identical. Use a distinct endpoint for each separate model pool.

## Use Edge Reasoner

Edge supports the same REST session API in BF16 and FP8. In both
`model-download.yaml` and the streaming manifest's `ReasonerWorker`, set:

```yaml
- name: NIM_MODEL_VARIANT
  value: edge
- name: NIM_PRECISION
  value: fp8  # or bf16
- name: NIM_USE_DFLASH
  value: '0'
- name: NIM_STREAMING_SPEC_METHOD
  value: none
```

Replace the existing variant and precision entries rather than adding
duplicates. Keep the image, cache, and GPU selectors aligned between the Job and
workers. Edge does not support DFlash in this version. Set the streaming method
explicitly to `none`, because streaming defaults to DFlash and Edge has no
draft. The shared frontend needs no model-specific change. The client discovers
the Edge model from `/v1/streaming/config`. Use a separate streaming endpoint if
running Edge alongside a Nano or Super pool.

## Deploy and Check

Run from `.agents/skills/cookbooks/cosmos3/nim/dynamo`. If your image uses another registry,
configure its credentials and replace the `imagePullSecrets` entries in both
services and the download Job with `<your-image-pull-secret>`.

```bash
kubectl -n "$NAMESPACE" apply -f reasoner_with_streaming.yaml
kubectl -n "$NAMESPACE" get pods \
  -l nvidia.com/dynamo-graph-deployment-name=cosmos3-reasoner -w
```

Both YAML files define `cosmos3-reasoner`: apply only the chosen variant.
Applying the other variant updates that deployment and replaces its pods.
Existing streaming sessions are lost.

Follow the shared
[readiness and port-forward checks](dynamo_deployment.md#check-the-deployment),
including setting `NIM_URL`. Check frontend and worker logs for
`[Dynamo streaming]` messages confirming that the HTTP adapter and worker REST
handlers attached. Chat Completions use the same
[request example](dynamo_deployment.md#send-a-request). For the Python Reasoner,
Responses, and profile-inspection scripts, use the
[worker NIM URL commands](dynamo_deployment.md#run-the-cookbook-clients).

## Stream Video Frames Over REST

Use the forwarded frontend at `NIM_URL`. Create a session, send one JPEG or
PNG frame at a time, then delete the session. This Dynamo path supports raw
image bodies and JSON `image_b64`. `GET /v1/streaming/config` returns the
selected streaming worker's model and default retention/sampling settings:

```bash
curl -fsS "$NIM_URL/v1/streaming/config" | python3 -m json.tool
```

Use this config's `model` value when specifying a model for session creation.
It can be a local model path rather than the public alias in `/v1/models`.

The shared frontend exposes Dynamo's `/health` and `/v1/models`. NIM readiness,
metadata, and manifest routes belong to each worker's NIM endpoint. A 404 from
`/v1/health/ready` or `/v1/metadata` on the frontend does not establish that
streaming is unavailable.

From `.agents/skills/cookbooks/cosmos3/nim`, run the client in the
[pinned client environment](../prerequisites.md#initialize-the-example-environment):

```bash
uv run python examples/streaming.py --backend dynamo --transport rest \
  --nim-url "$NIM_URL" --fps 1 --max-frames 3
```

The client uses only the shared frontend. It checks for a registered `generate`
worker at `/health`, a served model at `/v1/models`, and streaming configuration
at `/v1/streaming/config`. It does not call worker metadata or manifest
endpoints and needs no worker URLs or port forwards. Missing runtime metadata
fields in older worker images do not block this mode. The client sends
JSON/base64 frames and requires REST in Dynamo mode.

These checks establish frontend discovery and streaming configuration, not the
readiness or selected profile of every worker. Operators can inspect individual
workers using the [deployment checks](dynamo_deployment.md#check-the-deployment).

After the readiness checks above, try one local JPEG frame:

```bash
(
  set -euo pipefail
  created=$(curl -fsS "$NIM_URL/v1/streaming/sessions" \
    -H 'Content-Type: application/json' \
    -d '{"system_prompt":"Describe each frame in one short sentence.","question":"What is happening?","fps":1,"sampling":{"max_tokens":48}}')
  session_id=$(printf '%s' "$created" | python3 -c \
    'import json,sys; print(json.load(sys.stdin)["session_id"])')
  trap 'curl -fsS -X DELETE "$NIM_URL/v1/streaming/sessions/$session_id"' EXIT

  curl -fsS "$NIM_URL/v1/streaming/sessions/$session_id/frame" \
    -H 'Content-Type: image/jpeg' --data-binary '@<path-to-frame.jpg>'
)
```

For video, decode frames on the client and wait for each reply before sending
the next. Keep the returned `session_id` unchanged: frame uploads and deletion
route to its owner even through a different frontend replica. This ownership
does not depend on the Chat Completions affinity header or its 300-second TTL.
Refer to [Streaming](../streaming.md#session-configuration) for session fields
and [streaming settings](../configuration.md#streaming-video-sessions) for
limits. The REST idle timeout defaults to 120 seconds.

A session-route 404 can indicate an old frontend/worker image or streaming
disabled on the worker. A 404 can also mean that an existing session expired.
A 503 can mean workers are loading, capacity is full, or the owning worker is
unavailable.
Sessions do not migrate after worker loss, so create a new one. Do not
automatically retry a frame after a timeout.

## Other Routes

The adapter forwards ordinary HTTP requests to the bundled Dynamo frontend,
including `POST /v1/responses`. Use `store=false` for Responses creation, as in
the [Dynamo request example](dynamo_deployment.md#send-a-request). Creation does
not establish support for stored-response retrieval, cancellation, background
responses, or `previous_response_id`. Those require separate backend storage
support. NIM's `NIM_DISABLE_RESPONSES_ROUTE` controls the worker proxy's route,
not the shared frontend's routes.

## Scale or Stop

Set `spec.services.ReasonerWorker.replicas` to `<worker-count>` in the streaming
YAML and apply that file again. Each replica needs the shared
[resources and cache](dynamo_deployment.md#prerequisites). Scaling down or
replacing workers loses their active sessions because session state does not
migrate. Set the count to `0` to release the GPUs while retaining the frontend
and cache. Restore a positive count and wait for readiness before sending more
requests.

To stop serving while keeping the cache PVC and secrets:

```bash
kubectl -n "$NAMESPACE" delete -f reasoner_with_streaming.yaml
```

Deleting the graph interrupts active requests and loses active sessions. Run
the delete command only when you intend to stop serving. Stop the port-forward
with Ctrl+C.
