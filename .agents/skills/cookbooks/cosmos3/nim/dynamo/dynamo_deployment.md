<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Deploy Cosmos3 Reasoner With Dynamo

Run an NVIDIA Dynamo frontend with Cosmos3 Reasoner workers on Kubernetes. This
example uses a pinned Dynamo frontend image and a Cosmos3 NIM worker image.
It launches `python3 -m dynamo.frontend` for Chat Completions, with
round-robin routing and session affinity. It starts
two Nano FP8 workers, each on one GPU, with persistent video sessions disabled.
For those sessions, use [Dynamo With Streaming](dynamo_with_streaming.md).

For required worker environment variables, discovery, and proxy settings,
refer to [Dynamo Configuration](../configuration.md#dynamo-configuration), including
[running NIM as a Dynamo worker](../configuration.md#run-nim-as-a-dynamo-worker).

The manifests use the tested operator's `nvidia.com/v1alpha1` API and
`spec.services` schema (`componentType`, `extraPodSpec.mainContainer`, and
service-level resources). They cannot be applied unchanged to an operator
that serves only the `v1beta1` `spec.components` schema. Helm is not needed,
but the Dynamo operator is still required.

Replace values in `<angle-brackets>` with your own values before applying.
The workers and [`model-download.yaml`](model-download.yaml) use the same NIM
image, `nvcr.io/nim/nvidia/cosmos3:2.1.0`. The regular frontend stays pinned
to `nvcr.io/nvidia/ai-dynamo/dynamo-frontend-nightly:20260906-c70f1c7`.
The tested NIM image bundled `ai-dynamo==1.5.0.dev20260902`, with the operator's
canonical `runtimeVersionOverride: "1.5.0"`. Verify compatibility among the
frontend, worker image, and operator when changing versions.

This setup is validated for Chat Completions. Validate streaming sessions,
Responses, and the other cookbook clients separately on your deployment.

## Prerequisites

Before you deploy, make sure you have the following:

- A Kubernetes cluster with NVIDIA GPU support and a compatible Dynamo
  operator/platform serving the `v1alpha1` `DynamoGraphDeployment` schema above.
  Check with `kubectl get crd dynamographdeployments.nvidia.com -o yaml` and
  `kubectl explain dynamographdeployment.spec.services --api-version=nvidia.com/v1alpha1`.
  The operator must provide Kubernetes discovery permissions and inject worker
  system-port health probes (`/live` and `/health` on port 9090).
- Capacity for two workers: each requests one GPU compatible with the image's
  Nano FP8 profile, 32 CPUs, and 96 GiB of pod memory (128 GiB limit). The
  frontend requests another 4 CPUs and 8 GiB (limits: 8 CPUs and 16 GiB).
  These are deployment resource settings, not profile memory floors. Verify
  each GPU and effective host/container RAM against the
  [Support Matrix](../support-matrix.md). Do not add memory across GPUs.
- A ReadWriteMany (RWX) storage class and a 200 GiB model cache writable by
  the image's user with group 1000. All workers and the download Job must be
  able to mount it. Workers must be able to download model artifacts if needed.
- Capacity for the download Job before workers start: one compatible GPU,
  4 CPUs and 16 GiB requested (limits: 8 CPUs and 32 GiB). This Job runs
  image-specific profile selection and downloads artifacts without serving.
  Its memory limit is not evidence that a serving worker fits that limit.
- `kubectl`, cURL, Python 3, and an exported `NGC_API_KEY` with image and model
  access. Refer to [Network and NGC
  Access](../prerequisites.md#network-and-ngc-access).

Both services and the download Job tolerate the `nvidia.com/gpu` taint and the
`kubernetes.io/arch=arm64:NoSchedule` taint. They leave node architecture unset,
allowing placement on compatible AMD64 or ARM64 nodes. Tolerations permit
scheduling, but they do not select an architecture or establish image/GPU
compatibility. Choose an architecture supported by your image and GPU profile,
and add matching `nodeSelector`/tolerations to both services and the download
Job if your cluster uses other constraints. Use your own node labels to
constrain placement on mixed-GPU clusters.

The example pins Nano FP8 to match the tested deployment. Keep the image,
`NIM_MODEL_VARIANT`, `NIM_PRECISION`, and `NIM_TAGS_SELECTOR` aligned in the Job
and workers when choosing another profile. Confirm image support during profile
selection. For unified-memory hosts, follow the
[Reasoner memory-share guidance](../deployment.md#set-the-reasoner-memory-share-on-unified-memory-systems)
in the Job and workers before starting either. Do not assume these
discrete-GPU pod settings apply unchanged.

To match the tested configuration, workers set `NIM_ENABLE_WARMUP=0` and disable
text, video, and SigLIP guardrails explicitly. Review those policy choices
before use. To enable the checks, remove those guardrail overrides or set them
to `1`. Disabled warmup means Ready does not establish representative inference
has completed. Do not copy a cold-start duration as an expected startup time.

## Prepare the Namespace and Cache

Run commands from `.agents/skills/cookbooks/cosmos3/nim/dynamo`, with the intended Kubernetes
context selected. Set your namespace (also export it in terminals used below).
Create it if needed, then create the two secrets referenced by the manifests.
Add namespace labels only if required by your cluster policy, using your own values:

```bash
export NAMESPACE="<your-namespace>"
kubectl create namespace "$NAMESPACE"
kubectl -n "$NAMESPACE" create secret docker-registry ngc-registry-secret \
  --docker-server=nvcr.io --docker-username='$oauthtoken' \
  --docker-password="$NGC_API_KEY"
kubectl -n "$NAMESPACE" create secret generic ngc-api-secret \
  --from-literal=NGC_API_KEY="$NGC_API_KEY"
```

Reuse these resources if they already exist. In
[`model-cache.yaml`](model-cache.yaml), replace
`<your-rwx-storage-class>` with your cluster's RWX storage class, then apply it:

```bash
kubectl -n "$NAMESPACE" apply -f model-cache.yaml
```

If you already have a suitable cache PVC, skip its creation.
For the `v1alpha1` graph manifests, also replace `spec.pvcs[0].name` and
`ReasonerWorker.volumeMounts[0].name` with that claim name, and keep
`create: false`. Set the download Job's
`volumes[].persistentVolumeClaim.claimName` to the same claim. The cache
mounts at `/opt/nim/.cache` without a subdirectory. A storage class using
`WaitForFirstConsumer` can leave the PVC Pending until the download Job or
workers are scheduled.

## Download Model Artifacts

Keep the image and profile selectors in
[`model-download.yaml`](model-download.yaml) matched to the workers, then run:

```bash
kubectl -n "$NAMESPACE" apply -f model-download.yaml
kubectl -n "$NAMESPACE" logs -f job/cosmos3-reasoner-model-download
kubectl -n "$NAMESPACE" wait --for=condition=Complete \
  job/cosmos3-reasoner-model-download --timeout=3600s
```

The Job first runs `profile_selection.startup`, sources the selected profile,
and invokes the image's `nimutils.download_models` helper. Selection success
establishes candidate-profile compatibility. Serving still requires a cold
start and representative requests on every participating device. The one-hour
wait is an operator wait ceiling, not an expected download duration. Inspect
Job status and logs if it fails or times out. Do not deploy workers until it
completes. The Job reserves one GPU while running and releases it on
completion. Completed Job objects expire after one day, but cached artifacts
remain on the PVC.

## Deploy

```bash
kubectl -n "$NAMESPACE" apply -f reasoner.yaml
kubectl -n "$NAMESPACE" get pods \
  -l nvidia.com/dynamo-graph-deployment-name=cosmos3-reasoner -w
```

## Check the Deployment

Wait for the frontend and worker pods to become Ready, then stop the watch
with Ctrl+C. Cold startup still includes model loading and can download any
missing artifacts.
The worker deliberately leaves system-port probes and `DYN_SYSTEM_PORT` to the
operator. Do not replace them with NIM HTTP liveness probes. Inspect the
generated pods to confirm the operator injected port 9090 and its health
probes. The operator creates the frontend Service. Forward it in a separate
terminal:

```bash
kubectl -n "$NAMESPACE" port-forward service/cosmos3-reasoner-frontend 8000:8000
```

Check registration and discover the served model:

```bash
export NIM_URL=http://localhost:8000
curl -fsS "$NIM_URL/health" | python3 -m json.tool
curl -fsS "$NIM_URL/v1/models" | python3 -m json.tool
```

Before sending requests, confirm `/health` lists a distinct `generate`
instance for each worker in this deployment and `/v1/models` is nonempty.
`/health` can also list other deployments' workers. Match the instance's
namespace and transport address to your worker pods. Frontend health alone
does not establish worker readiness. Workers expose NIM's
`/v1/health/ready`, `/v1/metadata`, and `/v1/metrics` on their own port 8000.
To inspect those endpoints, port-forward a worker pod to a different local
port.
Check each worker's readiness and match `selectedModelProfileId` from metadata
to `/v1/manifest`, using [inspect_profile.py](../examples/inspect_profile.py).

## Send a Request

The frontend serves `/v1/chat/completions`. Discover the model ID rather than
assuming it matches the Kubernetes resource name:

```bash
MODEL=$(curl -fsS "$NIM_URL/v1/models" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
curl -fsS "$NIM_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Describe how a robot can avoid obstacles.\"}],\"max_tokens\":128}"
```

Refer to [Reasoning](../reasoning.md) for image/video request formats. The
frontend uses Dynamo's API, including OpenAI `response_format`. The worker's
NIM proxy handles NIM-specific aliases. The frontend does not expose NIM
metadata. The Python Reasoner and Responses examples require a worker NIM URL.
Refer to [Run the Cookbook Clients](#run-the-cookbook-clients). Verify the selected profile through the worker's `/v1/manifest`. For
persistent video sessions, use the client's explicit
[Dynamo mode](dynamo_with_streaming.md#stream-video-frames-over-rest), which
uses only shared frontend endpoints and does not require worker metadata.

The Responses path needs separate validation. For a frontend build that
supports stateless Responses creation, try the discovered model:

```bash
curl -fsS "$NIM_URL/v1/responses" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"input\":\"Describe how a robot can avoid obstacles.\",\"store\":false,\"max_output_tokens\":128}"
```

Successful creation with `store=false` does not imply support for persisted
retrieval, cancellation, background responses, or `previous_response_id`.
The worker NIM proxy also forwards Responses creation unless
`NIM_DISABLE_RESPONSES_ROUTE=true`. That setting does not disable the shared
frontend's own route. Check the selected frontend build and operator settings
when a route is absent.

## Run the Cookbook Clients

`inspect_profile.py`, `reasoner.py`, and `reasoner_responses.py` use the worker's
NIM metadata and model endpoints. Use these commands with either deployment
variant after confirming the selected image exposes the required metadata.
Forward the worker's NIM port in a separate terminal:

```bash
kubectl -n "$NAMESPACE" port-forward pod/<reasoner-worker-pod> 8001:8000
```

From `.agents/skills/cookbooks/cosmos3/nim`, run the clients in the
[pinned client environment](../prerequisites.md#initialize-the-example-environment):

```bash
curl -fsS http://localhost:8001/v1/health/ready
NIM_URL=http://localhost:8001 uv run python examples/inspect_profile.py
NIM_URL=http://localhost:8001 uv run python examples/reasoner.py --case image_caption
NIM_URL=http://localhost:8001 uv run python examples/reasoner_responses.py
```

These per-command assignments leave the shared frontend's `NIM_URL` unchanged.
Use `--case all` to run the full Reasoner catalog. The
[Dynamo streaming client](dynamo_with_streaming.md#stream-video-frames-over-rest)
uses the shared frontend directly and needs no worker port-forward.

Both deployment manifests set the worker container's `DYN_NAMESPACE_PREFIX`
from its `nvidia.com/dynamo-namespace` pod label. The private frontend inherits
this setting, so model discovery includes the operator's suffixed worker
namespaces. Prefix matching takes precedence over the exact `DYN_NAMESPACE`
filter and covers all worker pools under that deployment prefix.

Check `/v1/models` on the interface receiving inference. A worker's NIM
readiness check can succeed while its private frontend's model list is empty.
In that case, check that `DYN_NAMESPACE_PREFIX` resolves to the deployment's
namespace prefix in the worker container. A Responses
"Model not found" error does not mean the Responses route is absent. Successful
requests through the shared frontend do not validate the worker proxy's routes.

## Scale or Stop

To add workers, set `spec.services.ReasonerWorker.replicas` to `<worker-count>` in
[`reasoner.yaml`](reasoner.yaml) and apply it again. Each
replica needs the listed resources and access to the shared cache. Changing
the model or GPU topology also requires matching `NIM_MODEL_VARIANT`,
`NIM_PRECISION`, `NIM_TAGS_SELECTOR`, and GPU requests/limits to the selected
image's profile. Update the download Job selectors too if you need artifacts
for a different profile. This example uses fixed replicas. Autoscaling is not
enabled.

Set `spec.services.ReasonerWorker.replicas` to `0` to release the GPUs while
retaining the frontend and cache. Restore a positive count and wait for
readiness to resume inference.

To stop serving, delete only the graph deployment:

```bash
kubectl -n "$NAMESPACE" delete -f reasoner.yaml
```

This removes the frontend and workers while retaining the externally managed
cache PVC (`create: false`) and secrets for reuse. Run it only when you intend
to stop serving. It interrupts active requests. Stop the port-forward with
Ctrl+C.
