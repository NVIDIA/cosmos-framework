<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Operations and Troubleshooting

Interpret health, inspect the selected profile, configure logging and
guardrails, and diagnose failures. Refer to [Deployment](deployment.md) for launch
and [Configuration](configuration.md) for environment variables.

## Health and Startup

| Probe | Healthy Meaning | Not Verified |
| --- | --- | --- |
| `/v1/health/live` | HTTP process is alive | Model artifacts are loaded or inference can run |
| `/v1/health/ready` | Selected backend reports ready after startup work | Every capability/media case has passed a basic verification test |

Check both probes:

```bash
export NIM_URL=${NIM_URL:-http://localhost:8000}
curl -i "$NIM_URL/v1/health/live"
curl -i "$NIM_URL/v1/health/ready"
```

Both runtimes return `{"object": "health.response", "message": "Service is
ready", "status": "ready"}` for readiness, and `"Service is live"` / `"live"`
for liveness. Treat the HTTP status as the primary result. Failed probes are
non-2xx.

Cold startup can include NVIDIA NGC download, cache materialization, engine build,
model load, and warmup, so a live-but-not-ready interval is expected. Size
orchestrator startup budgets for it. Profile selection reads each visible GPU's
free memory once, before materialization. A competing allocation can cause a
lower-memory fallback or startup failure, and later changes do not trigger
reselection.

## Inspect the Running Service

Use the active runtime rather than launch-time assumptions:

```bash
curl -fsS "$NIM_URL/v1/metadata" | python3 -m json.tool
```

Each endpoint answers a different operational question:

| Endpoint | Operational Question |
| --- | --- |
| `/v1/metadata` | Which `model_type`, primary `inference_endpoint`, profile/checkpoint, and Generator `model_variant` did the NIM report? |
| `/v1/models` | Which model ID should the Reasoner client send? This endpoint alone does not identify the active runtime. |
| `/v1/manifest` | Which profile/artifact information is present in this image? |
| `/v1/version` | Which release/server API is running? |
| `/v1/license` | Where is the bundled product license information? |
| `/openapi.json` | Which routes and schemas does this active runtime expose? |

`/v1/manifest` embeds the YAML manifest in `manifest_file`. To print the
selected profile's ID and manifest tags, initialize the
[pinned client environment](prerequisites.md#initialize-the-example-environment)
and run:

```bash
uv run python examples/inspect_profile.py
```

Profile IDs and tags are image-specific. Do not copy them to another image.

## Logging

Set log level, JSON-line output, backend verbosity, and Reasoner request-body
logging with the [Server and Logging](configuration.md#server-and-logging) and
[Advanced Settings](configuration.md#advanced-settings) variables. At `INFO`,
request prompts are redacted. At `DEBUG`, full prompts are logged, so use
`DEBUG` only during diagnosis and treat its logs as sensitive. Never log
secrets or raw media data URLs.

To read recent logs, run the command for your container:

```bash
docker logs --since 10m cosmos3-generator
docker logs --since 10m cosmos3-reasoner
```

## Metrics

Scrape `$NIM_URL/v1/metrics`. Refer to [Monitoring](monitoring.md) for Prometheus
and Grafana setup and the metric families of each runtime.

## Long-Running Requests

Generator `/v1/infer` requests are synchronous: the connection stays quiet
until inference and serialization finish, which alone does not mean a hang.
The backend queue-plus-execution ceiling is
[`NIM_TRITON_REQUEST_TIMEOUT`](configuration.md#generator-configuration),
default 30 minutes. A longer client timeout does not extend it. For the
60-minute Transfer client, launch the Generator with
`-e NIM_TRITON_REQUEST_TIMEOUT=3600000000`. Before retrying a quiet request,
check the container logs and GPU activity rather than readiness, and avoid
blind retries that duplicate expensive work.

## Unified-Memory Diagnostics

On NVIDIA DGX Spark, Jetson AGX Thor, and other unified-memory systems, [inspect
current memory](prerequisites.md#inspect-unified-memory-capacity-and-current-state)
to tell a capacity failure from a temporary one. For a current-memory failure,
stop the competing workload and rerun preflight. Do not clear the page cache,
delete the NIM model cache, or reboot as routine recovery.

## Guardrails

Generator guardrails are an input-prompt blocklist and an Alibaba Qwen3Guard text guard
(text switch), plus output face detection with blur and a Google SigLIP content check
(video switches). The Reasoner has no NIM-provided guardrails.

| Variable | Current Default | Effect |
| --- | --- | --- |
| `NIM_ENABLE_TEXT_GUARDRAILS` | true | Input-prompt blocklist and text classifier path |
| `NIM_ENABLE_VIDEO_GUARDRAILS` | true | Output image/video face-privacy guardrail path |
| `NIM_ENABLE_SIGLIP_GUARDRAILS` | true | Per-frame safety classifier when output visual guardrails are enabled |

A blocked request returns HTTP 422 and no usable partial output. Text checks run
on the prompt that reaches generation. When prompt upsampling is enabled, that
is the upsampled prompt.

Disabling controls can reduce safety/privacy protections and can violate
deployment policy. Do so only for an approved, isolated diagnostic:

```bash
-e NIM_ENABLE_TEXT_GUARDRAILS=0 \
-e NIM_ENABLE_VIDEO_GUARDRAILS=0 \
-e NIM_ENABLE_SIGLIP_GUARDRAILS=0
```

Disabling video guardrails also disables SigLIP. Disabling SigLIP alone keeps
the face path. Generator bring your own checkpoint (BYOC) does not replace the
NIM-owned guardrails.

## Prompt-Upsampling Diagnostics

Startup fails if the feature is enabled without endpoint, model, key, or
required templates. At request time, external failures log a warning and
generation continues with the original prompt. When debugging, confirm the
following:

- The configured URL is OpenAI-compatible and reaches `/v1/chat/completions`.
- The model supports image input when upsampling image-to-video (I2V).
  Text-to-image (T2I) and text-to-video (T2V) send text only.
- The external secret is present but not logged.
- Timeout and token limits suit the endpoint.
- Provider-specific fields are placed in
  `NIM_PROMPT_UPSAMPLING_EXTRA_BODY` only when that endpoint accepts them.

## Production Checks

Before accepting traffic, complete these checks:

- Require readiness, not only liveness.
- Start with enough free memory on every participating GPU and review any
  profile-fallback warning.
- Record version, active model, profile/checkpoint metadata, and image digest.
- Verify the intended Generator or Reasoner route exists in live OpenAPI.
- Send one small representative request for each enabled capability.
- Confirm output decoding and artifact storage permissions.
- Confirm logs do not contain secrets or unbounded media/request bodies.
- Scrape metrics and test alerts if metrics are part of the SLO.

## Errors

| Runtime | Status | When | Body |
| --- | --- | --- | --- |
| Generator | 422 | Schema, field, or media validation fails, or a guardrail blocks the request | Envelope, `"type": "UnprocessableEntityError"`; a validation `message` is `<field>: <message>`, with multiple errors joined by `; ` |
| Generator, Reasoner | 404 | Missing route, including the other runtime's inference route | Envelope, `"type": "NotFoundError"` |
| Generator | 500 | Unhandled or backend error | JSON object with a string-valued `error` field instead of the standard envelope (below) |
| Reasoner | 400 | Request-shape or sampling-parameter validation fails | Envelope, `"type": "BadRequestError"` |
| Reasoner | 422 | An error whose message concerns video, image, media, decode, or codec | Envelope, `"type": "UnprocessableEntityError"` |

The Generator 500 body is:

```json
{"error": "Internal Server Error. Please see server logs for more details."}
```

Use the HTTP status and stable error-object fields, and retain the returned
message for diagnosis. Do not make automation depend on exact error wording. A
typical NIM error envelope is:

```json
{
  "error": {
    "message": "<description>",
    "type": "<error class>",
    "code": 422
  }
}
```

A wrong-runtime 404 names the selected runtime, its route, and the deployment
the request needs, for example: `The selected runtime is 'generator', which
serves POST /v1/infer. POST /v1/chat/completions requires a reasoner
deployment.`

Task-specific validation belongs to [Generation](generation.md),
[Action](action.md), [Transfer](transfer.md), and [Reasoning](reasoning.md).

## Troubleshooting

### Startup and Deployment

For Kubernetes Pod, volume, and probe failures, refer to [Helm
Troubleshooting](helm.md#troubleshooting).

| Symptom | Likely Cause | Action |
| --- | --- | --- |
| Image pull unauthorized | Docker is not logged in or key lacks repository access | Review the complete [NGC authentication and credential-storage workflow](deployment.md#authenticate-to-ngc), then log in with `NGC_API_KEY` and literal `$oauthtoken` |
| Artifact download fails | Container lacks `NGC_API_KEY`, entitlement, DNS/network, or storage | Verify key injection and NGC connectivity; inspect cache capacity/ownership |
| Cache permission denied | Host mount is not writable by the container | Fix ownership/ACLs and retain a persistent writable `/opt/nim/.cache` |
| No compatible profile | Compute capability, requested precision, total or free per-device VRAM, or system memory (container cgroup or host RAM below the profile's [system-memory floor](support-matrix.md#generator-profiles)) does not fit the selected model/precision/offload; on integrated GPUs, only 70% of the shared pool counts and offload profiles are not eligible; or the NVML/utility-capability availability probe failed | Run the [pre-download profile preflight](deployment.md#run-the-pre-download-profile-preflight), free unrelated GPU memory, raise the container memory limit, remove unnecessary pins, choose a smaller model or compatible precision, or use supported hardware; if the probe failed, verify the driver, Container Toolkit, GPU visibility, and utility capability |
| Preflight passes but cold start fails | Host RAM is too low for the workload, or artifact download/materialization or runtime initialization failed | Preserve the failed container, inspect its logs, and compare host RAM with the support-matrix floors |
| Conflicting selectors | A selector variable such as `NIM_MODEL_VARIANT` disagrees with `NIM_TAGS_SELECTOR` | Set each selector in one place and inspect the full launch environment |
| Visible GPUs sit idle | Selected profile uses fewer GPUs than Docker exposed | Restrict `--gpus` or intentionally pin an available layout matching the desired count |
| Container live but never ready | Cold materialization/build/warmup is still running or failed | Follow logs, cache/NGC/VRAM errors, and extend startup probe budget; do not send inference |
| Port already allocated | Another container uses the host port | Remove the active example container before reusing the default `-p 8000:8000`, or choose another host port and update that client's `NIM_URL` |
| Container name already in use | A previous example container still exists | Inspect its logs if startup failed, then remove it with `docker rm -f cosmos3-generator` or `docker rm -f cosmos3-reasoner` |
| Wrong-runtime 404 | `NIM_URL` reaches Generator for a Reasoner request, or Reasoner for `/v1/infer` | Inspect `/v1/metadata`, then start the intended runtime or correct the URL |
| `/dev/shm` or resource error | Shared memory/ulimits are too small | Apply the documented `--shm-size` and ulimits |

### Generator and Media

For Action and Transfer request failures, refer to [Action Common
Failures](action.md#common-failures) and [Transfer Validation and Common
Failures](transfer.md#validation-and-common-failures).

| Symptom | Likely Cause | Action |
| --- | --- | --- |
| HTTP 422, missing or invalid `model_mode` | The request does not identify a valid Generator task | Set one of the documented top-level modes |
| Selected variant rejects request mode or sampling fields | A specialist received another mode, or a four-step request supplied model-owned controls | Select a compatible `NIM_MODEL_VARIANT`; omit `num_inference_steps`, `guidance_scale`, and `flow_shift` for four-step variants |
| HTTP 422, media decode/fetch | Invalid base64/data URL, URL disabled/unreachable, or unsupported media | Prefer a MIME-aware data URL; check the selected image's codec and format support |
| HTTP 422, frame or resolution | Request violates T2I/video cadence rules or supplies `num_frames` for Action | Recompute with the task tables; Action derives frames from its chunk size |
| Content-policy 422 | Text or generated frames triggered guardrails | Rephrase and review content; disable only under approved diagnostic policy |
| Backend 500/OOM | Profile fit or runtime workload exceeded available memory | Reduce workload/concurrency, choose Nano/offload, or use a larger supported GPU; retain logs |
| Request/client timeout | Image or video generation exceeded client/backend timeout | Refer to [Long-Running Requests](#long-running-requests); inspect NIM logs and runtime metrics |
| MP4 does not play | Player lacks VP9-in-MP4 support | Use `mpv`/`ffplay` or re-encode to H.264 |

### Reasoner

| Symptom | Likely Cause | Action |
| --- | --- | --- |
| Model not found | Client hard-coded the wrong served ID | Discover it through `/v1/models` |
| HTTP 400 request error | Sampling, thinking controls, `top_logprobs`, or extension placement is invalid | Check current ranges, use strict JSON types, and place NIM extensions in `extra_body` |
| HTTP 422 media error | Media content/order/count/preprocessing failed | Put media before text, use data URLs, and check operator media limits |
| Chat Completions route 404 | Client reached Generator or the selected image lacks the route | Confirm Reasoner through `/v1/metadata` and inspect live OpenAPI |
| Responses route 404 | Route disabled, absent, or requested from Generator | Confirm Reasoner metadata, then use Chat Completions and inspect live OpenAPI/operator setting |
| Host OOM or near-zero `MemAvailable` on unified memory | The Reasoner memory share is too large for the shared host/device pool and the host workload | Remove any explicit `NIM_GPU_MEMORY_UTILIZATION` above the `0.70` default, or lower it from host measurements, then restart |
| Structured output is prose or invalid JSON | The request did not enforce a JSON schema | Send a JSON schema in `response_format` (Chat Completions) or `text.format` (Responses). With the catalog, add `--guided-output`. Refer to [Structured Output](reasoning.md#structured-output) |
| KV-cache/context OOM | Context, media tokens, batching, or concurrency is too large | Reduce media FPS and concurrency; the default unchunked multimodal path enforces a 16,384 batched-token minimum (24,300 on Edge), so enable multimodal chunking with `NIM_DISABLE_CHUNKED_MM_INPUT=0` only after validation before lowering that budget |

For DFlash and checkpoint-source failures, refer to [BYOC](#byoc).

### BYOC

Refer to [Bring Your Own Checkpoint Common
Failures](bring-your-own-checkpoint.md#common-failures). A long first start
with a new engine or remote checkpoint is expected. Keep a writable persistent
cache and wait for readiness.
