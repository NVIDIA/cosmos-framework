# Cosmos inference services

Use the existing `cosmos3-inference` skill. These services are Framework-owned
adapters; setup, approval, job records, and platform monitoring remain in
`../execution/`. Docker, SLURM, Kubernetes, Brev, and virtualenv are available
when their GPU/network prerequisites fit the selected service.

## Reasoner (Cosmos3 Nano / Edge and compatible Qwen checkpoints)

Install Framework's `workflows` extra into the selected GPU environment, from
the repository root:

```bash
uv sync --extra workflows --group=cu130   # CUDA 12.8: --group=cu128
```

Use `--group=cu130-train` instead when the same environment also trains. Prepare
a complete Hugging Face export first: use the reasoner checkpoint helpers for
DCP and PEFT inputs. Do not point the HTTP server at an unmerged adapter.

```bash
cosmos-reasoner-serve --model-path /models/reasoner --model-name cosmos \
  --media-root /data --host 127.0.0.1 --port 8080
```

`/health`, `/info`, `/v1/models`, `/infer`, and
`/v1/chat/completions` are implemented locally in
`cosmos_framework/inference/reasoner/service.py`. Chat requests accept text,
`image_url`, and `video_url` content parts, including base64 data URIs.
Only explicitly allowed media roots are readable. Remote media requires
`--allow-remote-media` and trusted callers (it grants outbound network access).
Requests are serialized; streaming is rejected. Sampling and vision options
are passed through to the native reasoner runtime.

## Deployment and safety

Bind to loopback by default. If exposing a container port, bind inside the
container to `0.0.0.0` and publish only to a controlled host interface. Configure
authentication, TLS, request limits, and a restrictive egress policy at the
ingress before exposing any endpoint to untrusted clients. These adapters do
not implement authentication. Do not bake tokens into images or CLI arguments.

Preview the concrete image/environment, model, source revisions, mounts,
ports, GPU allocation, estimated runtime, and output location before launch.
Use the selected platform's four verbs; do not infer liveness from job records.
CPU request-contract tests do not establish GPU/model compatibility.
