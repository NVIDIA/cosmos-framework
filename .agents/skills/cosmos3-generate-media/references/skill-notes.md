# Cosmos3 generate-media — skill notes (not in the notebooks)

Cross-variant text of `SKILL.md`, moved here on 2026-10-27 to keep SKILL.md within its 500-line budget (layout change only;
wording unchanged apart from pointers and lead-ins that had to follow the text). Nothing in this file comes from the cookbook notebooks: every figure is labelled with the eval run that
measured it. Read it together with SKILL.md before the first command; SKILL.md keeps a pointer at each former location.

## 1. Payload rules of the six `create_payload` notebooks (SKILL.md intro and "Bundled assets")

NIM prompt form (SKILL.md intro): (the NIM cookbook's page examples use plain-English captions, but its own `examples/i2v.py`
sends the bundled assets as compact JSON via `common.py::compact_json_file`; `/v1/infer` accepts the bundled structured-JSON assets serialised compactly — `prompt` max
20,000 chars; accepted with HTTP 200 on the 1.0.0 image, same documented limit on 2.1.0).

Rules of the six `create_payload` notebooks (payloads.md § 3.3, § 7, § 9.2; the
NIM cookbook's requests are inline literals, but the same compact-JSON string is
what to send as its `prompt` / `negative_prompt`): the prompt is the asset's JSON serialised compactly
(`json.dumps(asset, ensure_ascii=True, separators=(",", ":"))`); for `text2image`
the framework, diffusers, vLLM-Omni and SGLang payloads carry `"negative_prompt": ""`
(TensorRT-LLM leaves the key out) and the framework, diffusers and distilled
payloads set `num_frames = 1` (TensorRT-LLM drops `num_frames` / `fps`; vLLM-Omni
and SGLang send no frame fields); video modes load `assets/negative_prompts/<mode>/neg_prompt.json`
(video2video reuses the image2video file); the request size comes from the
`resolution` / `aspect_ratio` table, not from the asset's own `resolution`.

## 2. Measured footprints (SKILL.md "Model sizes vs GPU")

Measured by this skill's eval runs on one H100 80 GB (2026-10-21/23; not from the
notebooks): Edge — diffusers t2i 832x480 ~9.0 GiB peak allocated, 3–5 s; diffusers t2v
121 frames ~12.4 GiB allocated / 15.2 GiB reserved, 16–23 s; framework t2i 9,735–9,785 MiB
(nvidia-smi); TensorRT-LLM 17,633–17,770 MiB incl. its 832x480x121 warm-up, ready in
40–50 s, 640x640 t2i ~5.5 s. Nano — ~31.5 GB of bf16 weights (30.56 GiB resident after
load); diffusers 720x1280 t2i 33.8 GiB peak allocated / 35.4 GiB reserved (36,939 MiB
nvidia-smi); framework t2i 35,973 MiB; TensorRT-LLM with the shipped `cosmos3-nano-1gpu.yaml`
50,623–52,147 MiB after the default 720x1280x189 warm-up, `/health` in ~110–180 s, 1280x720
t2i ~6 s. Every Nano path fits an 80 GB GPU in bf16 as shipped — do not tune or offload
first. For GPUs under ~40 GB use the fp8 revision (cosmos-framework.md "Nano FP8: Text to
Image"; the NIM serves fp8 by default). "8B Nano" was the retired Generator NIM 1.0.0 table's label (setup guide @ 6e2c69af) and does not
describe the bf16 footprint (one eval agent counted ~15 B transformer parameters incl.
the understanding tower).

## 3. Disabling the guardrail, per backend

| Backend          | How to disable (README / notebook wording; payloads.md § 2.3)                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| diffusers        | `enable_safety_checker=False` (the notebook's `get_pipe` passes `safety_checker=None, enable_safety_checker=True` to keep it enabled)                                                                                                                                                                                                                                                                                                                                                                                               |
| cosmos_framework | `--no-guardrails`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| vLLM-Omni        | `--no-guardrails` on `vllm serve` (server-wide), or `guardrails: false` in `extra_params` / `extra_args` (per request). The server loads the gated guardrail at startup; without Hugging Face access to it the server exits before serving, and the per-request switch does not fix that                                                                                                                                                                                                                                            |
| SGLang           | `-e SGLANG_DISABLE_COSMOS3_GUARDRAILS=1` on `docker run` (server-wide; required for smoke runs — otherwise the server loads the gated guardrail at startup and exits with `GatedRepoError 401`, like vLLM-Omni without `--no-guardrails`), plus `guardrails: false` in `extra_params` per request (the notebook's helpers send `True`, sglang.md § 6.2; the per-request flag alone does not prevent the startup exit). Not in the notebook or README: verified against `lmsysorg/sglang:dev` (cosmos3_pipeline.py) by the eval runs |
| TensorRT-LLM     | `TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1` before starting the server, or `use_guardrails` set to `False` in the request `extra_params`                                                                                                                                                                                                                                                                                                                                                                                                  |
| NIM              | `NIM_ENABLE_TEXT_GUARDRAILS=0 NIM_ENABLE_VIDEO_GUARDRAILS=0 NIM_ENABLE_SIGLIP_GUARDRAILS=0` (2.1.0 README; "Disabling video guardrails also disables SigLIP", nim/operations.md)                                                                                                                                                                                                                                                                                                                                                    |

## 4. Things not obvious from the docs

- **Guardrail gating**: default server configs load the access-gated
  `nvidia/Cosmos-1.0-Guardrail`; smoke runs should disable guardrails (see
  flags above) rather than acquire access. SGLang's switch is
  `-e SGLANG_DISABLE_COSMOS3_GUARDRAILS=1` on `docker run` (eval-verified against
  `lmsysorg/sglang:dev`; not in the notebook/README).
- **Server-mode variants are client/server**: vLLM/SGLang/TRT-LLM/NIM need
  the server started, health-polled, and torn down by whoever automates them.
  Shared headless pattern (skill addition, not in the notebooks):

  ```bash
  # Docker variants (Variant 3 vLLM-Omni shown; Variant 4 SGLang: NAME=cosmos3-sglang, READY=http://localhost:30000/health)
  NAME=cosmos3-vllm; READY=http://localhost:8000/v1/models
  docker run -d --name "$NAME" --runtime nvidia --gpus all ...   # rest of the notebook command unchanged
  until curl -sf "$READY" >/dev/null; do
    docker inspect -f '{{.State.Running}}' "$NAME" | grep -q true || { docker logs --tail 80 "$NAME"; exit 1; }
    sleep 10
  done
  # ... requests ...
  docker logs "$NAME" > server.log 2>&1; docker stop "$NAME" && docker rm "$NAME"
  ```

  ```bash
  # trtllm-serve variants (5 and 6); run from a script (job control off) so $! is the process-group leader.
  # Launch with `setsid … &` from the script itself. A server started inside a backgrounded subshell `( … ) &` dies when
  # the agent's tool kills that command at its timeout (one eval run, 2026-10-24, lost the offload server this way after
  # 60–120 s; nohup+disown also survived it) — setsid puts the server in its own session and process group, out of reach.
  export TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1
  setsid trtllm-serve nvidia/Cosmos3-Edge --port 8000 > serve.log 2>&1 &   # Nano: add its --visual_gen_args as documented in trt-llm.md § 7.7; distilled: run the Variant 6 launch script, do not compose this line
  PID=$!
  until curl -sf http://localhost:8000/health >/dev/null; do kill -0 "$PID" 2>/dev/null || { tail -50 serve.log; exit 1; }; sleep 10; done
  # ... requests (check_trtllm_server / post_image) ...
  kill -INT -- "-$PID" 2>/dev/null || echo "server already gone — see serve.log"   # SIGINT the process group: 'Application shutdown complete', GPU back to 0 MiB in 5-8 s
  for _ in $(seq 30); do kill -0 -- "-$PID" 2>/dev/null || break; sleep 1; done   # poll the GROUP: the leader can exit while a worker in its group keeps the GPU
  # group still alive 30 s after SIGINT (one eval run, 2026-10-24, needed this): escalate to TERM, then KILL. Every kill is
  # guarded (`|| true`): under `set -e` an unguarded kill on a group that died meanwhile would abort the script here
  kill -0 -- "-$PID" 2>/dev/null && { kill -TERM -- "-$PID" 2>/dev/null || true; sleep 10; kill -0 -- "-$PID" 2>/dev/null && kill -KILL -- "-$PID" 2>/dev/null || true; } || true
  wait "$PID" 2>/dev/null || true
  # leftovers — in two eval runs the recorded PID's group was not the whole server (a setsid leader that was not the
  # trtllm-serve process; a worker outside the group that survived the group's SIGKILL) and the GPU stayed occupied:
  nvidia-smi --query-gpu=memory.used --format=csv,noheader; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader   # must read 0 MiB and list nothing (shared GPU: your pre-launch figure, none of your PIDs); survivors: kill -KILL them by PID
  pgrep -fl '[t]rtllm-serve' || true   # the server by name, if still there ([t]: the line never matches itself; ignore your own shell if listed; workers are multiprocessing children without the name)
  ```

  (interactive shell with job control: drop `setsid`, each background job already
  owns its group.) Measured on one H100 80 GB (eval runs 2026-10-21/23), not a
  notebook figure: Edge ready 40–50 s, peak ~17.7 GB (17,633–17,770 MiB), 640x640
  t2i ~5.5 s; Nano ready ~110 s with cached weights (~180 s incl. download),
  ~50–52 GB (50,623–52,147 MiB) peak with the default 720x1280x189 warm-up, 1280x720 t2i
  ~6 s; distilled (bf16 + CPU offload) ~250 s, distilled patched `fp8` branch ~60 s (two runs, 2026-10-24: 125.6 GiB PCIe host and 251 GiB SXM host, 70,587–70,983 MiB).
- **Checkpoint cache**: the notebooks `export HF_HOME=/path/to/large/huggingface/cache`
  (setup guide: "Nano downloads plus CUDA dependencies can take tens of GiB"); the
  vLLM-Omni container sees it only through the `-v "${HF_HOME}:/root/.cache/huggingface"`
  mount, SGLang mounts its own `SGLANG_HF_CACHE`. Every `-v` cache path (`HF_HOME`,
  `SGLANG_HF_CACHE`, `LOCAL_NIM_CACHE`) is resolved by the Docker daemon, not by your
  shell; only the NIM runs as a non-root user (uid 1000), so only its mount needs the
  preflight of Variant 7.
- **NIM (Variant 7, `nvcr.io/nim/nvidia/cosmos3:2.1.0`)**: the detailed pages are the NIM cookbook in this checkout
  (`../../cookbooks/cosmos3/nim/`); what this skill adds to them — launch detached without `--rm` (as the cookbook does) and save
  `docker logs` to `<OUT>/cosmos3_generator_nim/container.log` before `docker rm` (a stopped container still holds the name); the
  uid-1000 cache preflight above (the first 1.0.0 launches failed with `ManifestDownloadError … Permission denied (os error 13)`
  when the daemon created the mount root-owned); selectors `NIM_MODEL_VARIANT` / `NIM_PRECISION` / `NIM_PERF_PROFILE` checked by the
  cookbook's `profile_selection.startup` preflight (measured 2026-10-27 on an H100 PCIe 80 GB: image pull 163 s, preflight 17 s →
  nano / fp8 / latency, resident guardrails, floors 44 GiB VRAM / 35 GiB RAM); pins (`NIM_TAGS_SELECTOR` keys `model_variant`,
  `precision`, `profile`, …; `NIM_MODEL_PROFILE=<id>`) are "for validated needs only"; the 1.0.0-era `NIM_MODEL_SIZE`, `model_size=`
  tag and `list-model-profiles` command are no longer documented (the binary still ships). The compact-JSON `prompt` /
  `negative_prompt` (8,151 / 14,674 chars, under the 20,000 limit) were accepted with HTTP 200 on the 1.0.0 image (sent as `image`
  then), not yet re-verified on 2.1.0.
- **`LD_LIBRARY_PATH=` (empty)** is prefixed to every `torchrun` run cell of the
  cosmos_framework notebook (`CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" LD_LIBRARY_PATH= "$COSMOS3_UV_ENV/bin/torchrun" ...`).
- **Benign diffusers load warnings** (measured in the eval smoke runs 2026-10-21 with
  git-main diffusers 0.42.0.dev0; not a notebook fact): `from_pretrained(..., torch_dtype=...)`
  prints a `torch_dtype is deprecated` FutureWarning (keyword kept as the README/notebook
  write it), and `nvidia/Cosmos3-Edge` prints `config attributes {'backbone_type': 'cosmos3_edge_nemotron_dense', 'temporal_compression_factor': 4} ... will be ignored`;
  generation is unaffected — do not treat either as an error.
- **Measured Edge footprint** (one H100 80 GB, bf16, torch 2.14.1+cu130; eval runs
  2026-10-21, not a notebook figure): load ~7.6 GiB allocated; t2i 832x480 / 20 steps
  ~9.0 GiB peak allocated, 3–5 s; t2v 121 frames ~12.4 GiB peak allocated (15.2 GiB
  reserved), 16–23 s.

## 5. Expected smoke facts (SKILL.md "Verify the output")

- Expected smoke facts (measured on one H100 80 GB, eval runs 2026-10-21/23): diffusers Edge
  t2v → h264, 832x480, 24/1, 121 frames, 5.041667 s; NIM resolution `"256"` / 25-frame
  request → vp9, 320x192, 25 frames, ~1.04 s (measured on the 1.0.0 image; checked on the host, never inside the NIM
  container — Variant 7). Record them in `result_summary.md`.

## 6. uv version detail (SKILL.md "Read this first", item 3)

Do NOT install uv 0.11.29 for Variants 1, 5 and 6 (standalone venvs): it cannot resolve tensorrt_llm's
pre-release dependency `nvidia-cutlass-dsl==4.8.0.dev0` without `--prerelease=allow`,
which then also pulls `mpmath 1.5.0a1` and breaks the import (`ImportError: cannot
import name 'bitcount' from 'mpmath.libmp'`; eval run 2026-10-24) — 0.11.29 is for
`uv sync` in Variant 2 only.
