# Cosmos3 inference-nim — skill notes (not in the cookbook pages)

Skill-authored text of `SKILL.md`, moved here on 2026-11-07 to keep SKILL.md within its 500-line budget when deployment.md's
bash blocks were quoted there (layout change only; wording unchanged apart from lead-ins; § 5 adds the rows of the support
matrix that SKILL.md's floors table does not keep; § 2's `cache_ok` commands were added after eval 2026-11-08). Nothing in this
file is a cookbook page's text unless it is quoted and
names the page, and nothing in it is measured by this skill. The cookbook is
[`../../cookbooks/cosmos3/nim/`](../../cookbooks/cosmos3/nim/README.md). Read this file with SKILL.md before the first command.

## 1. The example clients' inputs are git-LFS objects (SKILL.md "Read this first", item 4)

The clients read their inputs from sibling cookbook directories, not from this skill: the `t2i`, `t2v`, `i2v`, `v2v` and
`*_4step` clients from `../../cookbooks/cosmos3/generator/audiovisual/assets/` (the JSON prompts and negative prompts are plain
files; the JPEG images and MP4 video are git-LFS objects), the `reasoner`, `reasoner_responses` and `streaming` clients from
`../../cookbooks/cosmos3/reasoner/assets/` (every file there is a git-LFS object), the `action` client from
`../../cookbooks/cosmos3/generator/action/assets/`, the `transfer` client from `../../cookbooks/cosmos3/generator/transfer/assets/`
(plus the audiovisual car-driving video for derived controls). In a checkout made without git-lfs those media files are
~130-byte pointer files (first line `version https://git-lfs.github.com/spec/v1`), not media. Before running a client,
materialise exactly the files it reads: `git lfs pull --include=<path from the repository root>` when git-lfs is installed,
otherwise download the blob from `https://media.githubusercontent.com/media/NVIDIA/cosmos-framework/main/<path from the repository root>`
(checked 2026-11-07 for `../../cookbooks/cosmos3/reasoner/assets/robot_153.jpg`: the response is the file itself,
`Content-Type: image/jpeg`, ETag = the pointer's `oid sha256`, length = the pointer's `size`). The clients write next to
themselves ("Examples save outputs under the ignored `examples/outputs/` directory", README): that directory is created by
the first client run (`OUTPUT.parent.mkdir(...)` in [`t2i.py`](../../cookbooks/cosmos3/nim/examples/t2i.py), `output_dir.mkdir(parents=True, ...)`
in [`reasoner.py`](../../cookbooks/cosmos3/nim/examples/reasoner.py)) and is ignored by `.agents/skills/.gitignore`, so a fresh checkout has none — copy what a task asks for
to the location it names.

## 2. Headless additions for an unattended deployment (SKILL.md "Readiness and selection")

None of this is in deployment.md. The eval cells run unattended, from a container that shares the host's Docker daemon.

- **Readiness loop**: the page's `until curl -fsS "$NIM_URL/v1/health/ready" >/dev/null; do sleep 10; done` never exits when
  the container dies. Add a container-exit check (`docker inspect -f '{{.State.Running}}' <name> | grep -q true`) and a
  deadline, print `docker logs --tail 160 <name>` on either, and record the time-to-ready.
- **Logs**: the cookbook's cleanup (`docker rm -f cosmos3-generator`, `docker rm -f cosmos3-reasoner`) discards the container's
  logs. Save them first — `docker stop <name>; docker logs <name> > <OUT>/<name>.log 2>&1; docker rm <name>` — a stopped
  container still holds its name (operations.md "Container name already in use") until the `rm`.
- **Cache mount from a container**: the Docker daemon resolves every `-v` path on the host and creates a missing bind-mount
  source itself, root-owned, while the page's `chmod -R a+rwX "$LOCAL_NIM_CACHE"` runs inside your own container. The NIM runs
  as UID/GID 1000 (deployment.md "Prepare the Model Cache"; operations.md "Cache permission denied": "Host mount is not
  writable by the container" → "Fix ownership/ACLs"), so after the cache block and before EITHER launch (Generator and Reasoner
  alike) run, with `NIM_IMAGE` and `LOCAL_NIM_CACHE` exported:

  ```bash
  cache_fix() { docker run --rm --user 0 --entrypoint /bin/sh -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" "$NIM_IMAGE" -c 'chown -R 1000:1000 /opt/nim/.cache && chmod -R u+rwX /opt/nim/.cache'; }
  cache_ok() { docker run --rm --entrypoint /bin/sh -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" "$NIM_IMAGE" -c 'mkdir -p /opt/nim/.cache/.w && touch /opt/nim/.cache/.w/f && rm -r /opt/nim/.cache/.w && [ -z "$(find /opt/nim/.cache ! -writable -print -quit)" ]' 2>/dev/null; }
  cache_fix; cache_ok || { echo "LOCAL_NIM_CACHE=$LOCAL_NIM_CACHE is still not writable by uid 1000 after the repair: use a directory of your own"; exit 1; }
  ```

  Both lines use the already-pulled NIM image (its default user is uid 1000 and it has `/bin/sh`); they are the `cache_ok` test and
  repair of `../../cosmos3-inference-generate-media/SKILL.md` Variant 7 minus its `NIM_CACHE_REPAIR` opt-in; the repair now runs first and
  unconditionally (idempotent `chown -R 1000:1000`, operations.md "Fix ownership/ACLs") and the test then creates a directory and a file
  and walks the tree (`find ! -writable`): a cache the daemon created root-owned, or populated by an earlier run as another uid, passed
  the old single `touch` and still killed the launch with `Permission denied` (two cells, eval 2026-11-09). A launch without a passing `cache_ok` dies within a minute with
  `ManifestDownloadError … Permission denied (os error 13)`; do not drop the `-v` mount instead ("Keep the cache between runs to
  avoid repeated downloads", deployment.md; "retain a persistent writable `/opt/nim/.cache`", operations.md).
- **Eval-verified form**: the Variant 7 block of [`../../cosmos3-inference-generate-media/SKILL.md`](../../cosmos3-inference-generate-media/SKILL.md)
  is the Generator launch with the three additions above written out as one script (its readiness loop has a 40-min deadline)
  plus that skill's guardrail switches. For the Reasoner, the command stays SKILL.md's quoted Nano Reasoner block — add only the
  three additions above around it (no `NIM_PERF_PROFILE`; no guardrail switches: "The Reasoner has no NIM-provided guardrails", operations.md).

## 3. Output checks (SKILL.md "Verify the output")

The tool commands are skill additions that use only the pinned client environment (`pillow==12.1.0`, `av==16.1.0` in
`../../cookbooks/cosmos3/nim/pyproject.toml`) or the optional FFmpeg client tooling of prerequisites.md; run them from
`.agents/skills/cookbooks/cosmos3/nim`.

- Image: `uv run python -c "from PIL import Image; p='examples/outputs/t2i_robot_draping.jpg'; Image.open(p).verify(); im=Image.open(p); print(im.format, im.size)"`
  — expect `JPEG (960, 960)` for the `t2i` client, whose request sends `resolution` `720_1_1` (generation.md: "`720_1_1` (960 × 960)").
- Video: `uv run python -c "import av,sys; c=av.open(sys.argv[1]); s=c.streams.video[0]; n=sum(1 for _ in c.decode(s)); print(s.codec_context.name, s.width, s.height, n, float(s.average_rate))" examples/outputs/t2v_robot_kitchen.mp4`
  — expect `vp9` with the request's size and frame count (video defaults, generation.md: "`720` (1280 × 720)", `num_frames` 189, `fps` 24.0).
- Playback (generation.md "Output and Playback"): "VP9-in-MP4 is not supported by every browser or stock player. The declared
  FFmpeg client package includes `ffplay` for direct playback" — `ffplay examples/outputs/t2v_robot_kitchen.mp4`; "For a broadly
  compatible H.264 copy":
  `ffmpeg -i examples/outputs/t2v_robot_kitchen.mp4 -c:v libx264 -crf 18 -pix_fmt yuv420p examples/outputs/t2v_robot_kitchen-h264.mp4`.
- Reasoner artifacts: [`reasoner.py`](../../cookbooks/cosmos3/nim/examples/reasoner.py) writes `request.json`, `response.json`, `output.txt` (the answer from
  `message.content`), `validation.json` and `report.md` (its sections: the case header, "## Prompt", "## Final answer",
  "## Qualitative review") under `examples/outputs/reasoner_<case>/`, plus `output.json` and `annotated.png` for structured
  and spatial cases.

## 4. uv (SKILL.md "Read this first", item 5)

The client project has its own `[tool.uv]` table (`package = false`), so the framework's `[tool.uv] required-version = "==0.11.29"`
pin (which `../../cosmos3-inference-generate-media/SKILL.md` has to work around) does not apply inside `.agents/skills/cookbooks/cosmos3/nim`:
`uv sync --locked` there resolves with the stock uv the prerequisites install (checked 2026-11-07 with uv 0.13.0, `--dry-run`).
Do not install uv 0.11.29 for the client environment.

## 5. Hardware floors, as the support matrix states them (SKILL.md "Hardware floors")

Quoted from [`support-matrix.md`](../../cookbooks/cosmos3/nim/support-matrix.md) ("Generator Profiles", "Reasoner Profiles";
"Memory requirements use binary GiB: VRAM floors are per device, and system-memory floors are per host") — no figure here is
measured by this skill. SKILL.md keeps the single-GPU rows; this table holds the remaining rows of the page's two tables (the
`super` Generator rows "also apply to `super-t2i`, `super-t2i-4step`, `super-i2v`, and `super-i2v-4step`").

| Runtime   | `NIM_MODEL_VARIANT`         | Precision | Layout, as the page states it                            | Minimum VRAM per device | System-memory floor | Transfer minimum VRAM per device |
| --------- | --------------------------- | --------- | -------------------------------------------------------- | ----------------------: | ------------------: | -------------------------------: |
| Generator | `edge-droid` (experimental) | BF16      | Model offload None, guardrails Resident; 1, 2, 4, 8 GPUs |                  15 GiB |              16 GiB |                    Not supported |
| Generator | `nano`                      | BF16      | Model offload Layer, guardrails Offloaded; 1 GPU         |                  31 GiB |              64 GiB |                           35 GiB |
| Generator | `nano`                      | FP8       | Model offload Model, guardrails Resident; 1 GPU          |                  38 GiB |              36 GiB |                           44 GiB |
| Generator | `nano`                      | FP8       | Model offload Layer, guardrails Resident; 1 GPU          |                  32 GiB |              36 GiB |                           38 GiB |
| Generator | `nano`                      | FP8       | Model offload Layer, guardrails Offloaded; 1 GPU         |                  31 GiB |              64 GiB |                           35 GiB |
| Generator | `super`                     | BF16      | Model offload None, guardrails Resident; 1, 2, 4, 8 GPUs |                 150 GiB |             112 GiB |                          160 GiB |
| Generator | `super`                     | BF16      | Model offload Model, guardrails Resident; 1 GPU          |                  93 GiB |             150 GiB |                           99 GiB |
| Generator | `super`                     | BF16      | Model offload Layer, guardrails Resident; 1 GPU          |                  42 GiB |             150 GiB |                           50 GiB |
| Generator | `super`                     | FP8       | Model offload Model, guardrails Resident; 1 GPU          |                  64 GiB |              80 GiB |                           76 GiB |
| Generator | `super`                     | FP8       | Model offload Layer, guardrails Resident; 1 GPU          |                  38 GiB |              80 GiB |                           50 GiB |
| Reasoner  | `super`                     | BF16      | 1 GPU, TP 1                                              |                 135 GiB |              72 GiB |                                — |
| Reasoner  | `super`                     | BF16      | 2 GPUs, TP 2                                             |                  73 GiB |              72 GiB |                                — |
| Reasoner  | `super`                     | NVFP4     | 1 GPU, TP 1                                              |                  73 GiB |              33 GiB |                                — |

"The 31-GiB Nano layer-offload rows target 32-GB client GPUs such as the GeForce RTX 5090, which does not meet their 35-GiB
Transfer minimum." "Leave offload and guardrail residency on automatic selection unless a configuration is validated for the
deployment." Compute-capability floors ("GPU Architecture and Topology"): BF16 8.0, FP8 8.9, NVFP4 10.0. "On a discrete GPU,
Reasoner measures free memory after a 4-GiB runtime reserve (`NIM_GPU_MEMORY_HEADROOM_GIB`)." The unified-memory thresholds, the
tested GPU inventory and "Transfer Headroom" are on the page.
