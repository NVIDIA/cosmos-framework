# The Windows-on-Arm environment, pin by pin

Everything here was validated on RTX Spark N1X units (Windows 11 on Arm, GB10-class GPU with
unified memory) between September and November 2026, on native Windows: no WSL, no containers,
no emulation. The x64 Windows route (RTX PRO 6000) shares the framework-side guards but not the
wheels; it is not described here.

## Interpreter

ARM64 CPython 3.13, managed by uv. Ask for the Arm64 build by name,
`uv python install cpython-3.13-windows-aarch64-none`, and create the venv from the path that
`uv python find cpython-3.13-windows-aarch64-none` prints. A bare `3.13` is not enough: an x64
build of uv, which is what a Windows-on-Arm machine often has, resolves it to x64 CPython.
`platform.machine()` must print `ARM64`. An x64 CPython under emulation is the most common
first-day mistake: everything installs, `torch.cuda.is_available()` is False, and pip pulls the
CPU-only `win_arm64` torch from PyPI if torch is not installed first.

uv keeps managed interpreters and its cache under `%APPDATA%` and `%LOCALAPPDATA%`. A packaged
(MSIX) application, the Claude desktop app for one, redirects those folders for the processes it
starts, and `uv python install` then fails with `Missing expected target directory for Python
minor version link`. Set `UV_PYTHON_INSTALL_DIR` and `UV_CACHE_DIR` to ordinary folders first;
Block 1 puts both under `$ROOT`.

Create the venv with `--seed` so pip exists, and `--no-config` so a cosmos-framework checkout's
`[tool.uv]` pin does not apply to this standalone environment.

## Torch

Install torch before the dependency file, from NVIDIA's Windows-on-Arm index, with
`--only-binary=:all:` so nothing is built from source by accident:

| Package     | Pin            | Where                                                                                                             |
| ----------- | -------------- | ----------------------------------------------------------------------------------------------------------------- |
| torch       | `2.15.0+cu134` | `https://pypi.nvidia.com/nvtorch_oot/` (stable; what the PyTorch "additional platforms" page lists for RTX Spark) |
| torchvision | `0.29.0+cu134` | same                                                                                                              |
| torchaudio  | `2.11.0+cu134` | same                                                                                                              |

Skill note (not in the source): the earlier validation stack used the nightly
`torch 2.15.0.dev20260902+cu134` from `https://pypi.nvidia.com/nvtorch_oot_nightly/`, paired with
a TorchCodec built against it. The stable pin above is the target for the hackathon stack;
`validation.md` says which pairing each number was measured on. Nightlies after 2026-11-04
dropped DLL preloading and need `torch\lib` on the DLL search path, which the `.pth` helper below
provides anyway.

## The dependency file

`assets/requirements-windows-arm64.txt` is the framework's dependency set as pinned for
Windows, with `platform_machine` markers:

- skipped on ARM64, no `win_arm64` wheel: `pyarrow`, `obstore`, `datasets` (needs pyarrow);
  `duckdb` is installed in pyarrow's place for the mining scripts of the TAO skills;
- `av`: the framework's pin 18.0.0 has no `win_arm64` wheel, so ARM64 gets PyAV 19.0.2, an
  abi3 wheel that installs on 3.13 (`qwen-vl-utils` requires PyAV in any case, so pip installs
  it with or without the line). The reasoner path does not import it;
  `scripts/make_smoke_set.py` encodes the smoke clips with its bundled libx264 when the machine
  has no ffmpeg executable, which is the normal case with a self-contained TorchCodec wheel.
  `pip check` then reports three lines for cosmos-framework (obstore and pyarrow not installed,
  `av<19` not met); all three are expected. `opencv-python` and `decord` have no `win_arm64` wheels (tester
  report, 2026-11-06): an external harness that imports them needs a PyAV or TorchCodec reader
  in their place. The Framework evaluator needs none of the three;
- the x64 torch lines are marked out; torch comes from the index above.

With PR #303 the framework imports pyarrow optionally, so the reasoner config graph imports
without it; Parquet features fail where they are used, with an `AttributeError` that names the
missing package (the stand-in for the module raises it on first attribute access).

## The framework

`main` with PR #303 ("make training importable and runnable on native Windows"). Until it
merges: clone `arif-ahmed-nv/cosmos-framework`, fetch `windows-compat` and check out commit `985d116`
(the PR head of 2026-11-17: `main` at 0822f1d plus the Windows commits). The branch itself moves while
the PR is in review, so Block 2 pins the commit. The skill's blocks were run on `e682c47`, an earlier
state of the same PR, and the PR's unit tests on `c8441aa`; `985d116` is `c8441aa` rebased onto `main`
and has not been run on Windows yet (`validation.md` § Lifecycle on PR #303). At this commit the
evaluator imports `toml`, which the dependency file therefore lists. Install editable with `--no-deps` (the dependency file
already satisfied everything). Checkouts without #303 fail on this platform at import (pyarrow),
at the first training step (attention backend) or at the first checkpoint save (statvfs, libuv,
pinned staging); do not patch them, update.

The repository tracks cookbook media with Git LFS. This skill needs none of it, and a Git
without the `git-lfs` binary (MinGit, some Arm64 installs) fails the checkout with
`git-lfs: command not found`. Block 2 clones with the LFS filter switched off for that
repository (`git clone -c filter.lfs.smudge= -c filter.lfs.process= -c filter.lfs.clean=
-c filter.lfs.required=false ...`); the settings are stored in the clone, later checkouts
honour them, and the LFS files stay as small pointer files.

## TorchCodec

The trainer decodes video on the GPU through TorchCodec and refuses CPU fallback; a wheel for
your exact torch is mandatory. Three routes, in order of preference:

1. **A wheel built for your torch (win_arm64, CUDA 13.4, cp313).** NVIDIA's Windows-on-Arm
   PyTorch team builds TorchCodec wheels in its CI (internal distribution today; public
   distribution through the same channel as torch is in preparation). The wheel validated with
   `torch 2.15.0+cu134` is `torchcodec-0.18.0.dev20261005+cu134-cp313-cp313-win_arm64.whl`.
   It is self-contained: `site-packages\torchcodec.libs` carries the FFmpeg 8 libraries
   (`avcodec-62.dll`), libjpeg, libpng, zlib and nvJPEG, so no FFmpeg or GStreamer install is
   needed. The `dev20261105` build bundles FFmpeg 9 and decodes as well; no training was run
   with it. Install with `--no-deps` and verify with
   `from torchcodec.decoders import VideoDecoder`. For any other build, check before you rely on
   it: `Get-ChildItem $VENV\Lib\site-packages\torchcodec.libs\avcodec-*.dll`. No match means
   the wheel does not bundle FFmpeg: install the FFmpeg shared build of the major version the
   wheel was built for and register its `bin` with the `.pth` helper (`--extra-dll-dir`,
   `--force`).
2. **Build from source** against the installed torch: clone `pytorch/torchcodec`, install the
   FFmpeg 9 Windows ARM64 shared LGPL build
   (`https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n9.0-latest-winarm64-lgpl-shared-9.0.zip`;
   the validated build below used the GPL variant of the same release, the LGPL one is untested)
   and the GStreamer 1.28 MSVC ARM64 runtime (for `jpeg8.dll` and `png16.dll`), build with CUDA
   enabled, and register FFmpeg's `bin` and GStreamer's `msvc_arm64\bin` with the `.pth` helper.
   The validated build was `torchcodec 0.17.0a0` with CUDA 13.4 NVDEC video, nvJPEG, CPU JPEG
   and PNG; WebP, AVIF, GIF and HEIC omitted. Budget an hour.
3. Internal NVIDIA users: the prebuilt wheel of the earlier validation (`0.17.0a0+1ebfad0`,
   built against the 2026-09-02 nightly) exists; it pairs only with that torch.

## The DLL search path

Python 3.13 on Windows does not resolve extension-module DLL dependencies from `PATH`.
`scripts/configure_windows_arm64_dll_paths.py --venv <venv>` writes one `.pth` file into the
venv's site-packages that calls `os.add_dll_directory` for torch's `lib`, the CUDA ARM64 `bin`
and cuDNN directories if a toolkit is installed, GStreamer's `msvc_arm64\bin` if present, and
anything passed with `--extra-dll-dir`, and keeps the handles alive in every interpreter the
venv starts, including spawned workers and console scripts. Only torch's `lib` is required: the
helper stops if it is missing and otherwise writes the file, with a note when no GStreamer
directory was found. That is the normal case with a CI-built TorchCodec wheel. A source-built
TorchCodec needs FFmpeg's `bin` and GStreamer's `msvc_arm64\bin`: pass both with
`--extra-dll-dir`, and `--force` to replace an existing file. An import failure for
`libtorchcodec_image.dll` means one of those directories is missing from the file.

## The model

`nvidia/Cosmos3-Nano` is a public Hugging Face repository. When checked on 2026-11-06 it was
not gated (`gated: false` in the Hub API), so there is no acceptance step. Export `HF_TOKEN`
anyway if you have one: authenticated downloads are throttled less and the commands keep
working if a gate is added.

The release is about 33 GB. A tester measured 0.8 to 4 MB/s, which is hours:
where several machines are set up at once, stage the release, or the 16 GB converted directory
that every later step reads, on a share or a drive and copy it. The generator-only folders
`vae/`, `sound_tokenizer/`, `assets/` and `images/` (about 3.3 GB) can be left out with
`hf download nvidia/Cosmos3-Nano --local-dir <dir> --exclude "vae/*" --exclude "sound_tokenizer/*"
--exclude "assets/*" --exclude "images/*"` (one `--exclude` per pattern). Validated on
2026-11-10 (`validation-spark-a.md`): a conversion from a release without those four folders
gave 750 tensors in 4 shards, every file byte-identical to the conversion from the full
release. The saving is the 3.2 GiB of the download and nothing more: the converter fetches its
own VAE and sound tokenizer from the Hub in either case.

The converter (`python -m cosmos_framework.scripts.convert_model_to_vlm_safetensors --checkpoint-path
<release dir> -o <VLM dir>`) runs on the CPU, overlays the Cosmos3 vision tower and language
model onto a Qwen3-VL-shaped HF directory. It fetches what it needs from the Hub through a
nested `uv run ... hf download`, into the default Hugging Face cache: the complete
`Qwen/Qwen3-VL-8B-Instruct` repository (16.3 GiB of weights, not only the processor), the
Wan2.2 VAE (2.8 GB) and the AVAE sound tokenizer (2 GB, kept as a blob and a converted copy).
That fetch is the slow part: 63 minutes at 5 MB/s on a machine that had just pulled the release
at 114 MB/s, unauthenticated and without `hf_xet`. Stage the cache folder
`models--Qwen--Qwen3-VL-8B-Instruct` along with the release where several machines are set up.
Keep the venv's `Scripts` on `PATH` while it runs so the nested `uv` call resolves; that `uv`
uses its default cache and interpreter folders, not the ones Block 1 set (not tested under a
packaged agent app). Expected: `overlaid 750/750 Cosmos3 tensors`, 750 tensors in 4 shards,
16.3 GiB (17.6 GB). Disk budget: release 35 GB, Hub cache 24 GB, converted 17.6 GB, plus
35.3 GB per DCP checkpoint and 17.6 GB per export.

## Memory and GPU readings

`nvidia-smi` on WDDM shows the dedicated segment (8 GiB on one 128 GB unit, 32 GiB on another)
and not the CUDA budget; the preflight prints the budget (101.5 to 106.8 GiB on 128 GB units;
about 45 GiB on the 64 GB unit). Under WSL the readings differ again; this skill does not cover
WSL.

## Verify

`scripts/preflight_windows_arm64.py --allocation-gib N` (ARM64, torch, CUDA, a real allocation of
N GiB), the three import lines of Variant 1 Block 2, and a one-step training smoke before any
real run.

The preflight is slow by design: the allocation step writes every page, about 1.5 s per GiB
(111 s for 72 GiB on a tester's unit), and a first run with the torch import and CUDA start-up
can pass five minutes. Give the call a ten-minute timeout. Progress lines go to stderr; the JSON
report goes to stdout and to `--output <file>`. Take the report from the file or from stdout
alone and do not merge stderr into it (`2>&1`). Windows PowerShell 5.1 turns each stderr line
of a native program into an error record: the first progress line then arrives as a
`NativeCommandError` block in front of the JSON, which no longer parses, and in a script that
sets `$ErrorActionPreference = 'Stop'` that first line ends the script with exit code 1 and no
report, although the preflight itself passes. Without the redirection the progress lines go to
the console, stdout parses by itself, and the exit code is the preflight's.
