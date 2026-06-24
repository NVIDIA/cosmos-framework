# Reproducing the LanceDB-vs-base dataloader benchmarks

Everything an independent user/agent needs to recreate these numbers from scratch on their
own machine. Three regimes: **LOCAL** (apples-to-apples, cosmos's documented workflow), **S3**
(Lance object-store-native vs the base's stock S3 access), and **DEFAULT-MIXED** (each loader on
its real default storage). All comparisons are **CPU-decode on both sides** (the base can only
decode on CPU — never compare CPU-vs-GPU).

## 0. Hardware / OS
- Linux, x86-64. A CUDA GPU is **not** required for the dataloader benchmarks (decode is CPU);
  it is only needed for the training-equivalence scripts (`train_equiv_real.py`).
- System `ffmpeg` (the loaders decode via torchcodec/ffmpeg). FFmpeg 7 or 8 both work.
- ~5 GB disk for the subsets + Lance tables. For the S3 regime, an AWS account + bucket.

## 1. Python environment (exact — this is the fiddly part)
Python 3.12 venv. **torchcodec must match torch exactly**, and its `.so` needs the CUDA + NPP +
ffmpeg libs on `LD_LIBRARY_PATH` — even for CPU decode (the wheel links them). Pin torch with a
constraints file so installing the data deps can't silently downgrade it to a CPU build.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip

# (a) the CUDA torch stack — torchcodec 0.10 pairs with torch 2.10 (cu128)
pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchcodec==0.10.0+cu128
pip install nvidia-npp-cu12==12.3.3.100      # torchcodec_core*.so needs libnppicc

# (b) pin torch so the next installs can't clobber it
printf 'torch==2.10.0+cu128\ntorchvision==0.25.0+cu128\ntorchcodec==0.10.0+cu128\n' > /tmp/cons.txt

# (c) data + framework deps (under the constraint)
pip install -c /tmp/cons.txt --extra-index-url https://download.pytorch.org/whl/cu128 \
    lerobot webdataset transformers peft einops datasets \
    scipy opencv-contrib-python imageio imageio-ffmpeg mediapy \
    loguru cattrs hydra-core omegaconf termcolor tyro msgpack nvidia-ml-py av obstore \
    boto3==1.40.0 botocore s3fs iopath \
    pytest pytest-xdist pytest-custom_exit_code
```

**Always `source benchmarks/lance/_env.sh` before running** — it puts the NPP/CUDA/ffmpeg lib
dirs on `LD_LIBRARY_PATH` and the repo on `PYTHONPATH`. Verify:
```bash
source benchmarks/lance/_env.sh
python -c "import torch,torchcodec,lerobot,lance; from torchcodec.decoders import VideoDecoder; \
  print('ok', torch.__version__, torch.cuda.is_available())"
```

## 2. Datasets (public on HF)
```bash
export HF_TOKEN=...   # needed for LLaVA-OneVision streaming/download
# action: DROID
hf download lerobot/droid_1.0.1 --repo-type dataset --local-dir <droid_raw>
# vision-SFT: BridgeData2 synthetic captions  (has train/video_dataset_file.jsonl + videos/)
hf download nvidia/BridgeData2-Subset-Synthetic-Captions --repo-type dataset --local-dir <bridge>
# VLM: LLaVA-OneVision-Data — the figureqa subset (streamed at run time for the base; converted for Lance)
```

## 3. Build the Lance tables + Cosmos-format subset (offline, one-time)
```bash
source benchmarks/lance/_env.sh
# action: rename DROID -> Cosmos schema, then pre-compose 3 views -> 1 all-intra clip/episode
python tools/lance_datagen/prepare_droid_subset.py --src <droid_raw> --out <droid_out> --num-episodes 327
python tools/lance_datagen/build_composed_droid.py --root <droid_out>/success --uri <droid_lance_dir> --gop 1
# vision-SFT: re-encode each clip pre-resized + all-intra into a blob-v2 table
python tools/lance_datagen/build_vision_sft.py --jsonl <bridge>/sft_dataset_bridge/train/video_dataset_file.jsonl \
    --uri <vsft_lance_dir> --resolution 256 --gop 1
# VLM: convert the figureqa subset to a Lance table (stores original PNG bytes inline, no re-encode)
python -c "from datasets import load_dataset; from cosmos_framework.data.lance.vlm_dataset import convert_llava_to_lance; \
  convert_llava_to_lance(load_dataset('lmms-lab/LLaVA-OneVision-Data', name='figureqa(cauldron,llava_format)', split='train'), '<llava_lance_dir>')"
# (optional, for the webdataset-tar VLM base variant) python tools/lance_datagen/build_wds_shards.py --out <wds_dir>
```

## 4. Equivalence (prove identical output before trusting throughput)
```bash
DROID_COSMOS_ROOT=<droid_out>/success DROID_LANCE_URI=<droid_videoblob_lance> \
BRIDGE_JSONL=<bridge>/sft_dataset_bridge/train/video_dataset_file.jsonl VISION_SFT_LANCE_URI=<vsft_lance_dir> \
  python -m pytest tests/data/lance/test_action_equivalence.py tests/data/lance/test_vision_sft_equivalence.py
# expect 15 passed (action video/labels bit-exact; vision-SFT token ids exact)
```

## 5. Benchmarks
Run `--trios base` and `--trios lance` in **separate processes** (a single process hits a benign
torchcodec/lance SIGABRT at teardown between trios). Numbers below were measured on a 48-CPU + L40S
node, 327 DROID episodes, 1:1:1 mixer, 6 workers/loader, batch 16.

### 5a. LOCAL (apples-to-apples — cosmos's documented download-to-local workflow)
```bash
for t in base lance; do
  python benchmarks/lance/bench_combined_faithful.py \
    --action-root <droid_out>/success --action-uri <droid_lance_dir> \
    --vlm-wds "<wds_dir>/shard-{00000..00019}.tar" --vlm-uri <llava_lance_dir> \
    --vsft-jsonl <bridge>/.../video_dataset_file.jsonl --vsft-uri <vsft_lance_dir> \
    --batch-size 16 --num-workers 6 --rounds 30 --warmup 10 --trios $t
done
```
Expected: action **1.93×**, VLM raw 1.63×, vision-SFT **7.57×**, **combined 3.11×** (122→380 samples/s).

### 5b. S3 (Lance native `s3://` vs the base's stock S3 access)
Upload the Lance tables + the vision-SFT base videos to a bucket; set AWS creds (`AWS_PROFILE`) and
`LANCE_IO_THREADS=256`. The base reads each dataset the way its stock loader does — action/VLM via an
s3fs FUSE mount (no native reader), vision-SFT via boto3 download-per-sample (`--vsft-s3-bucket/prefix`).
```bash
export AWS_PROFILE=<profile> LANCE_IO_THREADS=256
for t in base lance; do
  python benchmarks/lance/bench_combined_faithful.py \
    --action-root <s3fs_mount>/.../success --action-uri s3://<bucket>/.../droid_composed \
    --vlm-wds "<s3fs_mount>/.../shard-{00000..00019}.tar" --vlm-uri s3://<bucket>/.../llava \
    --vsft-jsonl <local>/video_dataset_file.jsonl --vsft-uri s3://<bucket>/.../vision_sft \
    --vsft-s3-bucket <bucket> --vsft-s3-prefix <prefix>/sft_dataset_bridge/train \
    --region <region> --batch-size 16 --num-workers 6 --rounds 30 --warmup 10 --trios $t
done
```
Expected: action 1.71×, VLM raw 1.70×, vision-SFT 2.66×, **combined 2.64×** (95→252 samples/s).

### 5c. DEFAULT-MIXED (each loader on its real default storage)
base: action=LOCAL, vision-SFT=S3(boto3), VLM=HF-Hub streaming · lance: action=LOCAL, vision-SFT=S3, VLM=S3.
Same command as 5b but `--action-root`/`--action-uri` are **local**, and add
`--vlm-hf-subset "figureqa(cauldron,llava_format)"` (streams the base VLM from HF — needs `HF_TOKEN`).
`storage_options` auto-applies only to `s3://` uris, so local action + S3 vsft/VLM coexist in one run.
Expected: action 1.70×, vision-SFT 2.51×, **combined 2.66×** (95→254 samples/s). (VLM shows a huge raw
ratio — base HF-stream 901 vs Lance S3-scan 39,428 — but it's never the mixer bottleneck.)

**All three regimes agree: combined ≈ 2.6–3.1×**, gated by the slowest (video) loader.

## 6. Single-loader / diagnostic scripts
- `bench_action_faithful.py --modes base-random base-episode lance-random lance-episode` — the action
  2×2 (shows the speedup is worker-count-dependent, shuffle-mode-neutral locally).
- `bench_vlm.py`, `bench_vision_sft.py`, `bench_decode.py` — per-loader / decode microbenchmarks.
- `bench_filtered.py` — predicate-pushdown (curriculum/quality filtering) capability demo.
- `train_equiv_real.py`, `train_databound_demo.py`, `train_multigpu_time.py` — training-time / equivalence (need a GPU).

## 7. Gotchas (learned the hard way)
- **Same decode device both sides** — always CPU. The base can't use GPU; cu128 torchcodec ≠ GPU decode.
- **Separate process per trio** (`--trios base` then `--trios lance`) to dodge the teardown SIGABRT.
- **The combined number is bottleneck-gated** (aggregate ≈ 3×slowest loader); report the per-loader
  breakdown alongside it, never a bare combined multiple.
- **S3 base access matters**: ffmpeg-through-FUSE is much slower than boto3 download-per-sample — use
  each base loader's *actual* stock S3 path, or you'll inflate the win (see BENCHMARKS.md).
- We did **not** modify any stock base loader; S3 reading is either FUSE (no code change) or the base's
  own already-shipped boto3 reader.
