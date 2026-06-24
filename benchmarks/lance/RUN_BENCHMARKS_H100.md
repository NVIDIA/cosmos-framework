# Benchmark runbook — LanceDB vs base Cosmos dataloaders on 8× H100 / H200 / B200

**Audience:** a coding agent on a fresh multi-GPU node. Execute top-to-bottom. The goal is to
reproduce, on faster GPUs, the dataloader-throughput and **end-to-end training** comparison between
the stock Cosmos dataloaders and the LanceDB ports — for a **tiny custom model** (data-bound regime)
and the **real 8B path** (Qwen3-VL-8B / Cosmos3-Nano), in **both LOCAL and S3** storage. The
hypothesis being tested: on slow GPUs training is compute-bound and the dataloader is hidden; faster
GPUs (and 8-way data parallelism) push training toward **data-bound**, where the Lance loader's
throughput wins translate into faster training. **Your job is to find where that crossover lands on
this hardware and report the numbers.**

Background already established on an L40S node (for context, reproduce/verify these trends):
- Dataloader throughput (combined 3-loader mixer): Lance 2.85–6.48× over base depending on regime +
  worker allocation; biggest win is full-S3.
- E2E training, single L40S: at ≥2 transformer layers the step is **compute-bound** → base == lance
  wall-clock (GPU data-wait <8%); at tiny compute it's **data-bound** → lance ~2× (614 vs 305 samp/s).
- The data-bound threshold on L40S was ~305 samp/s (base MIXED ceiling); faster GPUs cross it sooner.

---

## 0. Hardware-specific environment

```bash
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv   # record GPU model, count, sm
nproc                                                               # record CPU core count (drives worker tuning)
```

**CUDA/torch pins by GPU arch** (torchcodec must match torch exactly, and its `.so` needs CUDA+NPP+ffmpeg on `LD_LIBRARY_PATH`):
- **H100 / H200 (sm_90):** the L40S pins work — `torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchcodec==0.10.0+cu128` + `nvidia-npp-cu12`.
- **B200 / GB200 (sm_100, Blackwell):** needs CUDA 12.8+ **and** a torch build with sm_100 kernels. Use the newest stable `cu128` (or `cu129`) wheels; if `torch.cuda.is_available()` works but matmuls error with "no kernel image", upgrade to a torch nightly that lists `sm_100`. Verify with `python -c "import torch;print(torch.cuda.get_device_capability())"` → expect `(10,0)`.

```bash
cd <repo>                       # the cosmos-framework fork, branch: lancedb-dataloader-experiments
python3.12 -m venv .venv-gpu && source .venv-gpu/bin/activate
pip install -U pip
pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchcodec==0.10.0+cu128   # adjust per arch above
pip install nvidia-npp-cu12==12.3.3.100
printf 'torch==2.10.0+cu128\ntorchvision==0.25.0+cu128\ntorchcodec==0.10.0+cu128\n' > /tmp/cons.txt
pip install -c /tmp/cons.txt --extra-index-url https://download.pytorch.org/whl/cu128 \
    lerobot webdataset transformers peft einops datasets scipy opencv-contrib-python imageio \
    imageio-ffmpeg mediapy loguru cattrs hydra-core omegaconf termcolor tyro msgpack nvidia-ml-py \
    av obstore boto3 botocore s3fs iopath pytest lancedb pylance
pip install -e . --no-deps           # cosmos-framework editable
# torchcodec LD_LIBRARY_PATH (append to the venv activate so it always applies):
echo 'export LD_LIBRARY_PATH="'$PWD'/.venv-gpu/lib/python3.12/site-packages/nvidia/npp/lib:$LD_LIBRARY_PATH"' >> .venv-gpu/bin/activate
```
> **Do NOT use `benchmarks/lance/_env.sh`** — it points at a stale venv. Always `source .venv-gpu/bin/activate`.
> Verify: `python -c "import torch,torchcodec,lance,lerobot;from torchcodec.decoders import VideoDecoder;print('ok',torch.cuda.is_available())"`

Credentials (for S3 + HF). Write to a **gitignored** file and an AWS profile named `cosmosbench`:
```bash
cat > benchmarks/lance/.creds.env <<EOF
export AWS_ACCESS_KEY_ID=...    AWS_SECRET_ACCESS_KEY=...   AWS_DEFAULT_REGION=us-east-2
export HF_TOKEN=...             HUGGING_FACE_HUB_TOKEN=...
EOF
mkdir -p ~/.aws && printf '[cosmosbench]\naws_access_key_id=%s\naws_secret_access_key=%s\n' "$AWS_ACCESS_KEY_ID" "$AWS_SECRET_ACCESS_KEY" > ~/.aws/credentials
```

## 1. Data (LOCAL tables + S3 + s3fs mount for the base's S3 access)

The S3 bucket already holds prebuilt tables: `s3://lancedb-datasets-dev-us-east-2-devrel/cosmos/{droid327,llava,vision_sft}/{base,lance,wds}`.
Pull the LOCAL copies (or rebuild — see `REPRODUCE.md`). Required local layout under `$DATA=/home/ubuntu/work/data` (or your path; edit the constants at the top of the scripts):
- `droid327/success` (Cosmos-schema DROID, 327 eps) + `lance/droid_composed327_plain`
- `bridge_src/sft_dataset_bridge/train/video_dataset_file.jsonl` + `lance/vision_sft_plain`
- `wds/llava_figureqa/shard-{00000..00019}.tar` + `lance/llava_figureqa`

Build the **plain-binary** lance tables (faster on S3 than blob-v2; loaders auto-detect):
```bash
python tools/lance_datagen/build_composed_droid.py --root $DATA/droid327/success --uri $DATA/lance/droid_composed327_plain --gop 1 --storage plain
python tools/lance_datagen/build_vision_sft.py --jsonl $DATA/bridge_src/.../video_dataset_file.jsonl --uri $DATA/lance/vision_sft_plain --resolution 256 --gop 1 --storage plain
python -c "from datasets import load_dataset;from cosmos_framework.data.lance.vlm_dataset import convert_llava_to_lance;convert_llava_to_lance(load_dataset('lmms-lab/LLaVA-OneVision-Data',name='figureqa(cauldron,llava_format)',split='train'),'$DATA/lance/llava_figureqa')"
```
**For the base's S3 access** (stock action/VLM have no native S3 reader → s3fs FUSE; vsft uses boto3):
```bash
mkdir -p /home/ubuntu/s3mnt
s3fs lancedb-datasets-dev-us-east-2-devrel /home/ubuntu/s3mnt -o profile=cosmosbench -o endpoint=us-east-2 -o url=https://s3.us-east-2.amazonaws.com
ls /home/ubuntu/s3mnt/cosmos/droid327/base/success   # sanity
```
If you rebuilt tables locally, also upload the plain ones to S3 (boto3 `upload_file` over the `.lance` dir).

## 2. Sanity: correctness + GPU + re-tune worker allocation

```bash
# equivalence (must pass before trusting throughput)
DROID_COSMOS_ROOT=$DATA/droid327/success DROID_LANCE_URI=$DATA/lance/droid_video \
BRIDGE_JSONL=$DATA/bridge_src/.../video_dataset_file.jsonl VISION_SFT_LANCE_URI=$DATA/lance/vision_sft_plain \
HF_TOKEN=$HF_TOKEN pytest tests/data/lance/test_action_equivalence.py tests/data/lance/test_vision_sft_equivalence.py tests/data/lance/test_vlm_equivalence.py -q
```
**Re-tune workers for THIS core count.** The L40S optimum was 18/4/18 on 48 cores; the knee is ~3× the
action loader's per-loader peak, and oversubscribing cores *degrades* it. Sweep on the new box:
```bash
for a in 8 16 24 32; do
  python benchmarks/lance/bench_combined_faithful.py --action-root $DATA/droid327/success --action-uri $DATA/lance/droid_composed327_plain \
    --vlm-wds "$DATA/wds/llava_figureqa/shard-{00000..00019}.tar" --vlm-uri $DATA/lance/llava_figureqa \
    --vsft-jsonl $DATA/bridge_src/.../video_dataset_file.jsonl --vsft-uri $DATA/lance/vision_sft_plain \
    --action-workers $a --vlm-workers 4 --vsft-workers $a --rounds 22 --warmup 8 --trios lance
done
```
Record the allocation that maximizes `combined mixer`. Call it **$OPT** (e.g. `--action-workers 32 --vlm-workers 4 --vsft-workers 32` on a 128-core box). Use $OPT and `4/4/4` (cosmos default) below.

## 3. Phase 1 — dataloader throughput matrix (3 regimes × 2 allocations)

Use `benchmarks/lance/run_matrix.sh` (edit the path constants + the two allocations: `4 4 4` and your $OPT). It runs LOCAL / full-S3 / MIXED × {base,lance}, each trio isolated. Set `LANCE_IO_THREADS=256`.
```bash
bash benchmarks/lance/run_matrix.sh   # writes matrix_results.txt
```
**Report Table A:** for each (regime ∈ {LOCAL, S3, MIXED}) × (alloc ∈ {4/4/4, OPT}): base / lance combined samples/s + speedup. Expected shape: Lance wins all; full-S3 the biggest; OPT ≈ 4× the 4/4/4 row.

## 4. Phase 2 — e2e training, TINY custom model (finds the data-bound crossover)

`benchmarks/lance/train_combined_e2e.py` drives a real GPU train step (transformer fwd+bwd) from the
real combined mixer. Sweep `--layers` (compute per step). On fast GPUs the crossover shifts — find it.
```bash
for regime in local s3 mixed; do
 for L in 1 2 4 8 16 32; do
  for trio in base lance; do
   python benchmarks/lance/train_combined_e2e.py --trio $trio --regime $regime --layers $L \
     --dim 2048 --heads 16 --seq 2048 $OPT --batch-size 16 --steps 60 --warmup 18
  done
 done
done
```
**Report Table B** (per regime): for each `--layers`, base vs lance `steps/s`, `samples/s`, `data-wait%`.
Identify the **crossover layer count** — the largest model size at which lance still beats base (data-bound),
and the size at which they converge (compute-bound). Compare crossovers LOCAL vs S3 (S3 base is slower →
stays data-bound to larger models). Note: on H100/B200 the GPU is faster, so the crossover should sit at a
**larger** layer count than the L40S (which converged by 2 layers).

Optional — **simulate 8-way data-parallel data demand** without 8 model replicas: add a flag (or run 8
`train_combined_e2e.py` processes pinned to the 8 GPUs sharing nothing) so each rank pulls its own batches;
the aggregate read pressure on the dataset is what an 8-GPU job imposes. Report whether base saturates.

## 5. Phase 3 — e2e training, the REAL 8B path (Qwen3-VL-8B / Cosmos3-Nano)

This is the shipped single-modality vision SFT (`vision_sft_nano`, 8-GPU FSDP) driven by
`cosmos_framework.scripts.train`. Get the checkpoints first:
- `examples/checkpoints/Cosmos3-Nano` (BASE_CHECKPOINT_PATH), `examples/checkpoints/wan22_vae/Wan2.2_VAE.pth` (WAN_VAE_PATH), Qwen3-VL-8B tokenizer/weights (HF, may be gated → `HF_TOKEN`).
- Dataset: `examples/data/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge` (or point `DATASET_PATH` at the bridge data you already have).

**5a. BASE run** (stock dataloader):
```bash
DATASET_PATH=$DATA/bridge_src/sft_dataset_bridge bash examples/launch_sft_vision_nano.sh
```
The trainer **logs dataloader + iteration speed natively** — that is your measurement, no instrumentation
needed. Watch the log (`outputs/.../vision_sft_nano_sft.log`) for:
- `iter_speed` (steps/s or s/iter) and `dataloader_speed` (the metric wired at
  `configs/base/experiment/sft/vision_sft_nano.py` ~line 145/156). Record steady-state values (skip warmup).
- GPU utilization (`nvidia-smi dmon`) — low/spiky util ⇒ data-bound; pinned 100% ⇒ compute-bound.

**5b. LANCE run** (swap the dataset, keep everything else). Edit `configs/base/experiment/sft/vision_sft_nano.py`:
the dataset is built at ~line 242 as `dataset=L(get_sft_dataset)(... jsonl_paths=[...] ...)` inside
`PackingDataLoader`. Replace that inner `dataset=L(get_sft_dataset)(...)` with the Lance loader:
```python
from cosmos_framework.data.lance import LanceVisionSFTDataset
...
dataset=L(LanceVisionSFTDataset)(
    lance_uri="${oc.env:VSFT_LANCE_URI}",   # local dir OR s3://.../vision_sft/lance/vision_sft_plain
    table="vision_sft", decode_device="cpu",
    storage_options={"region": "us-east-2"},  # only for s3:// uris; omit/None for LOCAL
    num_video_frames=..., temporal_interval_mode=..., frame_selection_mode=...,  # mirror the base kwargs
),
```
`LanceVisionSFTDataset` is output-equivalent to `SFTDataset` (token-ids exact, video within H.264
tolerance — see `tests/data/lance/test_vision_sft_equivalence.py`), so `PackingDataLoader` and the model
are unchanged. **Verify the produced sample dict keys match** what `PackingDataLoader` expects (it does on
the bench harness; confirm under the real packer and adjust kwargs if a field is missing). Then:
```bash
VSFT_LANCE_URI=$DATA/lance/vision_sft_plain DATASET_PATH=$DATA/bridge_src/sft_dataset_bridge bash examples/launch_sft_vision_nano.sh   # LOCAL
VSFT_LANCE_URI=s3://lancedb-datasets-dev-us-east-2-devrel/cosmos/vision_sft/lance/vision_sft_plain bash examples/launch_sft_vision_nano.sh   # S3
```
Run base and lance for the same fixed #iterations; compare steady-state `iter_speed` + `dataloader_speed` + GPU util.

**5c. 8B-scale COMBINED proxy (optional, if the omni joint loader isn't wired):** run
`train_combined_e2e.py` with an 8B-sized transformer under FSDP so it exercises the **combined** mixer at
real-model compute. Wrap `PackedTransformer` in `torch.distributed.fsdp.FullyShardedDataParallel`, launch
with `torchrun --nproc_per_node=8`, and size to ~8B (`--dim 4096 --layers 32 --heads 32 --seq 4096`).
Report base vs lance `steps/s` + `data-wait%`, LOCAL and S3. (This keeps the data path real and the combined
mixer real; the model is a sized stand-in for the omni MoT — note that in the report.)

**Report Table C:** real 8B vision SFT — base vs lance: steady `iter_speed`, `dataloader_speed`, GPU-util%,
for LOCAL and S3. Plus the 8B-scale combined proxy if run. The key question: **at 8× H100/B200 FSDP, does
the real 8B step stay compute-bound (base == lance) or does the faster compute + 8-way data demand tip it
data-bound (lance faster)?** Report data-wait% explicitly — that is the verdict.

## 6. What to report (deliverable)

A short markdown with: GPU model/count, core count, chosen $OPT allocation; **Table A** (dataloader matrix),
**Table B** (tiny-model compute sweep + crossover layer per regime), **Table C** (real 8B base-vs-lance +
data-wait). Then a 3-line conclusion answering: (1) where is the data-bound crossover on this hardware vs
the L40S; (2) does the real 8B path become data-bound at 8 GPUs / on S3; (3) the per-regime lance speedup
at the optimal worker allocation. Include the raw logs.

## 7. Gotchas
- **Per-loader workers, not global** — `--action-workers/--vlm-workers/--vsft-workers`; re-tune for this core count (Phase 2). Cosmos default is a flat ~4 (no auto-balance).
- **spawn everywhere** — the combined bench forces `multiprocessing_context="spawn"`; mixing fork+spawn SIGABRTs. If a trio crashes, run `--trios base` and `--trios lance` as separate processes (the bench already `os._exit(0)`s to skip the benign teardown SIGABRT).
- **S3 reads:** `LANCE_IO_THREADS=256`; plain-binary tables read ~6× faster than blob-v2 via columnar `take` (don't switch tables to blob). `data_storage_version` stays **2.1** (2.2 is unstable in Lance 7.0.0).
- **Cold-cache** is not reproducible at these table sizes on a big-RAM box (torch worker RSS crowds out page cache before the 0.5–2 GB dataset does); S3 is the faithful I/O-bound proxy. `bench_cold_cache.py` supports a `systemd-run --scope -p MemoryMax=` cgroup if you must.
- **Rotate** the IAM key + HF token after the run.
```
