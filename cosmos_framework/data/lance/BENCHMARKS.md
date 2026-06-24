# Benchmarks — LanceDB vs base Cosmos dataloaders

All numbers from a single node (48 CPU + NVIDIA L40S), 327 DROID episodes, batch 16, **CPU decode on
both sides** (the base can only decode on CPU), RAW (no model) unless a row says otherwise. Lance tables
use **plain `large_binary`** storage (the loaders auto-detect; see [`HOW_IT_WORKS.md`](HOW_IT_WORKS.md) §4a).
Mechanisms behind every win: [`HOW_IT_WORKS.md`](HOW_IT_WORKS.md). Reproduce: §"Reproduce" below + [`REPRODUCE.md`](REPRODUCE.md).

Three storage regimes:
- **LOCAL** — all three loaders read local disk (the "pre-downloaded everything" workflow).
- **full S3** — everything on S3 (base action/VLM via s3fs FUSE since they have no native S3 reader; base vsft via boto3).
- **MIXED** — each loader on its *real default* storage: action LOCAL, vision-SFT S3, VLM HF-stream (base)/S3 (lance). This is how Cosmos actually reads (see §"Base loader storage").

---

## 1. Headline — combined 3-loader throughput (samples/s)

The combined 1:1:1 mixer is gated by the slowest loader. Read it under **three** framings:

Worker columns are action/vlm/vsft (`num_workers` per sub-loader's DataLoader).

| framing | base workers | lance workers | LOCAL | full S3 | MIXED |
| ------- | ------------ | ------------- | ----- | ------- | ----- |
| **A. same workers, cosmos default** | 4/4/4 | 4/4/4 | **2.85×** | **3.76×** | **3.79×** |
| **B. same workers, tuned** | 18/4/18 | 18/4/18 | **4.61×** | **6.48×** | **5.46×** |
| **C. Lance tuned vs Cosmos as-shipped** | 4/4/4 (flat-4, no auto-balance — what Cosmos ships) | 18/4/18 | **11.7×** | **19.0×** | **16.2×** |

**Framing C is the real out-of-the-box delta**: Cosmos defaults to ~4 workers per loader and does *not*
rebalance toward the bottleneck (its "multiplex" is ratio-based modality mixing, not worker allocation —
see §4). So a user who adopts the Lance loaders *and* tunes workers sees **12–19×**. Framing B isolates the
pure dataloader change (same workers); Framing A is the worst case (both untuned). All three are honest;
quote the one that matches your question.

### Full matrix (absolute samples/s)

| regime | base 4/4/4 | lance 4/4/4 | base 18/4/18 | lance 18/4/18 |
| ------ | ---------- | ----------- | ------------ | ------------- |
| LOCAL  | 88.8 | 252.7 | 224.7 | 1035.6 |
| full S3 | 67.4 | 253.4 | 197.3 | 1278.1 |
| MIXED  | 69.0 | 261.5 | 205.1 | 1120.7 |

Reproduce: `benchmarks/lance/run_matrix.sh` (each cell a separate `bench_combined_faithful.py --trios …`).
Note full-S3 lance (1278) > LOCAL lance (1036) at optimal workers — S3 reads run on the async IO-thread
pool, so they don't steal decode CPU the way local read syscalls + page-cache contention do.

---

## 2. Single-loader (per-modality) throughput

Most shipped recipes are single-modality (`action_policy_droid`, `llava_ov`, `vision_sft_nano`), so the
per-loader numbers matter standalone. base → lance (speedup), same run as the matrix.

**At the optimal allocation (action/vsft 18 workers, VLM 4):**

| loader (recipe) | LOCAL | full S3 |
| --------------- | ----- | ------- |
| action / DROID (`action_policy_droid`) | 162.7 → 295.6 (**1.82×**) | 143.8 → 385.4 (**2.68×**) |
| VLM / LLaVA (`llava_ov`) | 9,925 → 49,034 (**4.94×**) | 13,404 → 50,816 (**3.79×**) |
| vision-SFT / Bridge (`vision_sft_nano`) | 130.2 → 1,071.6 (**8.23×**) | 105.6 → 768.6 (**7.28×**) |

**At cosmos-default 4 workers:**

| loader | LOCAL | full S3 |
| ------ | ----- | ------- |
| action / DROID | 48.2 → 89.3 (1.85×) | 54.3 → 89.5 (1.65×) |
| VLM / LLaVA | 15,292 → 42,316 (2.77×) | 14,829 → 49,715 (3.35×) |
| vision-SFT / Bridge | 31.2 → 229.2 (7.35×) | 22.0 → 209.2 (9.5×) |

(MIXED VLM base = HF-Hub streaming: 724 samples/s vs lance S3-scan 50,355 = ~70× — different work; VLM is
never the mixer bottleneck.) vision-SFT is the biggest per-loader win and it **holds end-to-end** (~6.5×)
because its only non-video work is a cheap tokenize; the VLM raw win is ~1× e2e (image-processor bound).

---

## 3. Worker-allocation sweep (the dominant combined-throughput lever)

LOCAL lance combined samples/s by allocation (action/vlm/vsft):

| a/v/s (total) | combined | note |
| ------------- | -------- | ---- |
| 6/6/6 (18) | 351.8 | original equal-worker baseline |
| 12/2/12 (26) | 606 | |
| 16/2/16 (34) | 1169.6 | |
| 18/2/10 (30) | 875 | vsft starved |
| **18/4/18 (40)** | **1035–1272** | **optimum** (matrix 1036 / isolated run 1272; run-to-run variance) |
| 20/2/20 (42) | — | action collapses (394→231 samp/s) — core oversubscription |
| 28/2/10 (40) | 566 | action over-subscribed |

The ceiling ≈ 3× the action loader's per-loader peak (~394 samp/s at ~18 workers on 48 cores). Past ~18
workers/heavy-loader the 48 cores oversubscribe and throughput *degrades*. Optimal = give each heavy loader
~its peak worker count, minimal workers to VLM, total ≲ cores. **Re-tune for other core counts**
(`--action-workers/--vlm-workers/--vsft-workers`).

---

## 4. Storage format — plain `large_binary` vs blob-v2 (the S3 read win)

Same ~1.7 MB mp4 clips, read from S3:

| access method | clips/s | MB/s |
| ------------- | ------- | ---- |
| blob-v2 `take_blobs` + readall loop (old) | 31 | 55 |
| **plain `large_binary` + columnar `take` (new)** | **197** | **345** (**6.3×**) |

`take_blobs` returns lazy handles read one-at-a-time → serialized GETs (unchanged by `LANCE_IO_THREADS`,
`io_buffer_size`, or sorted indices — the reads are sequential in Python). Columnar `take` parallelizes
across the IO thread pool. Effect on the read-bound loaders, S3 e2e: vision-SFT **178 → 376 (2.1×)**,
action random **110 → 167 (1.5×)**. Loaders auto-detect the encoding; converters default to `--storage
plain`. `data_storage_version` stays at **2.1** (2.2 is unstable in Lance 7.0.0).

---

## 5. End-to-end TRAINING (does the dataloader win make training faster?)

Real GPU train step (transformer fwd+bwd, sized by `--layers` ≈ the omni MoT per-step compute) fed by the
real combined mixer, MIXED regime, 18/4/18 workers, batch 16, **single L40S**. (No turnkey
combined-dataloader training example ships in cosmos/cosmos-framework — the joint loader is wired in
experiment Python for the 8B omni FSDP job — so the data path is 100% real and the model is a sized stand-in.)

| per-step compute | base steps/s (samp/s) | lance steps/s (samp/s) | base data-wait | verdict |
| ---------------- | --------------------- | ---------------------- | -------------- | ------- |
| **tiny** (data-bound; fast-GPU proxy) | 19.1 (305) | **38.4 (614)** | 89.5% | **lance 2.0×** |
| 2-layer transformer | 5.56 (89) | 5.56 (89) | 7.1% | identical |
| 8-layer transformer | 1.44 (23) | 1.47 (23.5) | 1.7% | identical |

**On a single GPU at a realistic model size, training is compute-bound** → the GPU waits <8% on data →
base == lance wall-clock; the loader is hidden behind forward/backward. The Lance win converts to faster
*training* only when **data-bound**: tiny/cheap compute, very fast GPUs (H100/B200), large data-parallel
fan-out, or remote data. Even when hidden, Lance keeps the GPU fed with **far fewer CPU workers** (base
needs 18 to hit 305 samp/s; lance hits 614) — a host-cost/efficiency win + native object-store training.

**Weaker GPU = more compute-bound = hides the loader more.** A faster GPU finishes each step sooner →
demands data faster → tips data-bound → surfaces the win. To find the crossover on H100/H200/B200, run
[`RUN_BENCHMARKS_H100.md`](RUN_BENCHMARKS_H100.md).

---

## 6. Cold cache (is the LOCAL benchmark unfairly warm?)

Action loader, page cache dropped between passes: base **2–3%** / lance **11–19%** cold penalty — tiny,
because at subset scale the bottleneck is CPU decode, not I/O. A genuine larger-than-RAM regime is **not
reproducible** on a 372 GB box (torch worker RSS crowds out the page-cache budget before the 0.5–2 GB
dataset does, even under a `MemoryMax=6G` cgroup). S3 is the faithful I/O-bound proxy. Tool:
`benchmarks/lance/bench_cold_cache.py` (`--drop-caches`, or wrap in `systemd-run --scope -p MemoryMax=`).

---

## 7. Correctness (output-equivalent to the base — prerequisite for any throughput claim)

| loader | test | result |
| ------ | ---- | ------ |
| action / DROID | `tests/data/lance/test_action_equivalence.py` | **8/8 bit-exact** (`video max|Δ|=0`, `action max|Δ|=0`) |
| vision-SFT | `tests/data/lance/test_vision_sft_equivalence.py` | **7/7** — token-ids exact, video within H.264 tolerance |
| VLM | `tests/data/lance/test_vlm_equivalence.py` | **3/3** — records byte-identical vs the HF stream |

Plain-vs-blob storage is byte-identical, so equivalence holds for both encodings.

---

## 8. Base loader storage — local, remote, or combined? → **COMBINED**

Verified in the cosmos source:
- **action / LeRobot** — local filesystem only, `Path(root)` + `pq.read_table` (`data/vfm/action/datasets/base_dataset.py:65-80`).
- **VLM / LLaVA** — HuggingFace Hub streaming, `load_dataset(..., streaming=True)` (`configs/base/vlm/experiment/llava_ov_vlm.py:73-74`).
- **vision-SFT** — S3 via boto3, `download_from_s3(...)` (`data/vfm/local_datasets/sft_dataset.py:97,196,366`), local fallback (`helper.py:37-38`).

So real Cosmos training reads local disk **and** remote object storage at once — the MIXED regime.

---

## 9. Disk footprint (action loader) — the optimized clips are *smaller*

Composed gop=1 (shipped) = **0.35× the original** 3-view footage (fusing 3 views → 1 half-res clip offsets
the all-intra penalty); gop=8 → 0.18×. Per-frame JPEG (rejected) would be 1.8×. Full table in `README.md`.

---

## Reproduce

Env: Python 3.12, `torch==2.10+cu128` / `torchvision` / `torchcodec` matched, `nvidia-npp-cu12` on
`LD_LIBRARY_PATH` — `source benchmarks/lance/.venv-gpu/bin/activate` (NOT `_env.sh`, which is stale).
Datasets public on HF (`lerobot/droid_1.0.1`, `lmms-lab/LLaVA-OneVision-Data`,
`nvidia/BridgeData2-Subset-Synthetic-Captions`). Build the plain tables with the `tools/lance_datagen/*`
converters (`--storage plain`). Then:

```bash
# full combined matrix (LOCAL/S3/MIXED × 4-4-4 / 18-4-18)
bash benchmarks/lance/run_matrix.sh
# single-loader / worker sweep
python benchmarks/lance/bench_combined_faithful.py … --action-workers A --vlm-workers V --vsft-workers S --trios lance
# storage-format read win
python benchmarks/lance/bench_take_vs_blobs.py --uri s3://…/droid_composed327_plain/… --region us-east-2
# e2e training compute sweep
python benchmarks/lance/train_combined_e2e.py --trio {base,lance} --regime {local,s3,mixed} --layers L …
# multi-GPU H100/H200/B200: see RUN_BENCHMARKS_H100.md
```

Step-by-step (env, downloads, conversions, S3 setup, expected numbers): [`REPRODUCE.md`](REPRODUCE.md).
