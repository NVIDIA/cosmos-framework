# Optimization audit: borrow from base loaders + optimal lance usage

Audit of (A) optimizations in the base cosmos loaders worth borrowing into the Lance
loaders, and (B) whether the Lance loaders use lance/lancedb optimally. Validated items are
implemented on `lancedb-dataloader-experiments`; the rest are concrete recommendations.

## A. Borrowed from the base loaders
1. **Episode-shuffle stream** — *implemented + validated*. Base `ActionIterableShuffleDataset`
   shuffles per-episode block ORDER and streams windows WITHIN an episode sequentially. Ported
   as `LanceDROIDComposedIterable`: consecutive windows share an episode, so the per-episode
   clip decoder is built ONCE and reused, instead of `RandomSampler` rebuilding it (a fresh
   `take_blobs` + `VideoDecoder`) on cache misses.
   - Measured (composed loader, 4 workers): **S3 2.55×** (35.3 → 90.0 samples/s, cache=4 =
     many-episode regime); **local: neutral/-** (≈0.83–0.96×) because local clips are tiny/hot
     so re-reads are cheap and `RandomSampler`+LRU already reuse. Net: a real win in the
     realistic S3/scale regime, no benefit locally. `bench_episode_shuffle.py`.
2. **`COSMOS_DL_FILE_SYSTEM_SHARING`** — *recommend/honor*. Base flips torch DataLoader IPC to
   `file_system` so large video batches don't overflow `/dev/shm`. Our video loaders emit the
   same large tensors; set `COSMOS_DL_FILE_SYSTEM_SHARING=1` (already wired in `sitecustomize.py`)
   for many-worker video runs.
3. **uint8, skip the float round-trip** — *minor*. The composed loader decodes uint8 →`/255`→
   `_build_result`→`*255`→uint8. When augmentation is off it could return uint8 directly (halves
   transient memory + IPC). Left as-is for exact parity with the base `_build_result`.

## B. Lance-side — was our usage optimal?
4. **Scanner readahead** — *implemented*. `LanceVLMShuffleScan` now passes `batch_readahead=8`
   to `to_batches` (prefetches the next batches' IO; matters on S3). Falls back if unsupported.
5. **`optimize.compact_files()` after conversion** — *recommend*. Streaming `create_table`
   writes one fragment stream; compacting improves random-read layout at scale. Our tables are
   currently a single fragment (no-op here), but at production scale run
   `lance.dataset(uri).optimize.compact_files()` after conversion.
6. **`create_scalar_index` for filtered reads** — *recommend*. The filtered-sampling demo
   (`bench_filtered.py`) scans the predicate column. For real curriculum/quality filtering add a
   BTREE scalar index on the filter column (`ds.create_scalar_index("bucket", "BTREE")`) so the
   predicate is an index lookup, not a column scan — compounds the 1/selectivity win.
7. **`take_blobs` streaming vs `readall()`** — *minor*. We `readall()` the per-episode clip
   blob (small, ~1.6 MB — fine). The bit-exact `LanceDROIDDataset` reads a large concatenated
   blob with `readall()`; there, passing the `BlobFile` (range-read file-like) to the decoder
   would avoid the full download. Low priority (the composed/throughput path is the one used).

## What we were already doing right
Permutation API with `select_columns` + `with_format("arrow")`; batched `__getitems__`
(dedup + single fetch); worker-safe lazy handles (`__getstate__` nulls, `_ensure_open`);
`seek_mode="approximate"`; per-worker decoder LRU cache; blob-v2 byte-range reads via
`take_blobs`; columnar/selective reads.

## Re-measure: did the optimizations move the S3 training-throughput demo?
`train_databound_demo.py` now supports `--loader lance-episode`. 4× L40S, S3, data-bound:

| loader | s/epoch | samples/s | vs base |
| ------ | ------- | --------- | ------- |
| base | 5.3 | 151 | 1.0× |
| lance (random) | 3.0 | ~260 | 1.74× |
| lance-episode | 3.4–3.8 | ~220 | ~1.5× |

The demo subset is the first ~800 flat indices ≈ **3 episodes**, which fit entirely in the
decoder LRU (32), so `RandomSampler` never misses and episode-shuffle has nothing to recover
(its iterable overhead even makes it marginally slower). Episode-shuffle's win requires
episodes-in-flight > cache (the real many-episode regime), where `bench_episode_shuffle`
(cache=4, S3) measured **2.55×** (35 → 90 samples/s). Lesson: episode-shuffle is a
large-dataset/object-store optimization, not a small-subset one.
