# LanceDB-powered Cosmos Dataloaders

This directory contains LanceDB-backed implementations of the three main dataloaders used in Cosmos training:
- **Action (DROID/LeRobot)**: `LanceDROIDComposedDataset`
- **Vision-SFT (Local clips)**: `LanceVisionSFTDataset`
- **VLM (LLaVA-OneVision)**: `LanceVLMDataset`

These loaders are designed for higher throughput and native object-store (S3) access while maintaining verified equivalence with the original loaders (exact labels/tokens; video within one offline H.264 re-encode).

## Key Features

- **Higher Throughput**: Up to 3.8x speedup locally and 4.4x on S3 when tuned.
- **Native S3 Support**: Uses LanceDB's native object-store integration for parallel, selective reads without FUSE or full downloads.
- **Verified Equivalence**: VLM records byte-identical, vision-SFT token-ids exact, action labels (action/pose/caption) bit-exact with video within H.264 re-encode tolerance (~1.5%).

## Performance Summary

### Combined Throughput (samples/s)
Combined 3-loader throughput, 327 DROID episodes, batch 16:

| Workers (Action/VLM/VSFT) | Base (Local) | Lance (Local) | Base (S3) | Lance (S3) |
| ------------------------- | ------------ | ------------- | --------- | ---------- |
| 4/4/4 (Default)           | 86.5         | 249.4 (2.9x)  | 67.7      | 246.9 (3.6x)|
| 18/4/18 (Tuned)           | 253.7        | 961.9 (3.8x)  | 232.0     | 1016.9 (4.4x)|

## Memory: a note on the per-frame index (not a Lance advantage)

`ActionBaseDataset.__init__` builds a per-frame index (`self._rows`, a list of row dicts) and
ships it to every DataLoader worker. `DROIDLeRobotDataset` **never reads it** — it indexes via
compact column arrays and reconstructs window rows on demand — so the Lance loader drops it
(`self._rows = None`), which we verified is **output-neutral (bit-identical batches)**.

That accounts for most of the per-worker memory gap vs the *shipped* base (~2.7× at 1.5M
frames). **It is not a fundamental Lance advantage, though** — `_rows` is a freeable redundancy
the base could drop too. Once it does, memory is at parity (a `_rows`-freed base is ~0.70 GB vs
Lance ~0.98 GB per worker at 16×; Lance carries the torchcodec decoder cache). The genuine Lance
wins are **throughput and S3**, not memory. _(We've raised this upstream to confirm `_rows` is
safe to drop for `DROIDLeRobotDataset`.)_

What `_rows` costs — per-worker spawn payload (327-episode DROID subset replicated N×):

| Dataset Size       | base keeps `_rows` | base drops `_rows` |
| ------------------ | ------------------ | ------------------ |
| 96k frames (1×)    | 37 MB              | 11 MB              |
| 1.54M frames (16×) | 552 MB             | 133 MB             |
| 3.08M frames (32×) | 1.1 GB             | 263 MB             |
| 6.16M frames (64×) | 2.2 GB             | 524 MB             |

`_rows` ≈ 270 B/frame; the remaining ~85 B/frame is the compact arrays both keep. A base that
keeps `_rows` reaches a ~12 GB resident index at 64× (OOM territory at full-DROID scale).

## Mechanisms

1. **Pre-composed Clips**: For Action and Vision-SFT, frames are resized and composed offline once. The loader decodes a single optimized stream instead of multiple full-resolution views.
2. **Columnar Random Access**: Provides O(1) random access and true global shuffle via the LanceDB **Permutation API**.
3. **Batched I/O**: `__getitems__` performs batched reads and decodes per file/clip, maximizing I/O efficiency.
4. **S3 Reads**: Media is stored as plain `large_binary` and read via the Permutation API (columnar take across Lance's IO thread pool). _TODO: move to blob-v2 after optimizations — it's faster for larger per-row payloads when read in parallel._

## Usage

### 1. Build Tables
Use the provided tools to convert your datasets to Lance format:
```bash
# Action
python tools/lance_datagen/build_composed_droid.py --root <droid_root> --uri <lance_uri> --gop 1

# Vision-SFT
python tools/lance_datagen/build_vision_sft.py --jsonl <metadata.jsonl> --uri <lance_uri>
```

### 2. Integration
Replace the standard datasets with their Lance counterparts in your configuration.
```python
from cosmos_framework.data.lance import LanceDROIDComposedDataset, LanceVisionSFTDataset, LanceVLMDataset
```

## Testing
Run equivalence tests to verify parity with base loaders:
```bash
pytest tests/data/lance/
```
