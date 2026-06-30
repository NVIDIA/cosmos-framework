# LanceDB-powered Cosmos Dataloaders

This directory contains LanceDB-backed implementations of the three main dataloaders used in Cosmos training:
- **Action (DROID/LeRobot)**: `LanceDROIDComposedDataset`
- **Vision-SFT (Local clips)**: `LanceVisionSFTDataset`
- **VLM (LLaVA-OneVision)**: `LanceVLMDataset`

These loaders are designed for higher throughput, better memory scaling, and native object-store (S3) access while maintaining verified equivalence with the original loaders (exact labels/tokens; video within one offline H.264 re-encode).

## Key Features

- **Higher Throughput**: Up to 3.3x speedup locally and 4.9x on S3 when tuned.
- **Memory Efficiency**: Reduces per-worker memory footprint by up to 3x at scale by eliminating redundant per-frame indices.
- **Native S3 Support**: Uses LanceDB's native object-store integration for parallel, selective reads without FUSE or full downloads.
- **Verified Equivalence**: VLM records byte-identical, vision-SFT token-ids exact, action labels (action/pose/caption) bit-exact with video within H.264 re-encode tolerance (~1.5%).

## Performance Summary

### Combined Throughput (samples/s)
Combined 3-loader throughput, 327 DROID episodes, batch 16:

| Workers (Action/VLM/VSFT) | Base (Local) | Lance (Local) | Base (S3) | Lance (S3) |
| ------------------------- | ------------ | ------------- | --------- | ---------- |
| 4/4/4 (Default)           | 92.6         | 254.8 (2.7x)  | 72.6      | 265.2 (3.6x)|
| 18/4/18 (Tuned)           | 280.1        | 931.0 (3.3x)  | 251.7     | 1240.7 (4.9x)|

### Memory Scaling (Action Loader)
Per-worker PSS memory at scale:

| Dataset Size | Base | Lance |
| ------------ | ---- | ----- |
| 96k frames   | 651 MB | 737 MB |
| 1.54M frames | 2612 MB| 863 MB |

## Mechanisms

1. **Pre-composed Clips**: For Action and Vision-SFT, frames are resized and composed offline once. The loader decodes a single optimized stream instead of multiple full-resolution views.
2. **Columnar Random Access**: Provides O(1) random access and true global shuffle for VLM datasets.
3. **Batched I/O**: `__getitems__` performs batched reads and decodes per file/clip, maximizing I/O efficiency.
4. **Parallel S3 Reads**: Uses plain binary storage to leverage Lance's IO thread pool for concurrent GET requests.

## Usage

### 1. Build Tables
Use the provided tools to convert your datasets to Lance format:
```bash
# Action
python tools/lance_datagen/build_composed_droid.py --root <droid_root> --uri <lance_uri> --gop 1 --storage plain

# Vision-SFT
python tools/lance_datagen/build_vision_sft.py --jsonl <metadata.jsonl> --uri <lance_uri> --storage plain
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
