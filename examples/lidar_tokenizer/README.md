# LiDAR tokenizer V1.2 continuation

Follow the [post-training skill](../../.agents/skills/cosmos3-nano-lidar-tokenizer/post-training/SKILL.md)
for asset selection, local tar manifests, bounded training/resume, evaluation and
publication. The native recipe is
[`lidar_v1p2.py`](../../cosmos_framework/configs/tokenizer/lidar_v1p2.py).

This tokenizer objective has its own entry point:

```bash
set -euo pipefail
cd "${FRAMEWORK_DIR:?set this Framework checkout root}"
COSMOS_INTERNAL=0 COSMOS_TRAINING=1 \
  "${TRAIN_PYTHON:?set the configured training environment Python}" \
  -m cosmos_framework.scripts.train_lidar_tokenizer --help
```

Use an existing training environment or follow the [setup guide](../../docs/setup.md).
The skill provides manifest preflight, a 20-step train/save check, continuation
to step 40, and live/EMA export and evaluation. It requires a caller-supplied,
compatible V1.2 initializer and measured recordings; it does not download training
data or select a checkpoint on the caller's behalf. The
[validation record](VALIDATION.md) distinguishes CPU checks from unverified GPU
behavior.

After exporting and evaluating both weight variants, set `LIDAR_SELECTED_EXPORT`
to the measured winner's export directory. Load its explicit metadata with the
configured Python environment:

```python
import json
import os
from pathlib import Path

import torch

from cosmos_framework.model.generator.tokenizers.lidar.lidar_tokenizer_v1 import LidarTokenizerV1Interface

artifact = Path(os.environ["LIDAR_SELECTED_EXPORT"])
config = json.loads((artifact / "config.json").read_text())
tokenizer = LidarTokenizerV1Interface(
    vae_path=str(artifact / "tokenizer.pt"),
    device="cuda",
    dtype=torch.float32,
    sample_posterior=False,
    network_config=config["network_config"],
    range_projection=config["range_projection"],
    spatial_compression=config["spatial_compression"],
    temporal_compression_factor=config["temporal_compression"],
)
# metric_video: [B,3,T,128,1808], range metres / unit intensity / validity.
# latent = tokenizer.encode(metric_video)  # [B,128,T,8,113]
# reconstruction = tokenizer.decode(latent)  # [B,3,T,128,1808]
```

The serialized checkpoint includes its learned latent moments and metric range
bounds. Keep the accompanying configuration and statistics together. The
ordinary V1 default network configuration is not the V1.2 architecture.
