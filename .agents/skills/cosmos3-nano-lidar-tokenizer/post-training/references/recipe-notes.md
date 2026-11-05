# Recipe notes and validation limits

The maintained implementation is contained in this Framework checkout. Paths
below are relative to the repository root.

| Component                                  | Code                                                           |
| ------------------------------------------ | -------------------------------------------------------------- |
| V1.2 architecture, objective and schedule  | `cosmos_framework/configs/tokenizer/lidar_v1p2.py`             |
| Train, export and evaluate CLI             | `cosmos_framework/scripts/train_lidar_tokenizer.py`            |
| Training wrapper and latent statistics     | `cosmos_framework/model/tokenizer/lidar/model.py`              |
| Reconstruction and geometric losses        | `cosmos_framework/model/tokenizer/lidar/loss.py`               |
| Local tar manifest and clip adapter        | `cosmos_framework/data/tokenizer/lidar.py`                     |
| DDP training, checkpoint resume and export | `cosmos_framework/trainer/tokenizer/lidar.py`                  |
| Short-clip reconstruction evaluation       | `cosmos_framework/trainer/tokenizer/lidar_evaluation.py`       |
| Serving interface and preprocessing        | `cosmos_framework/model/generator/tokenizers/lidar/`           |
| Checkpoint and serving-format tests        | `cosmos_framework/model/tokenizer/lidar/post_training_test.py` |

The continuation recipe requires a local, compatible V1.2 initializer and its
expected SHA256. Record the artifact identity before launching; this recipe
does not select or download an initializer. Network-only warm start initializes
new optimizer, scheduler, latent statistics and power EMA history. A full resume
uses the complete checkpoint written by the native runner.

The local adapter uses explicit measured-LiDAR tar manifests and deterministic
first-eight-sweep windows. It prepends a duplicate first sweep for nine model
frames and repeats the last sweep for short clips. Select recording/session
boundaries deliberately and inspect decoded samples; distinct paths alone do
not establish a disjoint held-out corpus. Long recordings require explicit
window preparation to control training exposure.

Evaluate live and EMA exports from the same checkpoint with identical held-out
recordings and settings. The native evaluator uses FP32 posterior means and
reports range MAE on measured-valid rays and validity IoU, excluding the prepended
duplicate. It is a short-clip check, not a qualification of long-sequence
reconstruction, streaming behavior, point-cloud geometry or downstream generator
quality. Inspect reconstructions and BEV alongside scalar metrics before choosing
a serving candidate.

CPU tests can establish loss/backpropagation, scheduling, checkpoint restoration
and serving-format behavior. They do not establish full-resolution GPU memory,
throughput, distributed numerical stability or convergence. Before a continuation
campaign, complete a bounded GPU train/save/resume check and evaluate the exported
candidate on held-out measured recordings. Any hardware, precision, topology,
batch, objective or learning-rate change needs corresponding validation.

See the [public validation record](../../../../../examples/lidar_tokenizer/VALIDATION.md)
for the executed checks and their limits. A CPU test of a smaller consistent
network does not qualify the full V1.2 GPU training command.

Keep network, projection, compression and latent-statistics metadata together
when loading an export. A tokenizer change alters a generator's latent space;
qualify or adapt the consuming generator before replacing its bundled tokenizer.
