# Measured runs: Qwen3-VL-8B SFT on RoboVQA

Two runs of this recipe at seed 0, each on one NVIDIA GB300 (256,703 MiB, aarch64, driver 595.91.07, CUDA 13.0),
directly on the host without a container, in October 2026, with the framework at commit
`9a70e64d2ad7ff1417484d12d10fb121b270baaf`. The runs used the commands of [`../scripts/run.sh`](../scripts/run.sh)
from an earlier version of that script, with the same training config, data preparation and evaluator, and wrote the
HF export through the materializer with 5 GB shards (the change that ships with this skill).

## Versions

| Environment | Python | Key packages |
| --- | --- | --- |
| SFT (`docker/posttrain/environment`, groups `sft`, `cu130-runtime`, `cu130-aarch64`) | 3.13.15 | torch 2.11.0+cu130, transformers 5.8.1, torchcodec 0.11, apex `575968bc` built from source |
| Evaluation | 3.12 | torch 2.10.0+cu130, torchvision 0.25.0, torchcodec 0.10.0, transformers 4.57.6, nvidia-npp 13.0.0.50 |

Pins: `Qwen/Qwen3-VL-8B-Instruct` at `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`; `Tianli/robovqa` at
`eb5819490f0d9ec18a79774233d4b1c9ebd52eea`.

## Results on the 1,000 held-out clips

| | exact match | token-F1 | yes/no accuracy (597 questions) |
| --- | --- | --- | --- |
| Qwen3-VL-8B-Instruct, zero-shot | 0.548 | 0.676 | 0.918 |
| run 1, after one epoch | 0.632 | 0.809 | 0.953 |
| run 2, after one epoch | 0.620 | 0.798 | 0.946 |

- The zero-shot answers are deterministic: both runs (two independent setups) gave the same 1,000 answers.
- Training is not bit-reproducible: gradient norms differ from update 2 on, and the two fine-tuned models give the
  same answer on 763 of the 1,000 clips. A Hugging Face + TRL full SFT of the same data and epoch on 8 H100s gave
  0.633 ± 0.007 over three seeds (outside this skill; a reference point only).
- Part of the gain is the model learning RoboVQA's answer style (short, lower-case, imperative), which exact match
  rewards; yes/no accuracy, which reads only the first yes or no, rose from 0.918 to 0.946 and 0.953.

## Time and memory

| Step | Run 1 | Run 2 | Notes |
| --- | --- | --- | --- |
| 1 + model download | 704 s | 684 s | the apex build takes about 10 minutes of it |
| 2 + 3 (dataset download and training rows) | 1,367 s | 1,424 s | 48 processes; 7,998 frame files, 35,369,466,594 bytes, 119,827 frames |
| 4 preflight | 753 s | not run | CPU |
| 5 before | 802 s | 856 s | the evaluator alone 670 s and 761 s |
| 6 train | 7,512 s | 8,064 s | median 60.4 s and 63.1 s per update (updates 2 to 125); update 1 took 107 s and 134 s |
| 7 export | 121 s | 119 s | CPU; 16.3 GiB, 750 tensors in 4 shards |
| 8 after | 605 s | 657 s | the evaluator alone 526 s and 557 s |

- Training memory: torch peak 138,981.7 MiB allocated and 143,566 to 143,716 MiB reserved; nvidia-smi peaked
  144,357 MiB above the idle baseline (run 1).
- GPU time: about 2.5 GPU-hours (steps 5, 6 and 8).
- Disk: about 110 GB in all; frames 33 GB, model 17 GB, checkpoint 17 GB, export 17 GB.

## Training loss (run 2)

| update | 1 | 2 | 3 | 4 | 5 | ... | 120 | 121 | 122 | 123 | 124 | 125 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `train/loss_avg` | 1.041 | 0.630 | 0.815 | 0.602 | 0.616 | ... | 0.698 | 0.695 | 0.543 | 0.544 | 0.499 | 0.482 |

Run 1 had 1.041 at update 1 and 0.487 at update 125; the mean of updates 1 to 10 was 0.717 and of 116 to 125 0.591.

## Example answers (run 1)

The first held-out clip of each task family, fixed before any fine-tuned answer existed. ✓ / ✗ is the scorer's
normalised exact match.

| task | question | reference | zero-shot | fine-tuned |
| --- | --- | --- | --- | --- |
| success | Put the orange pen in the penholder Q: satisfied? | yes | ✓ yes | ✓ yes |
| affordance | put the box on the table Q: possible right now? | no | ✗ yes | ✓ no |
| planning | current goal is: Please place the Peas and strawberries into the bowl Q: immediate next step? | place strawberry in the bowl | ✗ Put the strawberry in the paper bowl | ✗ put the strawberry in the bowl |
| past_description | Q: what just happened? | pick up the white and green packet | ✗ The robot arm picked up a green packet from the shelf. | ✗ place the packet in the shelf |
| future_prediction | Q: what is likely to happen next? | drop the brown sachet in the plastic glass | ✗ Place the brown sugar packet in the glass | ✗ put the brown packet in the glass |

## Troubleshooting

Each entry: symptom, cause, check, fix. Paths are relative to the repository root.

- **`libnppicc.so.13: cannot open shared object file`** when TorchCodec loads. Cause: NVIDIA NPP is not on the
  library path. Check: `ls docker/posttrain/environment/.venv-sft/lib/python3.13/site-packages/nvidia/cu13/lib | grep nppicc`.
  Fix: re-run `run.sh setup`, which installs `nvidia-npp` in both environments; `run.sh` puts it on
  `LD_LIBRARY_PATH` for every command that decodes video. Never clear `LD_LIBRARY_PATH`.
- **`ImportError: libc10.so` importing `amp_C`.** Cause: apex's extensions link against torch's libraries. Check:
  `docker/posttrain/environment/.venv-sft/bin/python -c "import torch, amp_C"`. Fix: import `torch` first.
- **apex missing after running `uv sync` by hand.** Cause: an exact sync uninstalls packages the lock does not list.
  Check: the same import. Fix: re-run `run.sh setup` (it syncs with `--inexact` and rebuilds apex only when missing).
- **The SFT environment was built on Python 3.12.** Cause: with `--project`, uv does not read the checkout's
  `.python-version` and takes any interpreter in range. Check: `docker/posttrain/environment/.venv-sft/bin/python
  --version`. Fix: delete `docker/posttrain/environment/.venv-sft` and re-run `run.sh setup`, which pins
  `--python 3.13`.
- **`[materialize] FATAL ... model.safetensors.index.json missing`.** Cause: Transformers 5.8.1 writes a model under
  50 GB as one file without an index, and the materializer's verifier requires the index. Fix: a framework checkout
  that writes 5 GB shards (the change that ships with this skill), then re-run `run.sh export`.
- **The training step prints nothing for about three minutes.** Cause: model load and the first update's warm-up.
  Check: `nvidia-smi` shows the process growing toward 140 GB. Fix: none; wait for `step 1/125`.
- **`--validate-rows` prints `train: dp_world not checked`.** Expected outside torchrun; one GPU divides every row
  count.
