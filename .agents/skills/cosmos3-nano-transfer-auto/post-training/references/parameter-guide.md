# Adapt the recipe to your data and hardware

Use the [four-phase recipe](canonical-recipe.md) to train from Cosmos3-Nano.
For direct fine-tuning, load the released Transfer-Auto weights and start a new
optimizer, scheduler, iteration counter and EMA.

## Keep the model and data aligned

- Follow the [data contract](data-contract.md): physical camera IDs, synchronized
  controls/targets, range projection and captions grounded in the recorded scene.
- Keep the model configuration and tokenizers matched to the weights. All joint
  phases use deduplicated cross-view attention over the preceding 0.4 seconds;
  phase 1 uses single-view/two-way attention.
- Preserve the recipe's trainable modules, conditioning masks and loss reduction.
- Follow the exact checkpoint handoffs for the curriculum. Same-phase resume
  restores that run's model, optimizer, scheduler, counter, EMA and loader state.

## Settings to review

| Setting                             | Reference value                                                | How to use or adjust it                                                                                                                             |
| ----------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Curriculum task mixture             | Phase table in the recipe                                      | Track actual RGB/LiDAR and resolution exposure when changing the available data.                                                                    |
| Learning-rate schedule              | Linear schedule with phase-specific warmup and decay           | Reset scheduler/trainer counters at phase 1 → 2; preserve them across phases 2 → 3 → 4. Retain inherited EMA age. For a new fine-tune, start fresh. |
| Fine-tuning learning rate           | Reference schedule floor: `1e-5`                               | Use as a starting comparison point; tune using loss/gradient stability and held-out quality. Set duration from dataset size and exposure.           |
| EMA                                 | Power-law profile rate 0.1                                     | Retain the age offset when loading inherited EMA; initialize a new EMA for fresh fine-tuning.                                                       |
| Noise, conditioning and dropout     | Phase-specific recipe values                                   | Start with the supplied probabilities. Compare changes using per-task losses, text/control adherence and fixed held-out scenes.                     |
| Cameras, resolution and clip length | Phase-specific horizons; 480p/720p                             | Use calibrated views and synchronized intervals. Check the intended longest/largest batch when sizing memory.                                       |
| Packing ceiling                     | 240k / 300k / 1.5M for separate / short-joint / long-joint     | Include controls, targets and text in the token budget. Inspect packed lengths and rejected samples.                                                |
| Parallelism                         | CP32/FSDP32                                                    | Choose a mesh supported by your allocation and check memory, effective batch and save/resume.                                                       |
| Effective batch                     | 16 clips per update at the reference layout                    | Calculate `world_size / CP × clips_per_replica × accumulation` and verify the actual packed batch.                                                  |
| Gradient accumulation               | 1                                                              | Increase it to preserve effective batch when reducing independent data replicas, where practical.                                                   |
| Precision and recomputation         | BF16 compute, FP32 masters; selective activation checkpointing | Check finite forward/backward steps and memory after changing recomputation or offload settings.                                                    |
| Loader workers                      | Start conservatively                                           | Measure I/O and host memory before increasing concurrency. Track retries and source exposure.                                                       |
| Saves, previews and seed            | Recipe save intervals; seed 0                                  | Save at short-check endpoints; keep comparison inputs and seeds fixed. Confirm checkpoint completion and reload.                                    |

For example, halving independent data replicas and doubling gradient accumulation
retains the same logical batch. Keep learning-rate and scheduler changes explicit.
Use the [training guide](../../../../../docs/training.md) for parallelism and memory settings.
