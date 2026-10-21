# Adapt Sim-Auto training

Start with the [three-stage recipe](canonical-recipe.md). Keep the model, sensor
format and checkpoint handoffs consistent; adjust resource use and learning
settings for your dataset. Save the resolved configuration for each stage.

## Keep the model and data aligned

- Use the regular-network predecessor and teacher specified by the
  [checkpoint handoffs](checkpoints.md).
- Preserve camera IDs, paired control/target roles, range units and synchronized
  30-FPS RGB / 10-Hz LiDAR.
- Keep deduplicated 0.4-second cross-view attention, three RGB latent frames per
  causal chunk and the four-step student schedule.
- Preserve the TF → TF-dCM → SF-DMD objectives and their trainable/frozen modules.
  Export rolling settings as KV 2 / sink 1 / segmented, as shown in the export guide.

## Settings to review

| Setting                        | Reference value                                                   | How to use or adjust it                                                                                                                        |
| ------------------------------ | ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Training duration              | 4000 total trainer iterations per stage                           | Set the endpoint using dataset exposure and held-out quality. A resumed run continues to that absolute endpoint.                               |
| Learning rates and optimizers  | Stage-specific values in the recipe                               | Start with those values; compare one optimizer change at a time while tracking losses, gradients and held-out outputs.                         |
| Parallelism                    | CP32/FSDP64; accumulation 1                                       | Match the mesh to your allocation. Check memory, effective batch and save/resume. Use CP4/FSDP4 for the documented short check.                |
| Global packed batch            | `world_size / CP × grad_accum_iter`                               | Recalculate after topology changes. Samples and tokens per pack vary. Use integer accumulation to preserve the intended batch where practical. |
| Token budget                   | 1.6M model/loader tokens; Stage 3 max one sample                  | Use 100k/max one sample for the short check; size full runs for the selected cameras, horizon, resolution and caption length.                  |
| Source horizon                 | TF/TF-dCM: 400 requested frames; SF-DMD: 96 requested, 93 aligned | Keep RGB/LiDAR clocks aligned and Stage 3 rollout lengths equal. The short check uses 24 requested / 21 aligned RGB frames.                    |
| Resolution and camera sampling | 60% 480p / 40% 720p; bundle-defined camera subsets                | Preserve camera mappings and record the sampled resolution/view distribution when adapting the data.                                           |
| Caption dropout                | Sample 0.2; per-view 0.2                                          | Start with the recipe values and monitor text adherence on held-out scenes.                                                                    |
| Initial conditioning           | 0/1/2 RGB latents with probabilities 0.7/0.2/0.1                  | Keep the LiDAR prefix aligned with the camera prefix.                                                                                          |
| Whole-LiDAR dropout            | 0.1                                                               | Drop the LiDAR target and control together. Inspect batches with and without LiDAR.                                                            |
| RGB/LiDAR loss scales          | TF: 1/2; TF-dCM and SF-DMD: 100/200                               | Retain the native `local_item_mean` reduction and monitor both sensor losses.                                                                  |
| SF-DMD cadence                 | One student and five critic updates per six iterations            | At 4000 iterations this gives 667 student and 3333 critic updates. Resume both optimizer/scheduler states and phase counters.                  |
| Precision and EMA              | BF16 compute, FP32 master weights; EMA rate 0.1                   | Preserve regular-weight stage handoffs and the recipe's EMA initialization/resume policy.                                                      |
| Compilation                    | Stage-specific recipe settings                                    | The short check runs the student eagerly; Stage 3 retains fake-score compilation. Check finite forward/backward steps on the selected runtime. |
| Checkpointing                  | Every 100 iterations; asynchronous in full recipes                | Use synchronous endpoint saves for the short check. Provision host RAM and shared memory before enabling async saves.                          |
| Seed and output                | Trainer seed 0; a unique run identity                             | Keep seeds and inputs fixed for comparisons. Use the same run identity for resume and a new one for a new experiment.                          |

Follow the [training runtime guide](runtime.md) when sizing the allocation. Check training, save/reload and held-out inference
with the final data and workload.
