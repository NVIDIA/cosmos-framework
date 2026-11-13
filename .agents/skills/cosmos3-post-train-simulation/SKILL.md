---
name: cosmos3-post-train-simulation
description: >
  Entry skill for Cosmos3 Simulation post-training in Cosmos Framework: routes a request to one of three block skills
  that adapt a Simulation model to your data. Auto (cosmos3-post-train-simulation-auto): fine-tune Transfer-Auto on
  paired driving data, distill it into a Sim-Auto student. Sim-Bimanual (cosmos3-post-train-simulation-bimanual):
  train from Cosmos3-Nano. Sim-Transfer (cosmos3-post-train-simulation-transfer): train from converted Cosmos3-Nano
  weights. States what the blocks share: stage names, the weights each stage starts from, resume rules. Use when the
  user asks "post-train a Cosmos3 Simulation model", "which Simulation model can I fine-tune or distill", "which
  checkpoint do I start Simulation training from" or "what are the Simulation training stages". Not for inference with
  released checkpoints (cosmos3-inference-simulation) or generator or reasoner supervised fine-tuning
  (cosmos3-post-training).
---

# Cosmos3 Simulation Post-Training: Entry

This is the entry skill for Cosmos3 Simulation post-training: training, fine-tuning or distilling one of the four
Cosmos3-Nano Simulation models on your own data in Cosmos Framework. It runs no command: it names the block skill for
each model and starting checkpoint, states what the blocks share, and hands the request to one of three block skills
(Auto, Sim-Bimanual, Sim-Transfer). Every statement below about the blocks is taken from those block skills; where
they differ, this skill says so and names the place in the block.

## When to use this skill

- Use when a request is about post-training a Cosmos3 Simulation model and the model or the starting checkpoint is
  not settled: pick the row under "Route to a block skill", then follow that block skill from the top.
- Use for what the blocks share: "Setup and data", "Stages and starting weights", "Checkpoint and resume rules".
- Not for inference with the released checkpoints: `../cosmos3-inference-simulation/SKILL.md` (see "Try the released
  checkpoints first").
- Not for supervised fine-tuning (SFT) of the Cosmos3 generator or reasoner, which is
  `../cosmos3-post-training/SKILL.md`: the Simulation blocks call some of the same Framework scripts
  (`convert_model_to_dcp`, `export_model`, and `train --sft-toml` for Sim-Bimanual), but with their own recipes, local
  bundles and checkpoint handoffs.

## Route to a block skill

Open the block skill of the chosen model and follow it from the top; for the two Auto models, open the named section.

| Model         | What the block does                                                                                                                                 | Starts from                                                                                                  | Block skill                                                                                                                                   | When training fails                                                                                                   |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Transfer-Auto | Fine-tunes the released Cosmos3-Nano-Transfer-Auto model on paired driving data, or runs the four-phase training curriculum; exports the checkpoint | Released Transfer-Auto checkpoint (fine-tuning) or Cosmos3-Nano checkpoint (four phases)                     | [cosmos3-post-train-simulation-auto, section Fine-tune Transfer-Auto](../cosmos3-post-train-simulation-auto/SKILL.md#fine-tune-transfer-auto) | [Auto post-training troubleshooting](../cosmos3-post-train-simulation-auto/references/troubleshooting.md)             |
| Sim-Auto      | Distills Transfer-Auto into a Cosmos3-Nano-Sim-Auto student in three stages (TF, TF-dCM, SF-DMD); exports the student                               | Compatible Transfer-Auto checkpoint (released, or your adapted export), not plain Cosmos3-Nano               | [cosmos3-post-train-simulation-auto, section Distill Sim-Auto](../cosmos3-post-train-simulation-auto/SKILL.md#distill-sim-auto)               | [Auto post-training troubleshooting](../cosmos3-post-train-simulation-auto/references/troubleshooting.md)             |
| Sim-Bimanual  | Trains Cosmos3-Nano-Sim-Bimanual through Stages 0 to 3 with four native stage TOMLs; exports the Stage 3 student                                    | Cosmos3-Nano and a completed local bundle                                                                    | [cosmos3-post-train-simulation-bimanual](../cosmos3-post-train-simulation-bimanual/SKILL.md)                                                  | [Sim-Bimanual post-training troubleshooting](../cosmos3-post-train-simulation-bimanual/references/troubleshooting.md) |
| Sim-Transfer  | Trains Cosmos3-Nano-Sim-Transfer through Stages 0 to 3 on aligned control, RGB and caption data; runs inference from the Stage 3 checkpoint         | Compatible base regular weights (the pinned Cosmos3-Nano model converted to DCP) and a prepared local bundle | [cosmos3-post-train-simulation-transfer](../cosmos3-post-train-simulation-transfer/SKILL.md)                                                  | [Sim-Transfer post-training troubleshooting](../cosmos3-post-train-simulation-transfer/references/troubleshooting.md) |

The two Auto sections are ordered. A Sim-Auto student of your own is distilled from the released Transfer-Auto
checkpoint or from your adapted Transfer-Auto export, so to adapt Sim-Auto to your data or rig, post-train
Transfer-Auto first, then distill that checkpoint. Supervised fine-tuning of the released Sim-Auto student is not a
workflow the Auto block documents.

## What the blocks share

### Setup and data

- Each block's "Prerequisites" starts from `../cosmos3-setup/SKILL.md` for the Framework checkout and environment,
  then adds the model's own requirements.
- The commands run from the Framework checkout root with its `.venv` activated and
  `COSMOS_INTERNAL=0 COSMOS_TRAINING=1` exported, and training launches under `torchrun`.
- Training data is a local training bundle that you prepare from your own data; each block has the preparation steps
  and the checks for its bundle.
- `IMAGINAIRE_OUTPUT_ROOT` is the training-output root, on storage shared by every rank.

### Stages and starting weights

Sim-Auto, Sim-Bimanual and Sim-Transfer use the same three stage names, numbered 1 to 3: teacher forcing (TF), TF-dCM
and SF-DMD. Sim-Bimanual and Sim-Transfer train a Stage 0 first. Sim-Auto has no Stage 0 of its own: its Stage 1
student starts from Transfer-Auto. The cells follow each block's own stage table.

| Stage              | Sim-Auto                                                                                                 | Sim-Bimanual                                                                              | Sim-Transfer                                                                        |
| ------------------ | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| 0                  | None; the converted Transfer-Auto initializer is bound as `SIM_AUTO_STAGE0_CHECKPOINT`                   | Representation adaptation, from Cosmos3-Nano                                              | Bidirectional teacher, from compatible base regular weights                         |
| 1, teacher forcing | Student from Transfer-Auto; no teacher                                                                   | From the Stage 0 checkpoint                                                               | From your Stage 0 EMA                                                               |
| 2, TF-dCM          | Student from the accepted Stage 1; teacher is the frozen Stage 1                                         | Stage 1 checkpoint for student and frozen teacher                                         | Your Stage 1 EMA for student and frozen teacher                                     |
| 3, SF-DMD          | Student from the accepted Stage 2; teacher is the original bidirectional Transfer-Auto                   | Stage 2 checkpoint for student; Stage 0 checkpoint for reference/critic                   | Your Stage 2 EMA for student; your Stage 0 EMA for teacher and score model          |
| Stated in          | [Three stages](../cosmos3-post-train-simulation-auto/references/sim-auto-training-notes.md#three-stages) | [Continue training](../cosmos3-post-train-simulation-bimanual/SKILL.md#continue-training) | [Select a stage](../cosmos3-post-train-simulation-transfer/SKILL.md#select-a-stage) |

In all three chains Stage 3 alternates one student update with five updates of a second network (the blocks call it
critic, fake-score network or score model), and each block tells you to check or count both kinds of update.

Transfer-Auto, the bidirectional model that Sim-Auto is distilled from, has phases instead of these stages: four
phases from Cosmos3-Nano (separate sensors, short joint, long joint, long-joint continuation), or fine-tuning of the
released checkpoint with the final-model recipe `transfer_auto_phase4`. Phase commands and handoffs:
[four-phase recipe](../cosmos3-post-train-simulation-auto/references/transfer-auto-recipe.md).

### Checkpoint and resume rules

These hold in every block. The commands, variable names and weight variants are in the block.

1. A new chain starts from a checkpoint converted to DCP: model weights, without optimizer or trainer state.
2. A new stage or phase is a new run with its own run name. In the Sim-Auto, Sim-Bimanual and Sim-Transfer chains it
   starts fresh optimizer, scheduler and iteration state. The Transfer-Auto phases restore and reset different state
   at each transition: the transition table under
   [Configure the phases](../cosmos3-post-train-simulation-auto/references/transfer-auto-recipe.md#configure-the-phases).
3. Resuming the same stage or phase is a different operation: keep the run name and output root, and the complete
   checkpoint restores model and EMA, optimizer, scheduler, iteration and loader state.
4. `trainer.max_iter` is an absolute endpoint that includes restored iterations, not a count of additional ones.
5. Regular-weight loading stays strict.
6. Wait until checkpoint writes are complete before a restart and before the next stage loads the checkpoint.
7. Resolve the configuration with `--dryrun` before launching; a dry run loads no weights.
8. Each block has a short bounded run with a save and a resume or readback check, to do before a long run.

## What differs between the blocks

Read these in the block; do not carry a value from one block to another.

- **Allocation (GPU count, memory, storage):** what each block states is in its "Prerequisites" and, for the Auto
  block, also in its two launch steps: [Auto](../cosmos3-post-train-simulation-auto/SKILL.md#prerequisites),
  [Sim-Bimanual](../cosmos3-post-train-simulation-bimanual/SKILL.md#prerequisites),
  [Sim-Transfer](../cosmos3-post-train-simulation-transfer/SKILL.md#prerequisites).
- **Launcher entry point:** `cosmos_framework.scripts.train_experiment` in the Auto and Sim-Transfer blocks,
  `cosmos_framework.scripts.train --sft-toml` with four stage TOMLs in the Sim-Bimanual block. Process counts and
  topology overrides are in each block's launch step.
- **Weights handed to the next stage:** Sim-Auto: "Use regular-network weights between stages; do not substitute
  EMA." Sim-Transfer: "pass your selected EMA outputs through teacher forcing, four-step distillation and
  self-forcing". Sim-Bimanual: "A new stage uses the weight trees selected by its native builder, not an inferred
  EMA-to-regular conversion."
- **Weights that go to inference:** Transfer-Auto exports with EMA weights selected; Sim-Auto exports the regular
  Stage 3 student; Sim-Bimanual exports the Stage 3 student with student EMA selected; Sim-Transfer inference loads
  the Stage 3 student EMA directly from its DCP checkpoint. Sections:
  [Auto](../cosmos3-post-train-simulation-auto/SKILL.md#run-inference-with-your-checkpoint),
  [Sim-Bimanual](../cosmos3-post-train-simulation-bimanual/SKILL.md#export-and-check-inference),
  [Sim-Transfer](../cosmos3-post-train-simulation-transfer/SKILL.md#run-inference-from-your-trained-checkpoint).

## Try the released checkpoints first

Each block's starting-point table has a row for generating with the released checkpoint; the Auto block words it
"Try a released model before training". For that, start at the inference entry skill
[cosmos3-inference-simulation](../cosmos3-inference-simulation/SKILL.md): it chooses the model, states the shared
prerequisites and the rollout validation checklist, and routes to the inference block skills. The same inference
block skills run a checkpoint you trained; each post-train block has a section for that step.

## Related skills

| Skill                                      | When to use                                                                                                                                         |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `../cosmos3-inference-simulation/SKILL.md` | Inference with the released Simulation checkpoints: model chooser, shared prerequisites, validation checklist, router to the inference block skills |
| `../cosmos3-post-training/SKILL.md`        | Supervised fine-tuning (SFT) of the Cosmos3 generator (Cosmos3-Nano, or Cosmos3-Super with LoRA) and reasoner                                       |
| `../cosmos3-setup/SKILL.md`                | Installation and environment setup of the Cosmos Framework checkout                                                                                 |
| `../cosmos3-env-troubleshoot/SKILL.md`     | Environment failures that are not specific to one model: every block's "Troubleshooting" section points here                                        |
