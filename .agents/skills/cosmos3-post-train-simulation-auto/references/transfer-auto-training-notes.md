# Transfer-Auto fine-tuning notes

> **Skill note (not in the source):** these blocks are from the body of the Transfer-Auto post-training skill that this skill replaces (cosmos-framework `main` 45fecab), moved here word for word because SKILL.md has a size budget; only link targets were re-pointed. "After packaging the bundle" is a new heading over the paragraph that closes [Prepare a local training bundle](../SKILL.md#prepare-a-local-training-bundle) ("the builder" is its `auto_bundle` command); "Reproduce the four training phases" is the source's heading, which also stays in [SKILL.md](../SKILL.md#reproduce-the-four-training-phases) over a pointer. The rest of the body is in [SKILL.md](../SKILL.md#fine-tune-transfer-auto).

## After packaging the bundle

The builder copies media and tokenizer resources. It derives eligible camera
subsets from your available cameras; this changes exposure relative to an
eleven-camera corpus. Supply both resolution strata used by the recipe, or
remove the absent stream before constructing the loader. For this 720p/300-frame
example, use the [bounded sample settings](training-bundle-and-short-check.md#short-training-check).
Inspect a decoded sample and actual formatted batch before accepting the source.

## Reproduce the four training phases

Follow the [curriculum and handoffs](transfer-auto-recipe.md):

1. Cosmos3-Nano → separate RGB/LiDAR: 70k updates using single-camera RGB
   clips of up to 397 frames (400 requested) and separate 21-sweep LiDAR clips.
2. Separate-sensor checkpoint → short joint RGB/LiDAR: 61 RGB frames aligned
   with 21 LiDAR sweeps.
3. Short-joint checkpoint → long joint RGB/LiDAR: up to 397 RGB frames aligned
   with 133 sweeps, continuing to joint iteration 22,000.
4. Continue long joint training to joint iteration 50,000 with the same model,
   attention, dropout and schedule.

Separate-sensor tasks have different durations because they are independent
samples. Only the joint phases require matching RGB/LiDAR time windows.
Phases 2–4 use deduplicated cross-view attention over the preceding 0.4 seconds,
including the current time. Phases 3 and 4 both use 10% whole-LiDAR dropout.
