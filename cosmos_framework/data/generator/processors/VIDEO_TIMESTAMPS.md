# Reasoner video timestamp convention

Encoded videos decoded for Qwen3-VL and native Edge carry the selected original
frame indices, original frame count, and source average FPS through BytesToMedia,
TokenizeData, and the processor wrapper. This is the default for these processors.
The default `data_setting.video_timestamp_mode=qwen_index` selects this clock.
Frame sampling is unchanged. Explicit `legacy_fps` restores historical synthetic
FPS timing; optional native Edge `source_pts` uses first-frame-relative PTS.
Set the same policy on BytesToMedia and TokenizeData.

Frame selection still uses the existing rounded `linspace(start, end - 1, count)`.
Even spacing describes that selection policy. It does not imply that the selected
frames occurred at `0, 1 / requested_fps, 2 / requested_fps, ...`.

For each temporal patch, timestamps follow Qwen's source-index convention:

1. Divide each selected source frame index by source FPS.
2. Pad an incomplete temporal patch by repeating its final frame index.
3. Use the midpoint of the first and last times in each temporal patch.
4. Render the timestamp to one decimal place.

Source crop offsets are retained. Edge retains temporal patch size 1; Qwen3-VL
normally uses 2. Spatial merge size does not determine time grouping. For example,
indices `[0, 8, 20, 30]` at 30 FPS render as `[0.0, 0.3, 0.7, 1.0]` on Edge.
The images selected by the sampler are unchanged.

This matches Qwen's TorchCodec/Decord index/FPS convention. It is distinct from
using decoder presentation timestamps (PTS): those can differ on variable-rate
video. The optional crop-relative `source_pts` policy is documented in
[Video timestamp policies](../../docs/reasoner/source_video_timestamps.md). <!-- rumdl-disable-line MD057 -->

Existing manually supplied frames without source metadata retain their local
frame-sequence/FPS fallback. Other processor families retain their existing decoder
contract. Explicit source metadata must match the frame count, contain ordered
in-bounds indices, and use finite positive FPS; it cannot silently fall back.
Metadata is copied before processing because upstream timestamp padding can mutate
index lists in place.

Temporal-label augmentors use the same displayed patch timestamps. Audio extracted
from a cropped video carries its source start time so audio tokens and video patch
timestamps use one clock. Repeated frame times retain repeated anchors without
dropping audio tokens; decreasing anchors are invalid.

For multiple videos in one message, extracted audio uses the clock of its own
media key. Separate audio blocks may follow other videos, as in
`[video_a, video_b, audio_a]`, provided their own video has already appeared in
that message. `interleaved_av` requires matching adjacent `[video, audio]` pairs
because audio segments are inserted into that video's token sequence. Standalone
waveform entries retain positional pairing with the preceding video. Audio-only
messages retain their local audio clock.

`data_setting.qwen_video_temporal_mode=framewise` repeats each selected frame
across the pretrained temporal patch. In `qwen_index` mode, its absolute source
index is repeated with the pixels, so each temporal patch and label use that
frame's source time. Legacy framewise inputs retain their clip-relative index/FPS
clock. Framewise audio remains unsupported, as on the target branch.

## Reference

- [Qwen video input implementation](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py)
- [Transformers 4.57.6 Qwen3-VL processor](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3_vl/processing_qwen3_vl.py)

## Validation

`video_source_metadata_test.py` compares directly to the installed Transformers
Qwen3-VL timestamp helper, including odd counts, repeated indices and crop offsets.
Decoder tests compare selected pixels and source indices on generated CFR/VFR
videos. Wrapper and training augmentor tests exercise metadata propagation and
cropped audio partitioning.

`projects.cosmos3.cosmos3.scripts.validate_reasoner_video_timestamps` runs a bounded
one-GPU (or four-GPU) training smoke test with real encoded training videos and
their SFT annotations. It requires exact full-model checkpoint loading, Qwen timestamp parity,
unchanged pixel tensors, nonempty assistant supervision, finite loss/gradients, and
actual optimizer updates on every rank. This checks execution correctness; it does
not measure reasoning quality.

A minimal GPU execution check uses one GPU, one or two examples, and one optimizer
step with `--steps 1`. CPU tests cover the timestamp formulas, all three policy
paths, CFR/VFR decoding, crop offsets, label clocks, and paired audio partitioning;
this preprocessing change does not require a multi-GPU run. The smoke script
expects native Edge checkpoint metadata and a JSON list of samples with
`video_path`, `video_id`, and a two-turn string-content `conversation`.

```bash
torchrun --standalone --nproc_per_node=1 -m \
  projects.cosmos3.cosmos3.scripts.validate_reasoner_video_timestamps \
  --checkpoint /path/to/native-edge-checkpoint \
  --samples /path/to/samples.json --output /tmp/timestamp-smoke --steps 1
```
