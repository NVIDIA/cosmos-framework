# Assemble a generic video-caption JSONL manifest

Use [local data preparation](../../shared/local-dataset.md) for Sim-Transfer.
Its locally created manifest links reviewed structured captions to aligned
controls and uses exclusive-end whole-clip bounds. The HF `example2` assets
supply the videos and prompts from which the local dataset is built.

Use Framework's generic SFT converter only with a loader that accepts
`t2w_windows` with inclusive ends. It does not create Sim-Transfer bundles,
paired controls or the 93:7 caption policy.

```text
data/
  videos/example.mp4
  captions/example/
    caption.json
    caption.txt
    sample_args.json
```

Caption directory names must match video filename stems. The converter searches
`.mp4`, `.mov`, `.avi`, `.mkv` and `.webm` directly inside the video directory.

```bash
cd "$FRAMEWORK_DIR"
python -m cosmos_framework.scripts.captions_to_sft_jsonl \
  --captions-dir data/captions \
  --videos-dir data/videos \
  -o data/video_dataset_file.jsonl
```

Each record contains `uuid`, `duration`, `width`, `height`, `vision_path`, and
`t2w_windows`. The converter writes paths relative to the output JSONL's
parent. Move the manifest and its referenced media together. It creates one
whole-video window, from frame 0 to the final frame **inclusively**, with
`temporal_interval: 1`.

It emits the object from `caption.json` under `caption_json` and dense text
from `caption.txt` under `caption`. If dense text is absent, it can use the
JSON object's `temporal_caption`. Missing or malformed structured JSON can
produce a dense-only record; review warnings and the output representations.
The SFT loader's structured-first fallback does not implement Transfer's 93:7
mixture. This converter also accepts directories containing only `caption.json`.

The filters reject missing media, probe failures, duration above 61 seconds,
and windows shorter than
`max(61, --num-video-frames)`. The default `--num-video-frames -1` retains the
61-frame minimum. Optional `--min-short-edge` adds a dimension filter. These
are generic SFT limits; local Transfer has its own horizon policy. The converter
rejects absent caption content but does not validate the structured schema;
review `caption.json` before conversion.

The native converter writes `<output>.summary.json` with retained counts and
exclusion reasons. A run with no valid records exits with an error.
