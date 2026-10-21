# Caption your videos

Caption the target RGB video's appearance and motion. Review the text before
pairing it with controls: depth, edges and segmentation cannot establish
target colors or texture.

## Use the supplied example

[Download `example2` from Hugging Face](../../shared/hf-sample.md) for the complete
workflow. Its directory contains:

- `example2.mp4`: target RGB, 121 frames at 30 FPS and 1920 × 1080 pixels.
- `example2_depth.mp4` and `example2_segmentation.mp4`: aligned controls.
- `example2_prompt.txt`: the source prompt.
- `example2_upsampled_prompt.json`: the direct structured caption object.

The native training and inference paths derive edge and blur from RGB. The
saved JSON's `temporal_caption` is the dense branch for the 93:7 training
recipe; the separate source `.txt` is not that branch. Use the saved caption
without contacting a service, or use the command below to caption the RGB
again. Keep newly generated captions in their own directory.

`example1` is exclusively for long-rollout inference. Its JSON is an envelope;
inference reads its `prompt` string. See the [prompt guide](../../shared/prompt-guide.md)
for both file formats.

## Caption a new clip

Prepare the [Framework environment](../../shared/setup.md). Use an
existing video-capable endpoint or start the [local server](local-caption-server.md).
The endpoint must accept OpenAI chat-completions requests with `video_url`,
`top_k` and `min_p`. For local files, the client sends `file://` URLs, so the
server must read the same absolute media paths. The example uses a local
server with access limited to the downloaded media directory.

Run from `$FRAMEWORK_DIR` with its environment active. To caption the downloaded
`example2` RGB:

```bash
cd "$FRAMEWORK_DIR"
python -m cosmos_framework.scripts.caption_from_video \
  --video "$SIM_TRANSFER_DATASET/example2.mp4" \
  -o "$SIM_TRANSFER_OUTPUT/captions" \
  --server http://localhost:8000/v1 \
  --model Qwen/Qwen3-VL-8B-Instruct-FP8 \
  --max-workers 1
```

Use the model name served by your endpoint. The [packaged prompt](video_captioner.txt)
requests structured visual analysis and a dense paragraph. The client parses
`<scene_draft>` JSON, inserts `<final_prompt>` as `temporal_caption`, and adds
probed media fields using Framework's `StructuredCaption` schema. A successful
request writes:

```text
example2/
  caption.json
  caption.txt
  sample_args.json
```

Review subjects, appearance, lighting, camera movement and chronological actions
against the selected RGB interval. Correct invented details, leave unknown
details empty, and keep `caption.json.temporal_caption` equal to `caption.txt`.
`sample_args.json` records the source and extracted prompt.

For another dataset, pass an RGB file or directory to `--video`. Each filename
stem becomes an output name and must be unique. Use a fresh output directory
to keep existing captions. Check the logged successful/total count: individual
request failures do not change the client's exit code.

## Preserve the caption recipe

After text dropout, the reference loader selects structured captions 93% of
the time and the same object's nonempty `temporal_caption` 7%. Native configs
set text dropout to 0.1 for Stages 0–2 and zero for Stage 3. Use the same caption
pair for all four control modalities over the interval.

For `example2`, that interval is frames 0–120, expressed by exclusive bounds
`[0,121)`. Its timestamp span is 4 seconds and frame-count/FPS duration is about
4.033 seconds. Local preparation removes saved output metadata; the loader
appends the actual sampled duration, FPS and canvas.

Follow [local data preparation](../../shared/local-dataset.md) to attach reviewed
structured captions to native Sim-Transfer data. Local JSONL uses
`caption_path`, `caption_start: 0` and `caption_end` equal to the decoded frame
count. Cut aligned RGB/control clips first for shorter annotations. Captions
are embedded in `samples.json`; editing a standalone caption after preparation
does not update the dataset.

Use [generic manifest assembly](caption-manifest.md) only for loaders that
accept generic video-caption JSONL. That converter does not build Sim-Transfer
controls or implement the structured/dense 93:7 mixture.

## References

- [Hugging Face examples](../../shared/hf-sample.md)
- [Prompt and caption formats](../../shared/prompt-guide.md)
- [Optional local captioning server](local-caption-server.md)
- [Captioner prompt template](video_captioner.txt)
- [Local training bundle preparation](../../shared/local-dataset.md)
- [Generic video-caption manifest assembly](caption-manifest.md)
