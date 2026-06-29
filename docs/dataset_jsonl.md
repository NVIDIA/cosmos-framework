# JSONL Dataset

This guide describes the JSONL dataset format.

Prerequisites:

- [Training](./training.md)

## Inference

Run inference on a single sample:

```shell
export DATASET_PATH=$(uvx hf@latest download --repo-type dataset nvidia/BridgeData2-Subset-Synthetic-Captions --revision 40d018ac1c1a2a4b9734f17fdb21f3d933c49a01 --quiet)/sft_dataset_bridge

torchrun --nproc-per-node=8 -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i "$DATASET_PATH/val/inference_prompt*/episode_049683_clip000.json" \
    -o outputs/train_inference \
    --checkpoint-path Cosmos3-Nano \
    --seed=0
```

- The ground truth video is in `${DATASET_PATH}/val/videos/`.
- The input image for I2V is in `${DATASET_PATH}/val/images/`.
- The input 5-frame video clip for V2V is in `${DATASET_PATH}/val/videos_5frames/`.

### Result Comparison

Each example below uses the following layout:

- Row 1 (T2V): ground truth video (left), before SFT (middle), after 500 iterations of SFT (right).
- Row 2 (I2V): input image (left), before SFT (middle), after 500 iterations of SFT (right).
- Row 3 (V2V): 5-frame input clip (left), before SFT (middle), after 500 iterations of SFT (right).

**episode_049683_clip000**

<details><summary><b>Input prompt</b></summary>

> A robotic arm with articulated joints and a gripping mechanism is positioned centrally on a wooden kitchen countertop, manipulating a small silver metal object while also interacting with scattered black coffee beans. The arm moves the metal object slightly, adjusting its position before shifting focus to the coffee beans, scattering and repositioning them with precision. The countertop is surrounded by kitchen elements, including a stove on the right, a microwave on the left, and two canned goods labeled "Tomato Juice" and "Baking Soda" in the background. The scene is illuminated by bright, even indoor lighting, casting minimal shadows, and the camera remains static throughout, offering a top-down perspective that emphasizes the robotic arm's movements. The composition centers on the robotic arm and its interaction with the metal object and coffee beans, with a shallow depth of field keeping the focus sharp on these elements while softly blurring the background. The overall atmosphere is technical and functional, highlighting the precision and control of the robotic manipulation within a domestic kitchen setting.

</details>

<video src="https://github.com/user-attachments/assets/4f7979c3-f892-4979-b74c-6829bb7dd5db" controls width="100%"></video>

**episode_009171_clip000**

<details><summary><b>Input prompt</b></summary>

> A robotic arm with a black and metallic gripper, accented with blue near its base, extends over a white rectangular tray filled with scattered brown almonds, methodically picking up and placing each almond in a precise line across the tray's surface. The arm moves with deliberate, controlled motion, shifting its position to reach different almonds while maintaining a top-down perspective that captures the entire workspace. The background reveals an indoor setting with a wooden table and various kitchen items, including a metal bowl and utensils, subtly visible behind the tray. The lighting is bright and evenly distributed, casting minimal shadows and highlighting the contrast between the white tray, the brown almonds, and the metallic sheen of the robotic arm. The camera remains static throughout, offering a wide-angle view that emphasizes the robotic arm's precision and the systematic rearrangement of the almonds, creating a clean, minimalist aesthetic that underscores the technical nature of the task. The scene unfolds as a continuous, uninterrupted sequence, showcasing the robotic arm's efficiency in organizing the almonds without any cuts or transitions.

</details>

<video src="https://github.com/user-attachments/assets/743796e8-4567-44c9-a3c7-4a51bcc6abc1" controls width="100%"></video>

## Format

Each `t2w_window` may carry **two** caption representations:

- **`caption_json`** — the canonical structured-JSON caption (an object). The SFT loader
  prefers this and trains on it by default, serialising it to the exact JSON string the
  model consumes at inference. The dense narrative is embedded inside it as
  `temporal_caption`, and the clip's media fields (`resolution`, `aspect_ratio`,
  `duration`, `fps`) describe the source clip.
- **`caption`** — the dense narrative string, kept as the **backup** the loader falls
  back to when `caption_json` is absent.

This keeps the post-training example aligned with inference, which also uses the
structured-JSON prompt format (see [Inference](#inference)).

Example sample (the structured object is abbreviated for readability):

```json
{
    "uuid": "episode_000015_clip000",
    "duration": 17.4,
    "width": 256,
    "height": 256,
    "vision_path": "videos/episode_000015_clip000.mp4",
    "t2w_windows": [
        {
            "start_frame": 0,
            "end_frame": 86,
            "temporal_interval": 1,
            "caption_json": {
                "subjects": [
                    {"description": "A black robotic arm with articulated joints and a metallic finish", "action": "grasps and relocates small black objects across a white tray"}
                ],
                "background_setting": "An indoor workspace with visible equipment on a wooden table",
                "cinematography": {"camera_motion": "static", "framing": "medium shot", "camera_angle": "slightly angled top-down"},
                "actions": [{"time": "0:00-0:17", "description": "the arm repeatedly lifts and repositions clusters of objects"}],
                "temporal_caption": "A black robotic arm, featuring articulated joints and a metallic finish, extends over a white tray ... maintaining a minimalist and functional aesthetic throughout.",
                "resolution": {"H": 256, "W": 256},
                "aspect_ratio": "1,1",
                "duration": "17s",
                "fps": 5
            },
            "caption": "A black robotic arm, featuring articulated joints and a metallic finish, extends over a white tray placed on a wooden table, manipulating small black objects that resemble beads or marbles. The arm moves with precision, grasping clusters of these objects, lifting them, and relocating them across the tray’s surface in a methodical manner, often shifting them from one side to another. The background reveals an indoor workspace with visible equipment, illuminated by bright, even lighting that casts minimal shadows, emphasizing the technical nature of the scene. The camera remains static throughout, offering a medium shot that centers on the robotic arm and tray, with a slightly angled top-down perspective that highlights the contrast between the black objects, white tray, and wooden table. The robotic arm’s movements are continuous and deliberate, showcasing its ability to handle and reposition the objects with accuracy, while the scene maintains a minimalist and functional aesthetic throughout."
        }
    ]
}
```

> Older datasets that contain only the dense `caption` field still work unchanged — the
> loader simply falls back to it.

## Video Captioning

If you have video sources and would like to synthesize caption annotations to build video–text pairs for training, follow this section for data preprocessing. The script sends each video directly to a Reasoner (vision-language model), which analyzes the visual content via a two-phase process (Phase 1: structured-JSON scene analysis → Phase 2: dense narrative rewrite) and saves **both** outputs: a `caption.json` (the canonical structured caption that the Cosmos3 training pipeline and inference consume, with the dense narrative embedded as `temporal_caption`) and a `caption.txt` (the dense narrative on its own).

The captioning prompt template is available at [`cosmos_framework/inference/defaults/video_captioner.txt`](../cosmos_framework/inference/defaults/video_captioner.txt).

### Server setup

The captioning script passes video files to vLLM via `video_url` content parts using `file://` paths, so the server must be able to read files from the local filesystem. We recommend [Qwen/Qwen3-VL-8B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-FP8) as the Reasoner. Start the server — this may take a couple of minutes:

```shell
uvx --with nvidia-cuda-runtime-cu12 \
    vllm@0.19.0 serve Qwen/Qwen3-VL-8B-Instruct-FP8 \
    --tensor-parallel-size 1 \
    --allowed-local-media-path /
```

The server is ready when you see `Application startup complete.`

### Run Video Captioning

Caption a single video:

```shell
python -m cosmos_framework.scripts.caption_from_video \
    --video /path/to/video.mp4 -o outputs/captions \
    --server http://localhost:8000/v1
```

Caption all `.mp4` files in a directory:

```shell
python -m cosmos_framework.scripts.caption_from_video \
    --video /path/to/videos/ -o outputs/captions \
    --server http://localhost:8000/v1
```

Caption videos listed in a JSONL manifest (each line must have a `vision_path` field pointing to a video):

```shell
python -m cosmos_framework.scripts.caption_from_video \
    -i samples.jsonl -o outputs/captions \
    --server http://localhost:8000/v1
```

Options:

| Flag                     | Default  | Description                      |
| ------------------------ | -------- | -------------------------------- |
| `--max_workers`          | `16`     | Concurrent API requests          |
| `--prompt_template_path` | built-in | Path to a custom prompt template |
| `--debug`                | `False`  | Save raw API responses           |

Each video produces an output directory containing `caption.json` (the canonical structured caption), `caption.txt` (the dense narrative), and `sample_args.json` (metadata).

### Create Dataset

After generating the captions, you will have videos and captions stored in the following file structure:

```
path/to/dataset/
└── captions/
└── videos/
```

To create a video dataset JSONL file for post-training, run the following command:

```
python -m cosmos_framework.scripts.captions_to_sft_jsonl \
    --captions-dir outputs/sft_dataset/train/captions \
    --videos-dir outputs/sft_dataset/train/videos \
    -o outputs/sft_dataset/train/video_dataset_file.jsonl
```

Each JSONL line contains both `caption_json` (structured, preferred for training) and `caption` (dense, backup) for every window, plus the corresponding video path. The converter mirrors the loader's silent filters (clips longer than 61 s, and windows shorter than `max(61, --num-video-frames)` frames) so the kept count matches what training will actually consume — pass `--num-video-frames` to match your recipe (the example recipe uses `-1`, i.e. all frames, so the default keeps short example clips). A sibling `<output>.summary.json` records the kept count and per-reason drop counts.

#### Align the inference prompts

To make the validation inference prompts use the **same** structured-JSON format as training, rewrite each `val/inference_prompt{,_i2v,_v2v}/<episode>.json` file's `prompt` field with the clip's structured caption:

```shell
python -m cosmos_framework.scripts.inference_prompts_to_json \
    --val-dir outputs/sft_dataset/val
```

This reads `val/captions/<episode>/caption.json` and replaces the (dense) `prompt` with the serialized structured JSON, preserving `resolution`, `aspect_ratio`, `num_frames`, `fps`, and `vision_path`. Pass `--dry-run` to preview.

### Create Dataset from a Cosmos-Curator output directory

If your training videos came from the [Cosmos-Curator](https://github.com/nvidia/cosmos-curator) splitting pipeline, you can build the SFT JSONL directly from curator's per-clip metadata — no separate captioning step, no `ffprobe` re-read, and multi-window captions are preserved.

**Prerequisite.** Curator must be invoked with `--upload-clip-info-in-chunks` so that `<curator_output>/metas_jsonl/v0/*.jsonl` is written. Without this flag the converter has no input.

```shell
python -m cosmos_framework.scripts.curator_to_sft_jsonl \
    --curator-output outputs/curator_split/ \
    -o outputs/curator_split/cosmos3_sft.jsonl
```

By default the converter resolves each window's caption to the first non-empty `*_enhanced_caption` value, falling back to `*_caption`. Use `--caption-model` / `--enhanced-caption-model` to pin a specific captioner (e.g. `--caption-model qwen --enhanced-caption-model qwen_lm`). Pass `--min-short-edge N` to drop low-resolution clips, or `--min-window-frames N` / `--max-duration-s S` to tune the loader-matching filters.

The converter mirrors `sft_dataset.py`'s silent filters (duration > 61.0 s, per-window frames < 61) so dataset counts match what training will actually consume. A sibling `cosmos3_sft.jsonl.summary.json` records the kept count and per-reason drop counts.

Emitted `vision_path` values are rewritten relative to the output JSONL's directory (so the loader's relative-path branch resolves them). URIs like `s3://...` pass through unchanged.

> This path emits only the dense `caption` string per window (not the structured `caption_json`). The loader trains on it via its dense-caption fallback (see [Format](#format)). To train on structured-JSON captions instead, use the captioning workflow above.
