# Optional local captioning server

Start a video-capable vLLM endpoint in a separate `uvx` environment. It needs a supported NVIDIA GPU, sufficient memory and [Qwen model weights from Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-FP8). Run the client in the prepared Framework environment. Reuse a compatible endpoint if one is already running.

Use the [HF example2 download](../../shared/hf-sample.md) in a separate terminal.
Limit the server's local-file access to the media directory you want to caption:

```bash
caption_media_root="${SIM_TRANSFER_DATASET:?Download HF example2 first}"
uvx \
  --torch-backend=cu129 \
  --with nvidia-cuda-runtime-cu12 vllm@0.19.0 serve Qwen/Qwen3-VL-8B-Instruct-FP8 \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --allowed-local-media-path "$caption_media_root"
```

Use `--torch-backend=cu129` to select CUDA-enabled PyTorch for the pinned vLLM
release's CUDA 12.9 binaries; see [vLLM's installation guide](https://docs.vllm.ai/en/v0.19.0/getting_started/installation/gpu/).

Keep the terminal running. After `Application startup complete.`, check the served model:

```bash
curl --fail http://127.0.0.1:8000/v1/models
```

Run the client with `http://localhost:8000/v1` and the served model name. Keep the client and server on the same HF sample download, within the allowed media root.

After the model-list check, run the [caption command](caption-guide.md#caption-a-new-clip).
