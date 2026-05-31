# Cosmos3 vLLM Plugin

Start the vLLM server:

```shell
VLLM_USE_DEEP_GEMM=0 uvx --with-editable ./packages/vllm-cosmos3 --torch-backend=cu130 vllm@0.21.0 serve nvidia/Cosmos3-Nano \
  --hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}' \
  --tensor-parallel-size 1 \
  --mm-encoder-tp-mode data \
  --async-scheduling \
  --allowed-local-media-path "$(pwd)" \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --port 8000
```

**Note:* For CUDA 12.8, use `--torch-backend=cu128 vllm@0.19.1`.

Wait for the server to start (takes ~5 minutes). You will see `Application startup complete.` in the log.

In a separate terminal, submit a request:

```shell
curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "messages": [
        {
          "role": "user",
          "content": [
            {"type": "image_url", "image_url": {"url": "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/2b17a2413bd86b2cf9b03823637108851e4ddf2d/inputs/vision/robot_153.jpg"}},
            {"type": "text", "text": "Caption the image in detail."}
          ]
        }
      ],
      "max_tokens": 4096,
      "seed": 0
    }' | jq -r '.choices[0].message.content'
```

For more details, see:

- [Cosmos-Reason2 repository](https://github.com/nvidia-cosmos/cosmos-reason2)
- [Qwen3-VL repository](https://github.com/QwenLM/Qwen3-VL#online-serving)
- [Qwen3-VL vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html)
