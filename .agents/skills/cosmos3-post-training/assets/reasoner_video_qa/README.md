# RoboVQA video-QA sample

Four RoboVQA clips with their question/answer rows, in the conversation
manifest layout the reasoner training planner reads: three in `train/` and one
in `validation/`.

- Source: Hugging Face dataset `Tianli/robovqa`, revision `eb58194`
  (the same clips that `.agents/skills/cosmos3-qwen3-vl-robovqa-sft/assets/robovqa/`
  bundles).
- License: Apache-2.0, see [`LICENSE.txt`](LICENSE.txt).
- Changes: clips copied unmodified; the rows were rewritten from the source JSON
  into `manifest.json` (`id`, `video`, `conversations`).
- The clips are Git LFS objects. Run `git lfs pull` if a clone holds pointer files.
