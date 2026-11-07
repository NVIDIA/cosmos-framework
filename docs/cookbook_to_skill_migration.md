# Cookbook notebooks → agent skills: migration record

Where each retired file of `.agents/skills/cookbooks/cosmos3/generator/audiovisual/` went. The sources are the files'
last version on `main` before their retirement (`7738e9e`, after #329). Notebook cells were migrated 1:1 (Jupyter
plumbing and preview/viewer cells aside); everything the skill adds is marked "Skill note (not in the notebook)" or
declared inside the block that carries it.

| Retired source (`cookbooks/cosmos3/generator/audiovisual/`)                                                                                     | Destination (`.agents/skills/cosmos3-generate-media/`)                                                                                                                                                                                                                                                                                 | Notes                                                                                                                 |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `README.md` — introduction, licensing, asset catalogue, per-backend quickstarts                                                                 | `references/payloads.md` § 1–3 (introduction, licensing and access, asset catalogue); the quickstarts in `references/diffusers.md`, `cosmos-framework.md`, `trt-llm.md` and `distilled-trt-llm.md` § "Quickstart (cookbook README)" and `vllm-omni.md` § "Run with vLLM-Omni"; `SKILL.md` Variants 1–6 quote each backend's smoke path | The README's FP8 pointer (#329) is in `payloads.md` § 1                                                               |
| `run_with_diffusers.ipynb`                                                                                                                      | `references/diffusers.md`                                                                                                                                                                                                                                                                                                              | `SKILL.md` Variant 1                                                                                                  |
| `run_with_cosmos_framework.ipynb`                                                                                                               | `references/cosmos-framework.md`                                                                                                                                                                                                                                                                                                       | `SKILL.md` Variant 2                                                                                                  |
| `run_with_vllm_omni.ipynb`                                                                                                                      | `references/vllm-omni.md`                                                                                                                                                                                                                                                                                                              | `SKILL.md` Variant 3; follows the notebook's final version (#329: launch mounts, `HF_TOKEN`, "Quantized Checkpoints") |
| `run_with_sglang.ipynb`                                                                                                                         | `references/sglang.md`                                                                                                                                                                                                                                                                                                                 | `SKILL.md` Variant 4                                                                                                  |
| `run_with_trt_llm.ipynb`                                                                                                                        | `references/trt-llm.md`                                                                                                                                                                                                                                                                                                                | `SKILL.md` Variant 5                                                                                                  |
| `run_distilled_with_trt_llm.ipynb`                                                                                                              | `references/distilled-trt-llm.md`                                                                                                                                                                                                                                                                                                      | `SKILL.md` Variant 6                                                                                                  |
| `run_with_nim.ipynb` (retired earlier by #296)                                                                                                  | no copy: `SKILL.md` Variant 7 delegates to the NIM cookbook `.agents/skills/cookbooks/cosmos3/nim/` and its `cosmos3-nim-user` skill                                                                                                                                                                                                   | —                                                                                                                     |
| Content the six notebooks share — asset sets, payload helpers, sampling defaults, resolution table, use-case matrix, per-backend request fields | `references/payloads.md` § 4–11                                                                                                                                                                                                                                                                                                        | one copy instead of six                                                                                               |
| `assets/` — prompts, negative prompts, reference images                                                                                         | `assets/` (vendored copy)                                                                                                                                                                                                                                                                                                              | The cookbook keeps `assets/`, `distill/` and `finetune/`                                                              |

Links into the retired files: the cookbook READMEs and the docs navigation point at `SKILL.md`; the NVIDIA/cosmos
short links were re-pointed to `SKILL.md` and its vLLM-Omni section (NVIDIA/cosmos#390).

## Quantization (COOKBOOK row "Quantized (FP8)") → `cosmos3-inference-quantization`

Where each retired file of `.agents/skills/cookbooks/cosmos3/quantization/` went. The sources are the files' version on
`main` at `bbd4a67` (2026-11-07). The three notebooks were converted 1:1 (every markdown and code cell in notebook order;
`%%bash` cells as `bash` fences with the magic line removed; the committed notebooks carry no outputs) and the README was
copied whole — including its "Inference", "Mixed activation precision" and "What gets produced" sections, because the
inference skills link to it. The skill is part 2 of the cookbook (how to quantize); part 1 (inference with released FP8
checkpoints) stays in the inference skills' own FP8 sections. Everything the skill adds is marked "Skill note (not in
the source):" or is a summary in `SKILL.md` that quotes the source.

| Retired source (`cookbooks/cosmos3/quantization/`) | Destination (`.agents/skills/cosmos3-inference-quantization/`) | Notes                                                                                                                                                                                                             |
| -------------------------------------------------- | -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `README.md`                                        | `references/quantization-readme.md`                            | whole README; `SKILL.md` quotes its Setup (once) block, the three Run commands (as text — they executed the retired notebooks), the Demo vs. full runs table, the What gets produced tree and the Inference notes |
| `quantize_cosmos3.ipynb`                           | `references/quantize-cosmos3.md`                               | `SKILL.md` Variant 1 (Edge / Nano / Super); eval case `framework-edge-quantize-fp8`                                                                                                                               |
| `quantize_super_text2image.ipynb`                  | `references/quantize-super-text2image.md`                      | `SKILL.md` Variant 2; eval case `framework-super-t2i-quantize-fp8`                                                                                                                                                |
| `quantize_super_image2video.ipynb`                 | `references/quantize-super-image2video.md`                     | `SKILL.md` Variant 3; eval case `framework-super-i2v-quantize-fp8`                                                                                                                                                |
| (assets)                                           | —                                                              | the cookbook shipped no assets: nothing vendored, nothing left behind; the calibration prompts and the I2V fallback images are datasets the recipe fetches at run time                                            |

The notebook-derived references keep the notebooks' text; the converter's provenance note sits after each notebook's
first cell (title and introduction). Link targets re-pointed inside `references/` (link text unchanged):

- `quantization-readme.md`: `./quantize_cosmos3.ipynb` → `quantize-cosmos3.md` (2 links), `./quantize_super_text2image.ipynb`
  → `quantize-super-text2image.md` (2), `./quantize_super_image2video.ipynb` → `quantize-super-image2video.md` (2);
  `../README.md#cosmos-framework`, `../README.md`, `../README.md#vllm`, `../README.md#vllm-omni` →
  `../../cookbooks/cosmos3/README.md` with the same anchors (4);
  `../../../cosmos3-generate-media/references/vllm-omni.md#quantized-checkpoints` →
  `../../cosmos3-generate-media/references/vllm-omni.md#quantized-checkpoints` (1); the Inference table's
  `../generator/action/run_fd_with_vllm_omni.ipynb`, `../generator/action/run_id_with_vllm_omni.ipynb`,
  `../generator/transfer/run_video_transfer_with_vllm_omni.ipynb`, `../reasoner/run_with_vllm.ipynb` →
  `../../cookbooks/cosmos3/generator/action/…`, `../../cookbooks/cosmos3/generator/transfer/…`,
  `../../cookbooks/cosmos3/reasoner/…` (4; the notebooks' current location, so the links resolve at this change's head — the
  Action, Transfer and Visual Reasoning rows retire those notebooks, after which the four links are re-pointed at those rows'
  `references/` in the follow-up round).
- `quantize-cosmos3.md`, `quantize-super-text2image.md`, `quantize-super-image2video.md` (§ 3 Install Dependencies, each):
  `../README.md#cosmos-framework` → `../../cookbooks/cosmos3/README.md#cosmos-framework`, `../../../../../docs/setup.md` →
  `../../../../docs/setup.md`.

Inbound links re-pointed to `.agents/skills/cosmos3-inference-quantization/SKILL.md` (link text unchanged):
`.agents/skills/cookbooks/cosmos3/README.md` (`quantization/README.md`), `docs/inference.md`
(`../.agents/skills/cookbooks/cosmos3/quantization/README.md`), `.agents/skills/cosmos3-generate-media/references/payloads.md`,
`references/sglang.md` and `references/vllm-omni.md` (`../../cookbooks/cosmos3/quantization/README.md`). The cookbook pages
`.agents/skills/cookbooks/cosmos3/generator/action/README.md`, `generator/transfer/README.md` and `reasoner/README.md` and seven
notebooks also link the retired README; all of them are retired by the Action, Transfer and Visual Reasoning rows of this
migration and are left unchanged here — `generator/action/run_fd_with_sglang.ipynb`,
`run_fd_with_vllm_omni.ipynb`, `run_id_with_sglang.ipynb`, `run_id_with_vllm_omni.ipynb`,
`generator/transfer/run_video_transfer_with_sglang.ipynb`, `run_video_transfer_with_vllm_omni.ipynb`
(`../../quantization/README.md`) and `reasoner/run_with_vllm.ipynb` (`../quantization/README.md`). Their migrated copies in the
sibling rows link the README at its current location too and are re-pointed at `.agents/skills/cosmos3-inference-quantization/SKILL.md`
in the follow-up round; until then those links are dead on GitHub rendering once this change is merged (a merge-order dependency
between the rows, not a missing file). No navigation entry: the cookbook had none; the skill is
reached through the agent-skills table and the links above.
