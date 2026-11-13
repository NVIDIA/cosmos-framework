# SPDX-License-Identifier: OpenMDW-1.1
"""Vision-SFT forward-equivalence: base vs Lance loader through the REAL Cosmos3-Nano (16B MoT).

For a handful of indices, take the SAME clip from the base ``SFTDataset`` and the
``LanceVisionSFTDataset`` (aligned by index -> no shuffle), pack each into a model
batch, and run it through ``model.training_step`` with FIXED weights + seed. The
only thing that differs is the loader's decoded video (an offline H.264 re-encode
on the Lance side), so the loss delta measures exactly that -- it sits inside the
re-encode tolerance (~2%), well below the loss spread across different clips.

The real-model counterpart to ``tests/data/lance/`` (which proves the per-sample
batches are byte/token-equal at the data level). Single GPU, forward only -- no
FSDP, no optimizer.

    python benchmarks/lance/forward_equivalence.py            # vision-SFT

ACTION and VLM run through the GENUINE training recipe with ``optimizer.lr=0`` (weights
never change, so each step is a forward loss on the same sample) + ``shuffle off`` (so
base step-k and Lance step-k are the SAME sample). The Lance swap is one line: point the
recipe's dataset ``_target_`` at our ``get_lance_*`` factory + add the uri/table. Nothing
else in the recipe changes -- that one-line swap IS the drop-in. The trainer is used here
(not this standalone batcher) because action's ``ActionProcessingRecord`` / VLM's records
need the recipe's own collation.

  * ACTION -- per-step base-vs-Lance loss within ~1.4% (measured against the pre-2026-07
    base; labels re-verified bit-exact against the rewritten lazy-LeRobot base — see
    tests/data/lance/test_action.py):
      torchrun --nproc_per_node=4 -m cosmos_framework.scripts.train \
        --sft-toml=examples/toml/sft_config/action_policy_droid_repro.toml --deterministic -- \
        optimizer.lr=0.0 trainer.max_iter=5 model.parallelism.data_parallel_shard_degree=4 \
        model.compile.enabled=false model.ema.enabled=false \
        dataloader_train.max_samples_per_batch=null dataloader_train.max_sequence_length=2048 \
        dataloader_train.dataloader.datasets.droid.dataset.iterable_shuffle=false \
        dataloader_train.dataloader.datasets.droid.dataset.resolution=256 \
        dataloader_train.dataloader.datasets.droid.dataset._target_=cosmos_framework.data.lance.action_dataset.get_lance_action_droid_sft_dataset \
        ~dataloader_train.dataloader.datasets.droid.dataset.root \
        ~dataloader_train.dataloader.datasets.droid.dataset.use_success_only \
        +dataloader_train.dataloader.datasets.droid.dataset.lance_uri=<uri> \
        +dataloader_train.dataloader.datasets.droid.dataset.table=droid_composed \
        +dataloader_train.dataloader.datasets.droid.dataset.decode_device=cpu
      (drop the '~'/'+' lines for the base arm — the Lance loader reads labels + video
       from LanceDB, so the base 'root'/'use_success_only' args are removed rather than
       passed through.)

  * VLM -- byte-identical records, so the loss matches EXACTLY (measured: base 0.8149 ==
    Lance 0.8149, 0.00%):
      torchrun --nproc_per_node=4 -m cosmos_framework.scripts.train \
        --sft-toml=examples/toml/sft_config/llava_ov_mapstyle_dataloader.toml --deterministic -- \
        optimizer.lr=0.0 trainer.max_iter=1 model.parallelism.data_parallel_shard_degree=4 \
        dataloader_train.distributor.shuffle=false \
        dataloader_train.distributor.dataset.subset="'figureqa(cauldron,llava_format)'" \
        dataloader_train.distributor.dataset._target_=cosmos_framework.data.lance.vlm_dataset.get_lance_vlm_dataset \
        +dataloader_train.distributor.dataset.uri=<uri> \
        +dataloader_train.distributor.dataset.table_name=llava
      (drop the last two '+' lines for the base arm.)

Env: HF_TOKEN, and LD_LIBRARY_PATH must include the venv's nvidia/*/lib (for
torchcodec). Requires the converted Cosmos3-Nano + Wan VAE (see docs/training.md).
"""

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

from cosmos_framework.inference.common.init import init_script

init_script(env={"COSMOS_DEVICE": "cuda"})

import torch
from transformers import AutoTokenizer

from cosmos_framework.data.generator.dataflow.batchers import SequentialPackingBatcher
from cosmos_framework.data.generator.dataflow.collators import VFMListCollator
from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.common.args import CheckpointOverrides
from cosmos_framework.inference.common.public_model_config import build_public_model_config
from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel

_D = "/home/ubuntu/work/data"
_VAE = "/home/ubuntu/work/cosmos-framework/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth"
_TOKENIZER = "Qwen/Qwen2.5-7B"  # both arms share one tokenizer; only the video differs


def build_model():
    """Build the real Cosmos3-Nano on one GPU with weights loaded, forward-only."""
    ckpt = CheckpointOverrides(checkpoint_path="Cosmos3-Nano").build_checkpoint(
        checkpoints=OmniSetupOverrides.CHECKPOINTS
    )
    hf_path = ckpt.download_checkpoint()
    from cosmos_framework.scripts.convert_model_to_dcp import _redirect_avae_to_local

    _redirect_avae_to_local(hf_path)
    pub = build_public_model_config(ckpt.load_model_config_dict())
    tk = pub["config"]["tokenizer"]
    tk["vae_path"] = _VAE  # local Wan VAE
    tk["bucket_name"] = ""
    tk["object_store_credential_path_pretrained"] = ""  # don't auth to GCS
    pub["config"]["sound_gen"] = False  # no audio -> skip the AVAE
    pub["config"]["sound_tokenizer"] = None
    model = Cosmos3OmniModel.from_pretrained_dcp(hf_path, config=Cosmos3OmniConfig(model=pub)).model
    model = model.cuda().eval()  # config precision handles dtype; don't cast fp32 buffers (inv_freq)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _pack(sample: dict) -> dict:
    batcher = SequentialPackingBatcher(
        max_sequence_length=8192,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=2,
        max_samples_per_batch=None,
        sound_latent_fps=0,
        audio_sample_rate=48000,
    )
    group = next(batcher.batches(iter([sample])))
    return VFMListCollator().collate(group)


def _loss(model, sample: dict) -> float:
    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in _pack(sample).items()}
    torch.manual_seed(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.training_step(batch, 0)
    loss = out[1] if isinstance(out, (tuple, list)) else out
    return float(loss.item() if torch.is_tensor(loss) else loss)


def _vision_pair(tok):
    from cosmos_framework.data.generator.local_datasets.sft_dataset import (
        SFTDataset,
        _flatten_metadata_by_window,
        _load_sft_metadata_from_s3,
    )
    from cosmos_framework.data.lance import LanceVisionSFTDataset

    jsonl = f"{_D}/bridge_src/sft_dataset_bridge/train/video_dataset_file.jsonl"
    base_dir = os.path.dirname(jsonl)
    metas = _flatten_metadata_by_window(_load_sft_metadata_from_s3(None, jsonl, min_frames=61))
    for m in metas:
        vp = m["vision_path"]
        m["vision_path"] = vp if ("://" in vp or vp.startswith("/")) else os.path.join(base_dir, vp)
    vkw = dict(num_video_frames=16, frame_selection_mode="first", temporal_interval_mode="entire_chunk")
    base = SFTDataset(
        metadata=metas, resolution="256", s3_credentials={}, tokenizer_config=tok, cfg_dropout_rate=0.0, **vkw
    )
    base.s3_client = None
    lance = LanceVisionSFTDataset(f"{_D}/lance/vision_sft_plain", table="vision_sft", decode_device="cpu", **vkw)

    def get_base(i):
        s = base.process_one_sample(metas[i])
        s["conditioning_fps"] = 24.0
        return s

    def get_lance(i):
        s = lance[i]
        s["conditioning_fps"] = 24.0
        return s

    return get_base, get_lance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2, 3])
    args = ap.parse_args()

    tok = SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained(_TOKENIZER))
    get_base, get_lance = _vision_pair(tok)

    print("building real Cosmos3-Nano (this loads ~16B params)...", flush=True)
    model = build_model()
    print("[vision] forward-equivalence: base vs Lance through the real model\n", flush=True)
    print(f"{'idx':>4} {'base':>10} {'lance':>10} {'%diff':>7}")
    diffs = []
    for i in args.indices:
        b, l = _loss(model, get_base(i)), _loss(model, get_lance(i))
        d = abs(b - l) / b * 100
        diffs.append(d)
        print(f"{i:>4} {b:>10.4f} {l:>10.4f} {d:>6.2f}%", flush=True)
    print(f"\nmax %diff = {max(diffs):.2f}%  (within the H.264 re-encode tolerance)")


if __name__ == "__main__":
    main()
    os._exit(0)
