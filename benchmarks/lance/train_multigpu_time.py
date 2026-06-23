# SPDX-License-Identifier: OpenMDW-1.1
"""Multi-GPU (DDP) per-epoch training-time comparison: base loader vs Lance loader.

Answers "does the dataloader speedup make TRAINING faster?" — which depends on whether
training is data-bound (GPU waits on the loader -> Lance helps) or compute-bound (loader
hidden behind forward/backward -> Lance frees CPU but wall-clock is unchanged). We measure
steady-state per-epoch wall-clock (epoch 0 = warmup, discounted) AND the data-wait fraction.

Launch (one loader per run):
  torchrun --nproc-per-node=4 benchmarks/lance/train_multigpu_time.py --loader base  --epochs 3 --n 2000
  torchrun --nproc-per-node=4 benchmarks/lance/train_multigpu_time.py --loader lance --epochs 3 --n 2000
"""
from __future__ import annotations

import argparse
import io
import os
import time

import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP

MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
_PROC = None
_SPECIAL = None


def _proc():
    global _PROC
    if _PROC is None:
        from transformers import AutoProcessor
        _PROC = AutoProcessor.from_pretrained(MODEL)
    return _PROC


def _decode(image):
    return Image.open(io.BytesIO(image["bytes"])).convert("RGB") if isinstance(image, dict) else image.convert("RGB")


def _messages(conv, img):
    msgs, ins = [], False
    for t in conv:
        role = "user" if t["from"] == "human" else "assistant"
        text = t["value"].replace("<image>", "").strip()
        if role == "user" and not ins and img is not None:
            c = [{"type": "image", "image": img}, {"type": "text", "text": text}]; ins = True
        else:
            c = text
        msgs.append({"role": role, "content": c})
    return msgs


class Collate:
    """Runs in DataLoader workers (CPU): raw record -> model inputs (bs=1)."""
    def __call__(self, recs):
        global _SPECIAL
        p = _proc()
        rec = recs[0]
        enc = p.apply_chat_template(_messages(rec["conversations"], _decode(rec["image"])),
                                    tokenize=True, add_generation_prompt=False,
                                    return_dict=True, return_tensors="pt")
        ids = enc["input_ids"]
        if _SPECIAL is None:
            s = set(p.tokenizer.all_special_ids)
            im = p.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            if im is not None and im >= 0:
                s.add(im)
            _SPECIAL = torch.tensor(sorted(s))
        labels = ids.clone(); labels[torch.isin(ids, _SPECIAL)] = -100
        enc["labels"] = labels
        return enc


class BaseRecs(torch.utils.data.Dataset):
    def __init__(self, subset, n):
        from datasets import load_dataset
        self.ds = load_dataset("lmms-lab/LLaVA-OneVision-Data", name=subset, split=f"train[:{n}]")
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        r = self.ds[int(i)]; return {"image": r["image"], "conversations": r["conversations"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loader", choices=["base", "lance"], required=True)
    ap.add_argument("--subset", default="figureqa(cauldron,llava_format)")
    ap.add_argument("--lance-uri", default="/home/ubuntu/work/data/lance/llava_figureqa")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group("nccl"); torch.cuda.set_device(local)
    dev = torch.device("cuda", local)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText
    from cosmos_framework.data.lance.vlm_dataset import LanceVLMDataset

    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    model = DDP(model, device_ids=[local], find_unused_parameters=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    ds = BaseRecs(args.subset, args.n) if args.loader == "base" else LanceVLMDataset(args.lance_uri, "llava")
    if args.loader == "lance":
        ds = torch.utils.data.Subset(ds, list(range(args.n)))
    sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=0)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, sampler=sampler, num_workers=args.workers,
                                         collate_fn=Collate(), persistent_workers=True, prefetch_factor=4,
                                         multiprocessing_context="spawn")

    for ep in range(args.epochs):
        sampler.set_epoch(ep)
        model.train()
        torch.cuda.synchronize(); t_ep = time.perf_counter(); t_data = 0.0; last = time.perf_counter()
        nloss = 0.0; nsteps = 0
        for enc in loader:
            t_data += time.perf_counter() - last
            enc = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in enc.items()}
            out = model(**enc); out.loss.backward(); opt.step(); opt.zero_grad()
            nloss += float(out.loss.detach()); nsteps += 1
            last = time.perf_counter()
        torch.cuda.synchronize()
        ep_t = time.perf_counter() - t_ep
        # reduce timing/loss across ranks
        stats = torch.tensor([ep_t, t_data, nloss, nsteps], device=dev)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        ep_t_avg = stats[0].item() / world; data_avg = stats[1].item() / world
        loss_avg = stats[2].item() / stats[3].item()
        if rank == 0:
            tag = "WARMUP" if ep == 0 else "STEADY"
            print(f"[{args.loader}] epoch {ep} {tag}: {ep_t_avg:6.1f}s/epoch | "
                  f"data-wait {100*data_avg/ep_t_avg:4.1f}% | {int(stats[3].item())} samples "
                  f"({stats[3].item()/ep_t_avg:5.1f} samp/s global) | loss {loss_avg:.3f}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
