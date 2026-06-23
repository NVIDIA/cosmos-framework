# SPDX-License-Identifier: OpenMDW-1.1
"""Train-equivalence test: LoRA-SFT the same VLM with the BASE loader vs the
LanceDB loader and compare loss curves.

Same model init, same LoRA seed, same sample order — the ONLY difference is which
loader produces each sample (HF dataset record vs LanceVLMDataset record). The
LanceDB VLM loader stores token-exact + image-exact (lossless PNG) data, so if it
is a true drop-in the two loss curves should overlay near-exactly.

bs=1 (matches cosmos VLM packing; avoids padding). One frozen base model + a small
LoRA adapter trained on the assistant tokens (next-token CE).
"""
from __future__ import annotations

import argparse
import io
import json

import numpy as np
import torch
from PIL import Image


def _decode(image):
    if isinstance(image, dict):
        return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
    return image.convert("RGB")


def _messages(conversations, image):
    msgs, ins = [], False
    for t in conversations:
        role = "user" if t["from"] == "human" else "assistant"
        text = t["value"].replace("<image>", "").strip()
        if role == "user" and not ins and image is not None:
            content = [{"type": "image", "image": image}, {"type": "text", "text": text}]
            ins = True
        else:
            content = text
        msgs.append({"role": role, "content": content})
    return msgs


_SPECIAL = None


def to_inputs(rec, processor, device):
    global _SPECIAL
    msgs = _messages(rec["conversations"], _decode(rec["image"]))
    enc = processor.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    # Mask special + image-placeholder tokens (deterministic, identical for both
    # loaders; the exact scheme is irrelevant — only that base==lance inputs).
    if _SPECIAL is None:
        ids = set(processor.tokenizer.all_special_ids)
        img = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if img is not None and img >= 0:
            ids.add(img)
        _SPECIAL = torch.tensor(sorted(ids))
    labels = input_ids.clone()
    labels[torch.isin(input_ids, _SPECIAL)] = -100
    enc = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc.items()}
    enc["labels"] = labels.to(device)
    return enc


class BaseRecs(torch.utils.data.Dataset):
    """HF figureqa records (raw)."""
    def __init__(self, subset):
        from datasets import load_dataset
        self.ds = load_dataset("lmms-lab/LLaVA-OneVision-Data", name=subset, split="train")
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        r = self.ds[int(i)]
        return {"image": r["image"], "conversations": r["conversations"]}


def run(loader_name, recs, order, processor, model, init_state, lr, eval_ids, device):
    # reset LoRA to the shared init + reseed
    torch.manual_seed(0); np.random.seed(0)
    model.load_state_dict(init_state, strict=False)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    losses = []
    for step, i in enumerate(order):
        enc = to_inputs(recs[i], processor, device)
        out = model(**enc)
        out.loss.backward()
        opt.step(); opt.zero_grad()
        losses.append(float(out.loss.detach()))
    # eval loss on held-out (no grad)
    model.eval(); ev = []
    with torch.no_grad():
        for i in eval_ids:
            ev.append(float(model(**to_inputs(recs[i], processor, device)).loss))
    return losses, float(np.mean(ev))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--lance-uri", default="/home/ubuntu/work/data/lance/llava_figureqa")
    ap.add_argument("--subset", default="figureqa(cauldron,llava_format)")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--eval-n", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from cosmos_framework.data.lance.vlm_dataset import LanceVLMDataset

    device = "cuda"
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    # shared init snapshot (LoRA weights are the only trainable bits)
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    base = BaseRecs(args.subset)
    lance = LanceVLMDataset(args.lance_uri, "llava")
    assert len(base) == len(lance), (len(base), len(lance))
    rng = np.random.RandomState(0)
    order = rng.choice(len(base), size=args.steps, replace=False).tolist()
    eval_ids = rng.choice(len(base), size=args.eval_n, replace=False).tolist()

    print(f"\nmodel={args.model} steps={args.steps} lr={args.lr}")
    base_losses, base_eval = run("base", base, order, processor, model, init_state, args.lr, eval_ids, device)
    base2_losses, base2_eval = run("base2", base, order, processor, model, init_state, args.lr, eval_ids, device)
    lance_losses, lance_eval = run("lance", lance, order, processor, model, init_state, args.lr, eval_ids, device)

    bl, b2, ll = np.array(base_losses), np.array(base2_losses), np.array(lance_losses)
    print(f"\n{'step':>5}{'base loss':>12}{'lance loss':>12}{'base-lance':>12}{'base-base2':>12}")
    for s in list(range(0, args.steps, max(1, args.steps // 10))) + [args.steps - 1]:
        print(f"{s:>5}{bl[s]:>12.4f}{ll[s]:>12.4f}{abs(bl[s]-ll[s]):>12.2e}{abs(bl[s]-b2[s]):>12.2e}")
    print(f"\nstep-0 |base-lance|        : {abs(bl[0]-ll[0]):.3e}  (0 => identical inputs)")
    print(f"mean |Δ| base-vs-lance     : {np.abs(bl-ll).mean():.3e}  (max {np.abs(bl-ll).max():.3e})")
    print(f"mean |Δ| base-vs-base2 ctrl: {np.abs(bl-b2).mean():.3e}  (max {np.abs(bl-b2).max():.3e})  <- nondeterminism floor")
    print(f"final train loss  base={bl[-5:].mean():.4f}  base2={b2[-5:].mean():.4f}  lance={ll[-5:].mean():.4f}")
    print(f"held-out eval loss base={base_eval:.4f}  base2={base2_eval:.4f}  lance={lance_eval:.4f}")


if __name__ == "__main__":
    main()
