# SPDX-License-Identifier: OpenMDW-1.1
"""Real-training equivalence: LoRA-SFT the same VLM for several epochs with the
base loader vs the Lance loader and compare the *outputs*.

Same model init, LoRA seed, sample order, LR, epochs — only the loader differs.
Runs three trainings: base, base2 (base rerun = nondeterminism control), lance.
Then compares, base-vs-lance against base-vs-base2:
  (1) train loss curves, (2) held-out eval loss,
  (3) greedy generations on held-out prompts (exact-text match),
  (4) final LoRA weight max-abs diff.
If the Lance loader is a correct drop-in, base-vs-lance ≈ base-vs-base2 (nondeterminism).
"""
from __future__ import annotations

import argparse
import io
import numpy as np
import torch
from PIL import Image

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
_SPECIAL = None


def _decode(image):
    return Image.open(io.BytesIO(image["bytes"])).convert("RGB") if isinstance(image, dict) else image.convert("RGB")


def _messages(conv, img, drop_last_answer=False):
    msgs, ins = [], False
    turns = conv[:-1] if drop_last_answer else conv
    for t in turns:
        role = "user" if t["from"] == "human" else "assistant"
        text = t["value"].replace("<image>", "").strip()
        if role == "user" and not ins and img is not None:
            c = [{"type": "image", "image": img}, {"type": "text", "text": text}]; ins = True
        else:
            c = text
        msgs.append({"role": role, "content": c})
    return msgs


def to_inputs(rec, proc, dev):
    global _SPECIAL
    enc = proc.apply_chat_template(_messages(rec["conversations"], _decode(rec["image"])),
                                   tokenize=True, add_generation_prompt=False,
                                   return_dict=True, return_tensors="pt")
    ids = enc["input_ids"]
    if _SPECIAL is None:
        s = set(proc.tokenizer.all_special_ids)
        im = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if im is not None and im >= 0:
            s.add(im)
        _SPECIAL = torch.tensor(sorted(s))
    labels = ids.clone(); labels[torch.isin(ids, _SPECIAL)] = -100
    enc = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in enc.items()}
    enc["labels"] = labels.to(dev)
    return enc


def gen_text(rec, proc, model, dev, max_new=48):
    enc = proc.apply_chat_template(_messages(rec["conversations"], _decode(rec["image"]), drop_last_answer=True),
                                   tokenize=True, add_generation_prompt=True,
                                   return_dict=True, return_tensors="pt")
    enc = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    new = out[0, enc["input_ids"].shape[1]:]
    return proc.tokenizer.decode(new, skip_special_tokens=True).strip()


class BaseRecs(torch.utils.data.Dataset):
    def __init__(self, subset, n):
        from datasets import load_dataset
        self.ds = load_dataset("lmms-lab/LLaVA-OneVision-Data", name=subset, split=f"train[:{n}]")
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        r = self.ds[int(i)]; return {"image": r["image"], "conversations": r["conversations"]}


def train(model, init_state, recs, order, epochs, lr, proc, dev):
    torch.manual_seed(0); np.random.seed(0)
    model.load_state_dict(init_state, strict=False)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    losses = []
    for ep in range(epochs):
        for i in order:
            out = model(**to_inputs(recs[i], proc, dev)); out.loss.backward()
            opt.step(); opt.zero_grad(); losses.append(float(out.loss.detach()))
    return losses


def lora_vec(model):
    return torch.cat([p.detach().flatten().float().cpu() for n, p in model.named_parameters() if p.requires_grad])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="figureqa(cauldron,llava_format)")
    ap.add_argument("--lance-uri", default="/home/ubuntu/work/data/lance/llava_figureqa")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--eval-n", type=int, default=24)
    ap.add_argument("--gen-n", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from cosmos_framework.data.lance.vlm_dataset import LanceVLMDataset

    dev = "cuda"
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    base = BaseRecs(args.subset, args.n)
    lance = LanceVLMDataset(args.lance_uri, "llava")
    rng = np.random.RandomState(0)
    order = rng.permutation(args.n).tolist()
    eval_ids = rng.choice(args.n, size=args.eval_n, replace=False).tolist()
    gen_ids = rng.choice(args.n, size=args.gen_n, replace=False).tolist()

    print(f"\nmodel={MODEL} epochs={args.epochs} n={args.n} steps/run={args.epochs*args.n}")
    results = {}
    for tag, recs in [("base", base), ("base2", base), ("lance", lance)]:
        print(f"  training [{tag}] ...", flush=True)
        losses = train(model, init_state, recs, order, args.epochs, args.lr, proc, dev)
        model.eval()
        with torch.no_grad():
            ev = float(np.mean([float(model(**to_inputs(recs[i], proc, dev)).loss) for i in eval_ids]))
        gens = [gen_text(recs[i], proc, model, dev) for i in gen_ids]
        results[tag] = {"losses": np.array(losses), "eval": ev, "gens": gens, "vec": lora_vec(model)}

    b, b2, l = results["base"], results["base2"], results["lance"]
    print("\n=== TRAIN LOSS (every ~10%) ===")
    S = len(b["losses"])
    for s in list(range(0, S, max(1, S // 8))) + [S - 1]:
        print(f"  step {s:>4}: base {b['losses'][s]:.4f}  base2 {b2['losses'][s]:.4f}  lance {l['losses'][s]:.4f}")
    print(f"\nmean |Δ train loss|  base-vs-lance={np.abs(b['losses']-l['losses']).mean():.3e}  "
          f"base-vs-base2={np.abs(b['losses']-b2['losses']).mean():.3e} (noise floor)")
    print(f"held-out eval loss   base={b['eval']:.4f}  base2={b2['eval']:.4f}  lance={l['eval']:.4f}")
    print(f"final LoRA max|Δw|   base-vs-lance={(b['vec']-l['vec']).abs().max():.3e}  "
          f"base-vs-base2={(b['vec']-b2['vec']).abs().max():.3e}")
    bl = sum(x == y for x, y in zip(b["gens"], l["gens"]))
    bb = sum(x == y for x, y in zip(b["gens"], b2["gens"]))
    print(f"greedy generations identical  base-vs-lance={bl}/{args.gen_n}  base-vs-base2={bb}/{args.gen_n}")
    print("\n=== sample generations (held-out) ===")
    for k in range(min(3, args.gen_n)):
        print(f"  [{k}] base : {b['gens'][k][:90]}")
        print(f"      lance: {l['gens'][k][:90]}")


if __name__ == "__main__":
    main()
