"""Score a checkpoint on the held-out arithmetic questions.

Greedy decoding (top_k=1) so the number is a capability measurement, not a
sample. Reports the two holdouts separately:
  random   - unseen COMBINATIONS of numbers it has seen
  sum=137  - answers it has never once emitted during training
A model that memorised its training pairs scores near zero on the second.
"""
import argparse
import re

import torch

from config import DataConfig, TrainConfig
from nika.model import Nika
from nika.tokenizer import BPETokenizer, EOT

HELDOUT_SUM = 137


def load_heldout(path):
    """[(question, answer), ...] from the A:/B: blocks"""
    pairs = []
    for block in open(path, encoding="utf-8").read().split(EOT):
        block = block.strip()
        if not block:
            continue
        q, a = block.split("\nB:")
        pairs.append((q[len("A:"):].strip(), a.strip()))
    return pairs


def operands(question):
    """'what is 4 9 and 9 7?' -> (49, 97), digits are space separated"""
    return [int(n.replace(" ", "")) for n in re.findall(r"\d(?: \d)*", question)]


if __name__ == "__main__":
    cfg, dcfg = TrainConfig(), DataConfig()

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoint/nika_chat.pt")
    p.add_argument("--heldout", default="data/tasks_heldout.txt")
    p.add_argument("--limit", type=int, default=400, help="questions to score")
    p.add_argument("--show", type=int, default=8, help="wrong answers to print")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=cfg.device, weights_only=False)
    model = Nika(ckpt["cfg"]).to(cfg.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = BPETokenizer.load(dcfg.tokenizer_path)

    pairs = load_heldout(args.heldout)[:args.limit]
    stats = {"random": [0, 0], "sum137": [0, 0]} # slice -> [correct, total]
    wrong = []

    for question, answer in pairs:
        ids = tok.encode(f"A: {question}\nB:")
        idx = torch.tensor([ids], dtype=torch.long, device=cfg.device)
        out = model.generate(idx, 12, temperature=1.0, top_k=1, stop_token=tok.eot_id)

        got = tok.decode(out[0, len(ids):].tolist()).split("\n")[0]
        got = got.replace(EOT, "").strip()

        a, b = operands(question)[:2]
        slice_name = "sum137" if a + b == HELDOUT_SUM else "random"
        # both sides are least-significant-digit-first, so they compare directly
        ok = got.replace(" ", "") == answer.replace(" ", "")
        stats[slice_name][0] += ok
        stats[slice_name][1] += 1
        if not ok and len(wrong) < args.show:
            said = got.replace(" ", "")[::-1] # un-reverse just for reading
            wrong.append(f"  {a} + {b} = {a + b}, model said {said!r}")

    print(f"\n{args.checkpoint} on {len(pairs)} held-out questions\n")
    for name, (correct, total) in stats.items():
        if total:
            print(f"  {name:8s} {correct}/{total} = {100 * correct / total:.1f}%")
    total_c = sum(c for c, _ in stats.values())
    print(f"  {'overall':8s} {total_c}/{len(pairs)} = {100 * total_c / len(pairs):.1f}%")

    if wrong:
        print("\nwrong answers:")
        print("\n".join(wrong))
