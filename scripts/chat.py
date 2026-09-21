import argparse
import re

import torch

from config import DataConfig, TrainConfig
from nika.model import Nika
from nika.tokenizer import BPETokenizer


def space_digits(text):
    """'what is 23 + 45?' -> 'what is 2 3 + 4 5?', the format the model was trained on"""
    return re.sub(r"\d+", lambda m: " ".join(m.group()), text)


def unspace_digits(text):
    """'8 6' -> '68': answers are generated least significant digit first"""
    stripped = text.replace(" ", "")
    return stripped[::-1] if stripped.isdigit() else text


if __name__ == "__main__":
    cfg, dcfg = TrainConfig(), DataConfig()

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoint/nika_chat.pt")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--max-reply", type=int, default=80)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=cfg.device, weights_only=False)
    model = Nika(ckpt["cfg"]).to(cfg.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    block_size = ckpt["cfg"].block_size

    tok = BPETokenizer.load(dcfg.tokenizer_path)
    history = ""
    print(f"chatting with {args.checkpoint} (context {block_size} tokens). 'quit' to stop.\n")

    while True:
        msg = input("A: ").strip()
        if msg in ("quit", "exit"):
            break

        history += f"A: {space_digits(msg)}\nB:"
        ids = tok.encode(history)[-block_size:] # oldest turns drop out of the window
        idx = torch.tensor([ids], dtype=torch.long, device=cfg.device)

        out = model.generate(idx, args.max_reply, temperature=args.temperature,
                             top_k=args.top_k, stop_token=tok.eot_id)

        reply = tok.decode(out[0, len(ids):].tolist()) # only the new tokens
        reply = reply.split("\n")[0].replace(tok.decode([tok.eot_id]), "").strip()

        history += f" {reply}\n" # the history keeps the model's own format
        print(f"B: {unspace_digits(reply)}    [{len(ids)}/{block_size} tokens of context]")
