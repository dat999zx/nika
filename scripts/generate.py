import argparse

import torch

from config import DataConfig, TrainConfig
from nika.model import Nika
from nika.tokenizer import BPETokenizer

if __name__ == "__main__":
    cfg, dcfg = TrainConfig(), DataConfig()

    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="The ")
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--checkpoint", default=cfg.checkpoint_path)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=cfg.device, weights_only=False)
    model = Nika(ckpt["cfg"]).to(cfg.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tok = BPETokenizer.load(dcfg.tokenizer_path)
    ids = tok.encode(args.prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=cfg.device)

    for i in range(args.samples):
        out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
        print(f"\n--- sample {i + 1} (temp {args.temperature}, top_k {args.top_k}) ---")
        print(tok.decode(out[0].tolist()))