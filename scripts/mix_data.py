"""Shuffle several corpora into one file, document by document.

Concatenating them instead puts each source in a contiguous run, and since
encode_data splits train/val contiguously, the val set ends up being one source
only. That is how a fine-tune run reported val 0.889 against train 2.110: the
val slice was pure arithmetic, which is far easier than dialogue.
"""
import argparse
import os
import random

from nika.tokenizer import EOT

SEPARATOR = "\n" + EOT + "\n"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", default="data/finetune.txt")
    args = p.parse_args()

    docs = []
    for path in args.inputs:
        # a document is one conversation or one Q&A pair; shuffling smaller
        # pieces than that would interleave unrelated turns into nonsense
        found = [d.strip() for d in open(path, encoding="utf-8").read().split(EOT) if d.strip()]
        print(f"{len(found):>7} documents from {path}")
        docs += found

    random.seed(0)
    random.shuffle(docs)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for doc in docs:
            f.write(doc + SEPARATOR)

    print(f"{len(docs)} documents, {os.path.getsize(args.out) / 1e6:.1f} MB -> {args.out}")
