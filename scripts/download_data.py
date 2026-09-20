"""Download a slice of FineWeb-Edu into one text file.

Documents are separated by <|EOT|> (end of text) so the tokenizer can later
turn the boundary into a single special token.
"""
import argparse
import os

from datasets import load_dataset

OUT_PATH = "data/tiny.txt"
TARGET_BYTES = 100_000_000  # ~100 MB
SEPARATOR = "\n<|EOT|>\n"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=OUT_PATH)
    p.add_argument("--mb", type=float, default=TARGET_BYTES / 1e6, help="how much text to keep")
    args = p.parse_args()
    out_path, target_bytes = args.out, int(args.mb * 1e6)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

    total_bytes = 0
    n_docs = 0
    
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for doc in ds:
            chunk = doc["text"] + SEPARATOR
            f.write(chunk)
            total_bytes += len(chunk.encode("utf-8"))
            n_docs += 1
            if n_docs % 5000 == 0:
                print(f"{n_docs} docs, {total_bytes / 1e6:.1f} MB")
            if total_bytes >= target_bytes:
                break

    print(f"done: {n_docs} docs, {total_bytes / 1e6:.1f} MB -> {out_path}")


if __name__ == "__main__":
    main()