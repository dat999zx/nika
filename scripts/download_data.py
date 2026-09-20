"""Download a slice of FineWeb-Edu into one text file.

Documents are separated by <|EOT|> (end of text) so the tokenizer can later
turn the boundary into a single special token.
"""
import os

from datasets import load_dataset

OUT_PATH = "data/tiny.txt"
TARGET_BYTES = 100_000_000  # ~100 MB
SEPARATOR = "\n<|EOT|>\n"

def main():
    os.makedirs("data", exist_ok=True)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

    total_bytes = 0
    n_docs = 0
    
    with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as f:
        for doc in ds:
            chunk = doc["text"] + SEPARATOR
            f.write(chunk)
            total_bytes += len(chunk.encode("utf-8"))
            n_docs += 1
            if n_docs % 5000 == 0:
                print(f"{n_docs} docs, {total_bytes / 1e6:.1f} MB")
            if total_bytes >= TARGET_BYTES:
                break

    print(f"done: {n_docs} docs, {total_bytes / 1e6:.1f} MB -> {OUT_PATH}")


if __name__ == "__main__":
    main()