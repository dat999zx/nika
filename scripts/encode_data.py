"""Tokenize the text file into train.bin / val.bin.

Streams: documents are read, encoded and written a window at a time, so peak memory
stays flat whether the input is 100 MB or 20 GB.
"""
import argparse
import os
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

from config import DataConfig
from nika.tokenizer import BPETokenizer, EOT

BATCH = 500 # documents per task
WINDOW = 8 # tasks held in memory at once (bounds RAM: ~WINDOW * BATCH docs)
           # also how often progress is printed and bytes hit the disk

# one tokenizer per worker process, loaded once instead of pickled per task
_tok = None

def _init(tokenizer_path):
    global _tok
    _tok = BPETokenizer.load(tokenizer_path)

# must be module level: workers re-import this file and look the function up by name
def _encode_batch(batch_text):
    ids = _tok.encode(batch_text)
    ids.append(_tok.eot_id) # separator between this batch and the next
    return np.array(ids, dtype=np.uint16)


def read_batches(path, batch_size):
    """yield strings of batch_size documents, without loading the whole file"""
    eot_line = EOT + "\n"
    docs, doc = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line == eot_line: # the separator sits on its own line
                docs.append("".join(doc))
                doc = []
                if len(docs) == batch_size:
                    yield EOT.join(docs)
                    docs = []
            else:
                doc.append(line)
    if doc:
        docs.append("".join(doc))
    if docs:
        yield EOT.join(docs)


def windows(iterable, size):
    """group an iterable into lists of at most `size` items"""
    group = []
    for item in iterable:
        group.append(item)
        if len(group) == size:
            yield group
            group = []
    if group:
        yield group


if __name__ == "__main__":
    # paths on the command line, so the same script encodes the web text and the dialogue
    defaults = DataConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=defaults.data_path)
    p.add_argument("--train-bin", default=defaults.train_bin_path)
    p.add_argument("--val-bin", default=defaults.val_bin_path)
    a = p.parse_args()
    cfg = replace(defaults, data_path=a.data, train_bin_path=a.train_bin, val_bin_path=a.val_bin)
    tok = BPETokenizer.load(cfg.tokenizer_path)
    os.makedirs(os.path.dirname(cfg.train_bin_path), exist_ok=True)

    all_path = cfg.train_bin_path + ".all" # temp file, deleted after the split
    n_tokens = n_eot = n_batches = 0

    # pass 1: encode and append straight to disk, a window of tasks at a time
    with open(all_path, "wb") as out, Pool(initializer=_init, initargs=(cfg.tokenizer_path,)) as pool:
        for group in windows(read_batches(cfg.data_path, BATCH), WINDOW):
            for arr in pool.map(_encode_batch, group): # map keeps the order
                arr.tofile(out)
                n_tokens += len(arr)
                n_eot += int((arr == tok.eot_id).sum())
            n_batches += len(group)
            print(f"{n_batches} batches, {n_tokens / 1e6:.1f} M tokens", flush=True)

    print(f"{n_tokens / 1e6:.1f} M tokens, {n_eot} EOT tokens")

    # pass 2: split. memmap so the 90/10 copy never loads the whole file
    data = np.memmap(all_path, dtype=np.uint16, mode="r")
    n_val = int(len(data) * cfg.val_fraction)
    for path, part in ((cfg.train_bin_path, data[:-n_val]), (cfg.val_bin_path, data[-n_val:])):
        with open(path, "wb") as f:
            for i in range(0, len(part), 10_000_000): # 20 MB at a time
                part[i:i + 10_000_000].tofile(f)
        print(f"{len(part) / 1e6:.1f} M tokens -> {path}")

    data._mmap.close() # Windows will not delete a file that is still mapped
    del data
    os.remove(all_path) # the split files replace it

    # read back and decode: proves text -> tokens -> file -> tokens -> text works
    check = np.fromfile(cfg.train_bin_path, dtype=np.uint16, count=60)
    print(repr(tok.decode(check.tolist())))
