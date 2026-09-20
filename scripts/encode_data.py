import os
from multiprocessing import Pool

import numpy as np

from config import DataConfig
from nika.tokenizer import BPETokenizer, EOT

BATCH = 500 # documents per chunk

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

if __name__ == "__main__":
    cfg = DataConfig()
    tok = BPETokenizer.load(cfg.tokenizer_path)

    with open(cfg.data_path, encoding="utf-8") as f:
        text = f.read()
    print(f"{len(text) / 1e6:.1f} M chars")

    docs = text.split(EOT)
    batches = [EOT.join(docs[i:i + BATCH]) for i in range(0, len(docs), BATCH)]
    del text, docs # a few hundred MB, and the workers get their own copies of the batches

    # imap, not imap_unordered: the array order IS the text order, and the split depends on it
    arrays = []
    with Pool(initializer=_init, initargs=(cfg.tokenizer_path,)) as pool:
        for n, arr in enumerate(pool.imap(_encode_batch, batches), 1):
            arrays.append(arr)
            if n % 10 == 0:
                print(f"{n}/{len(batches)} batches")

    all_ids = np.concatenate(arrays)
    print(f"{len(all_ids) / 1e6:.1f} M tokens")
    print(int((all_ids == tok.eot_id).sum()), "EOT tokens")

    # contiguous split: val is the LAST 10%, text the model never trains on
    n_val = int(len(all_ids) * cfg.val_fraction)
    train_ids = all_ids[:-n_val]
    val_ids = all_ids[-n_val:]

    os.makedirs(os.path.dirname(cfg.train_bin_path), exist_ok=True)
    train_ids.tofile(cfg.train_bin_path)
    val_ids.tofile(cfg.val_bin_path)
    print(f"{len(train_ids) / 1e6:.1f} M train, {len(val_ids) / 1e6:.1f} M val")

    # read back and decode: proves text -> tokens -> file -> tokens -> text works
    check = np.fromfile(cfg.train_bin_path, dtype=np.uint16)
    assert len(check) == len(train_ids)
    print(repr(tok.decode(check[:60].tolist())))
