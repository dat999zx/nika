import numpy as np
from config import DataConfig
from nika.tokenizer import BPETokenizer, EOT

BATCH = 500 # documents per chunk

if __name__ == "__main__":
    cfg = DataConfig()
    tok = BPETokenizer.load(cfg.tokenizer_path)
    
    with open(cfg.data_path, encoding="utf-8") as f:
        text = f.read()
    print(f"{len(text) / 1e6:.1f} M chars")
    
    docs = text.split(EOT)
    arrays = [] # numpy pieces
    
    for i in range(0, len(docs), BATCH):
        batch = docs[i:i + BATCH] # array of BATCH docs
        batch = EOT.join(batch) # into 1 string with EOT at the end
        
        ids = tok.encode(batch)
        ids.append(tok.eot_id)
        
        arrays.append(np.array(ids, dtype=np.uint16))
        
        if i % (BATCH * 10) == 0:
            print(f"{i}/{len(docs)} docs")
    
    all_ids = np.concatenate(arrays)
    print(f"{len(all_ids) / 1e6:.1f} M tokens")
    print(int((all_ids == tok.eot_id).sum()), "EOT tokens")

    # contiguous split: val is the LAST 10%, text the model never trains on
    n_val = int(len(all_ids) * cfg.val_fraction)
    train_ids = all_ids[:-n_val]
    val_ids = all_ids[-n_val:]

    train_ids.tofile(cfg.train_bin_path)
    val_ids.tofile(cfg.val_bin_path)
    print(f"{len(train_ids) / 1e6:.1f} M train, {len(val_ids) / 1e6:.1f} M val")

    # read back and decode: proves text -> tokens -> file -> tokens -> text works
    check = np.fromfile(cfg.train_bin_path, dtype=np.uint16)
    assert len(check) == len(train_ids)
    print(repr(tok.decode(check[:60].tolist())))