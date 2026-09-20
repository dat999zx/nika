import time

from config import TokenizerConfig
from nika.tokenizer import BPETokenizer

if __name__ == "__main__":
    cfg = TokenizerConfig()
    print(cfg)

    with open(cfg.data_path, encoding="utf-8") as f:
        text = f.read(cfg.train_bytes)

    tok = BPETokenizer()
    start = time.time()
    tok.train(text, cfg.vocab_size, verbose=True)
    print(f"trained in {time.time() - start:.0f} s, vocab size {len(tok.vocab)}")

    tok.save(cfg.save_path)
    print(f"saved to {cfg.save_path}")

    with open(cfg.data_path, encoding="utf-8") as f:
        f.read(cfg.train_bytes)
        sample = f.read(1000)

    ids = tok.encode(sample)
    assert tok.decode(ids) == sample
    print(f"round trip ok, {len(sample) / len(ids):.2f} chars per token")
    print([tok.decode([i]) for i in ids[:40]])
