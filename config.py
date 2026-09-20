from dataclasses import dataclass

@dataclass
class TokenizerConfig:
    data_path: str = "data/tiny.txt"
    train_bytes: int = 10_000_000
    vocab_size: int = 4096
    save_path: str = "data/tokenizer.txt"