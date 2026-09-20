from dataclasses import dataclass

@dataclass
class TokenizerConfig:
    data_path: str = "data/tiny.txt"
    train_bytes: int = 10_000_000
    vocab_size: int = 4096
    save_path: str = "data/tokenizer.txt"

@dataclass
class DataConfig:
    data_path: str = "data/tiny.txt"
    tokenizer_path: str = "data/tokenizer.txt"
    train_bin_path: str = "data/train.bin"
    val_bin_path: str = "data/val.bin"
    val_fraction: float = 0.1