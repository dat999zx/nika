from dataclasses import dataclass

DATA_PATH = "data/tiny.txt"
VOCAB_SIZE = 4096

@dataclass
class TokenizerConfig:
    data_path: str = DATA_PATH
    train_bytes: int = 10_000_000
    vocab_size: int = VOCAB_SIZE
    save_path: str = "data/tokenizer.txt"

@dataclass
class DataConfig:
    data_path: str = DATA_PATH
    tokenizer_path: str = "data/tokenizer.txt"
    train_bin_path: str = "data/train.bin"
    val_bin_path: str = "data/val.bin"
    val_fraction: float = 0.1

@dataclass
class TrainConfig:
    block_size: int = 128 # context size
    batch_size: int = 32 # nums of docs feed into each training step
    device: str = "cuda"

@dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    n_embed: int = 64 # embedding vector size
    block_size: int = 128 # context size