from dataclasses import dataclass

DATA_PATH = "data/tiny.txt"
VOCAB_SIZE = 4096 # how many type of token it can learn
BLOCK_SIZE = 128 # context size (tokens)

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
    block_size: int = BLOCK_SIZE # T
    batch_size: int = 32 # B: windows per training step
    device: str = "cuda"
    max_iters: int = 8000 # training steps, run train.bin
    learning_rate: float = 1e-3
    eval_interval: int = 250 # measure train/val loss every this many steps
    eval_iters: int = 20 # batches averaged per measurement from train.bin and val.bin (to check bias-variance)
    warmup_iters: int = 200 # steps to ramp the LR up from 0 to max
    min_lr_frac: float = 0.1 # at the end of training, lr is 10% of max not 0
    checkpoint_path: str = "checkpoint/nika.pt"

@dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    n_embed: int = 256 # embedding vector size
    n_layer: int = 4 # number of blocks
    block_size: int = BLOCK_SIZE
    n_head: int = 4 # attention heads, n_embed must divide evenly by this