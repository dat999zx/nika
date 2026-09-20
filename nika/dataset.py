import numpy as np
import torch

class TokenDataset:
    def __init__(self, bin_path, block_size, batch_size, device):
        # read file without loading into RAM
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = torch.device(device)
    
    def get_batch(self):
        # pick random starting spots
        # torch.randint(high, shape)
        ix = torch.randint(len(self.data) - self.block_size - 1, (self.batch_size,))
        
        x = []
        y = []
        
        for i in ix:
            # self-supervised
            x_i = torch.from_numpy(self.data[i:i + self.block_size].astype(np.int64)) # feed forward
            y_i = torch.from_numpy(self.data[i + 1:i + self.block_size + 1].astype(np.int64)) # 1 token later is target
            
            x.append(x_i)
            y.append(y_i)
        
        x = torch.stack(x)
        y = torch.stack(y)
        
        return x.to(self.device), y.to(self.device)

if __name__ == "__main__":
    from config import DataConfig, TrainConfig
    from nika.tokenizer import BPETokenizer

    dcfg, tcfg = DataConfig(), TrainConfig()
    ds = TokenDataset(dcfg.train_bin_path, tcfg.block_size, tcfg.batch_size, "cpu")
    x, y = ds.get_batch()
    print(x.shape, y.shape, x.dtype)
    assert torch.equal(x[:, 1:], y[:, :-1])

    tok = BPETokenizer.load(dcfg.tokenizer_path)
    print(repr(tok.decode(x[0].tolist())))