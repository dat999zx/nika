import torch.nn as nn
from nika.attention import FastMultiHeadAttention
from nika.mlp import MLP

class Block(nn.Module):
    def __init__(self, n_embed, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attention = FastMultiHeadAttention(n_embed, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embed)
        self.mlp = MLP(n_embed)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x