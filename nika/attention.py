import torch
import torch.nn as nn
from torch.nn import functional as F

# 1 attention head
class AttentionHead(nn.Module):
    def __init__(self, n_embed, head_size, block_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size))) # casual mask

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x) # (B, T, head_size)
        k = self.key(x)
        v = self.value(x)
        
        scores = q @ k.transpose(-2, -1) / (self.head_size ** 0.5) # (B, T, T)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1) # (B, T, T)
        return weights @ v # (B, T, T) @ (B, T, head_size) = (B, T, head_size)

if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, C, hs = 2, 8, 16, 4
    head = AttentionHead(C, hs, block_size=T)
    x = torch.randn(B, T, C)
    out = head(x)
    print(out.shape)                                   # (2, 8, 4)

    # THE LEAK TEST: change the LAST token, earlier outputs must not move
    x2 = x.clone()
    x2[:, -1, :] = torch.randn(B, C)                   # rewrite position 7 only
    out2 = head(x2)
    assert torch.allclose(out[:, :-1], out2[:, :-1]), "FUTURE LEAK"
    assert not torch.allclose(out[:, -1], out2[:, -1]) # the last one SHOULD change
    print("no future leak")