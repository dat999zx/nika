import torch
import torch.nn as nn
from torch.nn import functional as F
from config import ModelConfig

class Nika(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.n_embed)
        self.lm_head = nn.Linear(cfg.n_embed, cfg.vocab_size)
    
    def forward(self, idx, targets=None):
        x = self.token_embed(idx) # (B, T, C) get embedding vector
        logits = self.lm_head(x) # (B, T, vocab_size) calculate score for each token in vocab, best is predicted as next token
        
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            # cross_entropy(x = (N, V), y = (N,))
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        
        return logits, loss