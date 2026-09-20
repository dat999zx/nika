import torch
import torch.nn as nn
from torch.nn import functional as F
from config import ModelConfig
from nika.block import Block

class Nika(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.n_embed)
        self.pos_embed = nn.Embedding(cfg.block_size, cfg.n_embed)
        self.blocks = nn.Sequential(*[ # * in python unpack the list
            Block(cfg.n_embed, cfg.n_head, cfg.block_size) for _ in range(cfg.n_layer)
        ])
        self.ln_final = nn.LayerNorm(cfg.n_embed) # before lm_head
        self.lm_head = nn.Linear(cfg.n_embed, cfg.vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_embed(idx) + self.pos_embed(pos) # general context + position
        x = self.blocks(x)
        logits = self.lm_head(self.ln_final(x)) # (B, T, vocab_size) calculate score for each token in vocab, best is predicted as next token
        
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            # cross_entropy(x = (N, V), y = (N,))
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:] # take only last context_size chunk
            
            logits, _ = self(idx_cond) # forward
            logits = logits[:, -1, :] # take last token prediction for next one
            logits /= temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf") # only top k remain
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # weighted random
            idx = torch.cat((idx, idx_next), dim=1) # add predicted token to the end
        return idx