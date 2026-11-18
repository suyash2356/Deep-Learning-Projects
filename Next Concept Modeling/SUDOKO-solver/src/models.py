import math
import torch, torch.nn as nn
from src.config import NUM_CELLS, VOCAB_SIZE
from src.groups import LEVEL_FLAT, NUM_LEVELS

D_MODEL, N_HEADS, N_LAYERS, D_FF = 128, 4, 4, 512
SEQ_LEN = 2 * NUM_CELLS


class Attention(nn.Module):
    def __init__(self, d, h, causal):
        super().__init__()
        self.h, self.dk, self.causal = h, d // h, causal
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dk).transpose(1, 2)
        k = k.view(B, T, self.h, self.dk).transpose(1, 2)
        v = v.view(B, T, self.h, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        if self.causal:                                  # the only difference
            mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
            att = att.masked_fill(~mask, float("-inf"))
        att = att.softmax(dim=-1)
        self.last_att = att.detach()      # saves attention so the visualization page can show it
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, d, h, dff, causal):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(d), Attention(d, h, causal)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, dff), nn.GELU(), nn.Linear(dff, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class SudokuGPT(nn.Module):               # the transformer (causal)
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos = nn.Embedding(SEQ_LEN, D_MODEL)
        self.blocks = nn.ModuleList(
            [Block(D_MODEL, N_HEADS, D_FF, causal=True) for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


class NCMLite(nn.Module):                 # NCM-lite (no causal mask)
    def __init__(self):
        super().__init__()
        self.register_buffer("level", torch.tensor(LEVEL_FLAT, dtype=torch.long),
                             persistent=False)
        self.tok = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos = nn.Embedding(NUM_CELLS, D_MODEL)
        self.cell_group = nn.Embedding(NUM_LEVELS, D_MODEL)
        self.pass_emb = nn.Embedding(NUM_LEVELS, D_MODEL)
        self.blocks = nn.ModuleList(
            [Block(D_MODEL, N_HEADS, D_FF, causal=False) for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, state, k):
        x = (self.tok(state)
             + self.pos(torch.arange(NUM_CELLS, device=state.device))[None]
             + self.cell_group(self.level)[None]
             + self.pass_emb(k)[:, None, :])
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))