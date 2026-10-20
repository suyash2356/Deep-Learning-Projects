import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))      # (1, max_len, d_model)

    def forward(self, x):                                # x: (B, L, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


def make_src_mask(src, pad_id):
    # (B, 1, 1, S): True where the token is real (not pad)
    return (src != pad_id).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt_in, pad_id):
    # (B, 1, T, T): real tokens only AND no peeking at future positions
    T = tgt_in.size(1)
    pad_mask = (tgt_in != pad_id).unsqueeze(1).unsqueeze(2)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=tgt_in.device))
    return pad_mask & causal


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.h = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.save_attn = False          # set True later to visualize attention
        self.attn_weights = None

    def forward(self, q, k, v, mask=None):
        B = q.size(0)

        def split(x, layer):            # (B, L, d_model) -> (B, h, L, d_k)
            return layer(x).view(B, -1, self.h, self.d_k).transpose(1, 2)

        Q, K, V = split(q, self.w_q), split(k, self.w_k), split(v, self.w_v)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)     # (B, h, Lq, Lk)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)
        if self.save_attn:
            self.attn_weights = attn.detach()
        attn = self.dropout(attn)

        out = attn @ V                                              # (B, h, Lq, d_k)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.w_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        h = self.norm1(x)
        x = x + self.drop(self.self_attn(h, h, h, src_mask))
        h = self.norm2(x)
        x = x + self.drop(self.ff(h))
        return x


class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        h = self.norm1(x)
        x = x + self.drop(self.self_attn(h, h, h, tgt_mask))
        h = self.norm2(x)
        x = x + self.drop(self.cross_attn(h, memory, memory, src_mask))
        h = self.norm3(x)
        x = x + self.drop(self.ff(h))
        return x


class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=4,
                 d_ff=1024, dropout=0.1, max_len=128, pad_id=0):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        self.encoder = Encoder(n_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ff, dropout)

        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.out_proj.weight = self.embed.weight          # weight tying
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.embed.weight, mean=0.0, std=self.d_model ** -0.5)
        with torch.no_grad():
            self.embed.weight[self.pad_id].fill_(0.0)

    def encode(self, src, src_mask):
        x = self.pos(self.embed(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask)

    def decode(self, tgt_in, memory, src_mask, tgt_mask):
        x = self.pos(self.embed(tgt_in) * math.sqrt(self.d_model))
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.out_proj(x)                           # (B, T, vocab)

    def forward(self, src, tgt_in):
        src_mask = make_src_mask(src, self.pad_id)
        tgt_mask = make_tgt_mask(tgt_in, self.pad_id)
        memory = self.encode(src, src_mask)
        return self.decode(tgt_in, memory, src_mask, tgt_mask)
