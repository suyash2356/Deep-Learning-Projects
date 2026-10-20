import os, json, math
import numpy as np
import torch
import torch.nn.functional as F
from model import MultiHeadAttention, make_src_mask, make_tgt_mask

NODES = 8     # each word's 256 numbers are shown as 8 neurons
_HERE = os.path.dirname(os.path.abspath(__file__))


def _lab(sp, i):
    return sp.id_to_piece(int(i)).replace("▁", "") or "_"


def _pool(t, mode="peak"):
    """(L, D) -> (L, NODES): split D into groups, keep the strongest value (or the mean) of each group."""
    a = t.detach().float().cpu().numpy()
    cols = []
    for g in np.array_split(np.arange(a.shape[1]), NODES):
        blk = a[:, g]
        if mode == "peak":
            cols.append(blk[np.arange(len(blk)), np.abs(blk).argmax(1)])
        else:
            cols.append(blk.mean(1))
    return np.round(np.stack(cols, 1), 3).tolist()


def _attn_pack(w):
    """(1, heads, Lq, Lk) -> [9][Lq][Lk]: 8 heads plus their average at index 8."""
    a = w[0].float().cpu().numpy()
    a = np.concatenate([a, a.mean(0, keepdims=True)], 0)
    return np.round(a, 3).tolist()


def build_trace(tr, sentence, max_src=14, max_new=30):
    """Run one sentence through the model and record everything the animation needs."""
    layers = [m for m in tr.net.modules() if isinstance(m, MultiHeadAttention)]
    for m in layers:
        m.save_attn = True
    try:
        return _trace(tr, sentence, max_src, max_new)
    finally:
        for m in layers:                 # leave the shared model exactly as we found it
            m.save_attn = False
            m.attn_weights = None


@torch.no_grad()
def _trace(tr, sentence, max_src, max_new):
    net, sp, device = tr.net, tr.sp, tr.device
    PAD, BOS, EOS = tr.PAD, tr.BOS, tr.EOS
    net.eval()

    ids = sp.encode(sentence)[:max_src]
    src = torch.tensor([[BOS] + ids + [EOS]], device=device)
    src_mask = make_src_mask(src, PAD)
    d = net.d_model

    # ---------------- encoder ----------------
    emb = net.embed(src) * math.sqrt(d)
    x = net.pos(emb)
    bands = [
        {"name": "English words", "kind": "tokens"},
        {"name": "Embedding", "kind": "dense", "vec": _pool(emb[0])},
        {"name": "+ Position", "kind": "dense", "vec": _pool(x[0])},
    ]
    for li, L in enumerate(net.encoder.layers):
        h = L.norm1(x)
        x = x + L.self_attn(h, h, h, src_mask)
        bands.append({"name": f"L{li + 1} Attention", "kind": "attn",
                      "vec": _pool(x[0]), "attn": _attn_pack(L.self_attn.attn_weights)})
        h2 = L.norm2(x)
        hid = F.relu(L.ff.fc1(h2))
        x = x + L.ff.fc2(hid)
        bands.append({"name": f"L{li + 1} FFN neurons", "kind": "dense", "vec": _pool(hid[0], "mean")})
        bands.append({"name": f"L{li + 1} FFN out", "kind": "dense", "vec": _pool(x[0])})
    memory = net.encoder.norm(x)
    bands.append({"name": "Memory", "kind": "dense", "vec": _pool(memory[0])})

    # ---------------- greedy decoding ----------------
    out = [BOS]
    for _ in range(max_new):
        tgt = torch.tensor([out], device=device)
        logits = net.decode(tgt, memory, src_mask, make_tgt_mask(tgt, PAD))
        nxt = logits[0, -1].argmax().item()
        out.append(nxt)
        if nxt == EOS:
            break
    tgt_in_ids, tgt_out_ids = out[:-1], out[1:]

    # ---------------- decoder pass over the final output ----------------
    tgt_in = torch.tensor([tgt_in_ids], device=device)
    tgt_mask = make_tgt_mask(tgt_in, PAD)
    y_emb = net.embed(tgt_in) * math.sqrt(d)
    y = net.pos(y_emb)
    dec = {"embed": _pool(y_emb[0]), "pos": _pool(y[0]), "layers": []}
    for L in net.decoder.layers:
        h = L.norm1(y)
        y = y + L.self_attn(h, h, h, tgt_mask)
        self_vec = _pool(y[0])
        h = L.norm2(y)
        y = y + L.cross_attn(h, memory, memory, src_mask)
        cross_vec = _pool(y[0])
        h = L.norm3(y)
        hid = F.relu(L.ff.fc1(h))
        y = y + L.ff.fc2(hid)
        dec["layers"].append({
            "self_vec": self_vec, "cross_vec": cross_vec,
            "ffnh_vec": _pool(hid[0], "mean"), "out_vec": _pool(y[0]),
            "self_attn": _attn_pack(L.self_attn.attn_weights),
            "cross_attn": _attn_pack(L.cross_attn.attn_weights),
        })

    probs = F.softmax(net.out_proj(net.decoder.norm(y))[0].float(), dim=-1)
    top = []
    for t in range(probs.size(0)):
        p, i = probs[t].topk(5)
        top.append([{"w": _lab(sp, ii), "p": round(float(pp), 3)} for pp, ii in zip(p.tolist(), i.tolist())])

    dec["tgt_in"] = [_lab(sp, i) for i in tgt_in_ids]
    dec["tgt_out"] = [_lab(sp, i) for i in tgt_out_ids]
    dec["top"] = top
    dec["prefix"] = [sp.decode([i for i in tgt_out_ids[:k] if i > 3]) for k in range(len(tgt_out_ids) + 1)]

    return {"sentence": sentence, "N": NODES, "src": [_lab(sp, i) for i in src[0].tolist()],
            "enc": {"bands": bands}, "dec": dec, "translation": dec["prefix"][-1]}


def render_html(trace):
    """Put the recorded trace inside the animation page and return the full HTML text."""
    with open(os.path.join(_HERE, "xray_template.html"), encoding="utf-8") as f:
        template = f.read()
    return template.replace("__DATA__", json.dumps(trace, ensure_ascii=False))
