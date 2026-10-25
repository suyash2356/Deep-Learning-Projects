import os, re, json
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
import streamlit.components.v1 as components
from model import MultiHeadAttention, make_src_mask, make_tgt_mask

_HERE = os.path.dirname(os.path.abspath(__file__))
NODES, TOPK_LINKS, TOPK_LOOK = 8, 6, 12


def _group(sp, ids):
    """Group SentencePiece ids into words: returns (word texts, word index of each token, token indices per word)."""
    groups = []
    for i, t in enumerate(ids):
        if i == 0 or sp.id_to_piece(int(t)).startswith("▁"):
            groups.append([])
        groups[-1].append(i)
    words = [sp.decode([int(ids[j]) for j in g]).strip() for g in groups]
    wof = np.zeros(len(ids), dtype=np.int64)
    for w, g in enumerate(groups):
        wof[g] = w
    return words, wof, groups


def _norm(w):
    return re.sub(r"[\W_]+", "", re.sub(r"['’]s$", "", w.lower()))


def _show(sp, t, eos):
    if t == eos:
        return "end"
    p = sp.id_to_piece(int(t))
    return "?" if p.startswith("<0x") else (p.replace("▁", "") or "·")


def _pool(x):
    """(n, 256) -> (n, NODES): the strongest value inside each group of dimensions."""
    a = x.float().cpu().numpy()
    cols = [a[:, g][np.arange(len(a)), np.abs(a[:, g]).argmax(1)] for g in np.array_split(np.arange(a.shape[1]), NODES)]
    return np.stack(cols, 1)


def build_trace(sm, article, min_len=25, no_repeat=3):
    """Run one article through the Summarizer's model and record everything the animation shows."""
    mods = [m for m in sm.net.modules() if isinstance(m, MultiHeadAttention)]
    for m in mods:
        m.save_attn = True
    try:
        return _trace(sm, article, min_len, no_repeat)
    finally:
        for m in mods:
            m.save_attn, m.attn_weights = False, None


@torch.no_grad()
def _trace(sm, article, min_len, no_repeat):
    net, sp, dev, PAD, BOS, EOS = sm.net, sm.sp, sm.device, sm.PAD, sm.BOS, sm.EOS
    net.eval()
    clean = re.sub(r"\s+", " ", article).strip()[:4000]
    all_ids = sp.encode(clean)
    n = min(len(all_ids), sm.max_src - 2)                    # tokens the model really reads
    words, wof, groups = _group(sp, all_ids)
    nr = int(wof[n - 1]) + 1                                 # words the model reads
    M = np.zeros((nr, n), dtype=np.float32)
    M[wof[:n], np.arange(n)] = 1                             # word x token membership
    Mn = M / M.sum(1, keepdims=True)

    # ---------- the reader (encoder) ----------
    src = torch.tensor([[BOS] + all_ids[:n] + [EOS]], device=dev)
    src_mask = make_src_mask(src, PAD)
    memory = net.encode(src, src_mask)
    layers = []
    for L in net.encoder.layers:
        A = L.self_attn.attn_weights[0].float().mean(0).cpu().numpy()[1:n + 1, 1:n + 1]
        Aw = Mn @ A @ M.T                                    # word-to-word attention
        imp = Aw.mean(0)                                     # how much attention each word receives
        links = []
        for i in range(nr):
            row = Aw[i].copy()
            row[i] = 0
            links.append([[int(j), round(float(row[j]), 3)] for j in np.argsort(-row)[:TOPK_LINKS] if row[j] > 0.01])
        layers.append({"imp": np.round(imp / max(imp.max(), 1e-9), 3).tolist(), "links": links})
    mem = Mn @ _pool(memory[0, 1:n + 1])
    mem = np.round(mem / max(np.abs(mem).max(), 1e-9), 2).tolist()

    # ---------- the writer (decoder), same rules as Summarizer (greedy) ----------
    out, steps = [BOS], []
    for step in range(sm.max_tgt - 2):
        tgt = torch.tensor([out], device=dev)
        lg = net.decode(tgt, memory, src_mask, make_tgt_mask(tgt, PAD))[0, -1].float()
        raw = int(lg.argmax())
        lp = F.log_softmax(lg, -1)
        lp[PAD] = lp[BOS] = -1e9
        if step < min_len:
            lp[EOS] = -1e9
        if no_repeat > 1 and len(out) >= no_repeat:
            pre = tuple(out[-(no_repeat - 1):])
            ban = [out[i + no_repeat - 1] for i in range(len(out) - no_repeat + 1) if tuple(out[i:i + no_repeat - 1]) == pre]
            if ban:
                lp[ban] = -1e9
        nxt = int(lp.argmax())
        pv, pi = lp.exp().topk(5)
        looks = []
        for L in net.decoder.layers:
            a = L.cross_attn.attn_weights[0, :, -1, :].float().mean(0).cpu().numpy()[1:n + 1]
            aw = M @ a                                       # attention per article word
            looks.append([[int(j), round(float(aw[j]), 3)] for j in np.argsort(-aw)[:TOPK_LOOK] if aw[j] > 0.005])
        steps.append({"id": nxt, "looks": looks, "over": _show(sp, raw, EOS) if raw != nxt else "",
                      "cands": [{"w": _show(sp, i, EOS), "p": round(float(p), 3)} for p, i in zip(pv.tolist(), pi.tolist())]})
        out.append(nxt)
        if nxt == EOS:
            break

    # ---------- label every summary word: copied / new / check ----------
    ids = [s["id"] for s in steps if s["id"] != EOS]
    sw, swof, _ = _group(sp, ids)
    seen = {_norm(w) for w in words[:nr]} - {""}
    stat = []
    for k, w in enumerate(sw):
        nw, init = _norm(w), (k == 0 or sw[k - 1][-1:] in ".!?")
        if not nw:
            stat.append("none")
        elif nw in seen:
            stat.append("copied")
        elif any(c.isdigit() for c in w) or (w[:1].isupper() and not init):
            stat.append("check")
        else:
            stat.append("new")
    for k, s in enumerate(steps):
        if s["id"] == EOS:
            s["piece"], s["status"] = "", "end"
        else:
            p = sp.id_to_piece(s["id"])
            s["piece"], s["status"] = ("" if p.startswith("<0x") else p.replace("▁", " ")), stat[int(swof[k])]
        del s["id"]

    ex = [{"word": words[w], "pieces": [sp.id_to_piece(all_ids[j]).replace("▁", "") for j in g]}
          for w, g in enumerate(groups[:nr]) if len(g) > 1 and len(words[w]) > 5][:4]
    alpha = [s for s in stat if s != "none"]
    return {"words": words[:600], "n_read": nr, "n_words": len(clean.split()), "n_tok": n, "examples": ex,
            "layers": layers, "memory": mem, "steps": steps, "summary": sp.decode(ids),
            "stats": {"n": len(alpha), "copied": sum(s == "copied" for s in alpha),
                      "check": list(dict.fromkeys(w for w, s in zip(sw, stat) if s == "check"))}}


def render_html(trace):
    with open(os.path.join(_HERE, "xray_sum_template.html"), encoding="utf-8") as f:
        t = f.read()
    return t.replace("__DATA__", json.dumps(trace, ensure_ascii=False).replace("</", "<\\/"))


@st.cache_data(show_spinner="Recording what happens inside the model...")
def _page(article, _sm):
    return render_html(build_trace(_sm, article))


def streamlit_section(sm, article):
    """Add the animated 'how the model works' section to the Streamlit app."""
    st.header("How the model reads and writes")
    st.caption("A step-by-step animation of what happens inside the model for the article above.")
    if len(article.split()) < 40:
        st.info("Paste an article of at least 40 words above to see the animation.")
        return
    if st.checkbox("Show the animation for this article"):
        components.html(_page(article, sm), height=1000, scrolling=True)
