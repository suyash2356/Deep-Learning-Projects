import numpy as np
import matplotlib.pyplot as plt
from src.config import N, BOX_R, BOX_C

GROUP_COLORS = ["#fbb4ae", "#fed9a6", "#f2f2f2"]     # same colors as results/groups.png


def heat(w):
    """0..1 -> white-to-orange css color (darker = more attention)."""
    w = float(np.clip(w, 0.0, 1.0))
    return f"rgb(255,{int(255 - 120 * w)},{int(255 - 235 * w)})"


def grid_html(texts, bgs, rings=None, subs=None, size=52):
    """Draw an N x N Sudoku grid as HTML.
    texts: N*N strings shown in the cells | bgs: N*N css colors
    rings: {cell_index: css color} outlines | subs: N*N small captions (e.g. '87%')"""
    rings = rings or {}
    rows = ""
    for r in range(N):
        row = ""
        for c in range(N):
            i = r * N + c
            thick, thin = "3px solid #222", "1px solid #888"
            L = thick if c % BOX_C == 0 else thin
            T = thick if r % BOX_R == 0 else thin
            R = thick if c == N - 1 else thin
            B = thick if r == N - 1 else thin
            ring = f"box-shadow: inset 0 0 0 4px {rings[i]};" if i in rings else ""
            sub = ""
            if subs is not None and subs[i]:
                sub = f"<div style='font-size:11px;line-height:11px'>{subs[i]}</div>"
            row += (f"<td style='width:{size}px;height:{size}px;text-align:center;"
                    f"font-family:monospace;font-size:22px;color:#111;background:{bgs[i]};"
                    f"border-left:{L};border-top:{T};border-right:{R};border-bottom:{B};{ring}'>"
                    f"{texts[i]}{sub}</td>")
        rows += f"<tr>{row}</tr>"
    return f"<table style='border-collapse:collapse;margin:auto'>{rows}</table>"


def mask_fig(causal, n=8):
    """Small picture of who-can-look-at-whom."""
    m = np.tril(np.ones((n, n))) if causal else np.ones((n, n))
    fig, ax = plt.subplots(figsize=(2.8, 2.8))
    ax.imshow(m, cmap="Greens", vmin=0, vmax=1.4)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(range(1, n + 1), fontsize=7)
    ax.set_yticklabels(range(1, n + 1), fontsize=7)
    ax.set_xlabel("token being looked at", fontsize=8)
    ax.set_ylabel("token that is looking", fontsize=8)
    ax.set_title("Causal: past only" if causal else "Bidirectional: sees all", fontsize=9)
    plt.tight_layout()
    return fig


TRANSFORMER_DOT = r"""
digraph G {
  rankdir=TB; nodesep=0.3; ranksep=0.35;
  node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Helvetica", fontsize=11];
  A [label="1. INPUT (72 tokens)\npuzzle (36) + SEP + answers written so far"];
  B [label="2. EMBEDDINGS\ntoken embedding + position embedding\n(every token becomes 128 numbers)"];
  subgraph cluster_block {
    label="3. TRANSFORMER BLOCK  x 4"; style="rounded,dashed"; color="#999999"; fontname="Helvetica"; fontsize=11;
    C1 [label="LayerNorm -> CAUSAL self-attention (+ residual)\nevery token looks only at EARLIER tokens", fillcolor="#fde9d9"];
    C2 [label="LayerNorm -> Feed-forward (+ residual)\neach token thinks on its own"];
  }
  D [label="4. OUTPUT HEAD\nLayerNorm -> Linear (128 -> 8)"];
  E [label="5. SOFTMAX\nprobability of each digit for the NEXT cell"];
  F [label="6. PICK the best digit (1-6)\nwrite it, feed it back in", fillcolor="#e2f4e2"];
  A -> B -> C1 -> C2 -> D -> E -> F;
  F -> A [style=dashed, label=" repeat 36 times", constraint=false];
}
"""

NCM_DOT = r"""
digraph G {
  rankdir=TB; nodesep=0.3; ranksep=0.35;
  node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Helvetica", fontsize=11];
  A [label="1. INPUT (36 cells)\ndigits 1-6, or ? for a hidden cell"];
  B [label="2. EMBEDDINGS (4 added together)\ntoken + position + group + pass number"];
  subgraph cluster_block {
    label="3. BLOCK  x 4"; style="rounded,dashed"; color="#999999"; fontname="Helvetica"; fontsize=11;
    C1 [label="LayerNorm -> BIDIRECTIONAL self-attention (+ residual)\nevery cell looks at EVERY cell (no mask)", fillcolor="#fde9d9"];
    C2 [label="LayerNorm -> Feed-forward (+ residual)\neach cell thinks on its own"];
  }
  D [label="4. OUTPUT HEAD\nLayerNorm -> Linear (128 -> 8)\nscores for ALL 36 cells at once"];
  E [label="5. KEEP only this pass's group\n(12 cells, minus the clues)"];
  F [label="6. WRITE their best digits\nthen run the next pass", fillcolor="#e2f4e2"];
  A -> B -> C1 -> C2 -> D -> E -> F;
  F -> A [style=dashed, label=" repeat 3 times", constraint=false];
}
"""