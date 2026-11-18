import numpy as np
import torch
from src.config import N, NUM_CELLS, SEP, MASK


def _att_layers(model):
    """Attention saved during the last forward pass: list of (heads, T, T), one per layer."""
    return [b.attn.last_att[0].cpu().numpy() for b in model.blocks]


@torch.no_grad()
def ar_step(ar, puzzle_flat, answers_so_far, device):
    """Run the transformer once to predict the NEXT cell.
    Returns probs (8,) and attn (layers, heads, T): how much the last token looks at every earlier token."""
    ids = list(puzzle_flat) + [SEP] + list(answers_so_far)
    x = torch.tensor(ids, dtype=torch.long, device=device)[None]
    logits = ar(x)
    probs = logits[0, -1].softmax(-1).cpu().numpy()
    attn = np.stack([a[:, -1, :] for a in _att_layers(ar)])
    return probs, attn


@torch.no_grad()
def ncm_pass(ncm, state_flat, k, device):
    """Run NCM-lite for pass k (0-based). state_flat: 0 = hidden cell.
    Returns probs (36, 8) and attn (layers, heads, 36, 36)."""
    s = torch.tensor(state_flat, dtype=torch.long, device=device)[None]
    s = torch.where(s > 0, s, torch.full_like(s, MASK))
    kk = torch.full((1,), k, dtype=torch.long, device=device)
    logits = ncm(s, kk)
    probs = logits[0].softmax(-1).cpu().numpy()
    attn = np.stack(_att_layers(ncm))
    return probs, attn


def pick_head(attn_lh, head):
    """attn_lh: (heads, ...). Choose one head, or the average of all heads."""
    return attn_lh.mean(0) if head == "avg" else attn_lh[head]