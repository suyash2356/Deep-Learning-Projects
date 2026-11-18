import numpy as np
import torch
from src.config import N, NUM_CELLS, SEP, MASK
from src.groups import LEVEL_FLAT, NUM_LEVELS
from src.models import SudokuGPT, NCMLite


def load_models(device):
    ar = SudokuGPT().to(device)
    ar.load_state_dict(torch.load("checkpoints/ar_best.pt", map_location=device))
    ncm = NCMLite().to(device)
    ncm.load_state_dict(torch.load("checkpoints/ncm_best.pt", map_location=device))
    return ar.eval(), ncm.eval()


@torch.no_grad()
def trace_ar(ar, puz_flat, device):
    """Returns 37 grids (flat, 0 = blank): grid[t] = state after t forward passes."""
    puz = torch.tensor(puz_flat, dtype=torch.long, device=device)[None]
    seq = torch.cat([puz, torch.full((1, 1), SEP, dtype=torch.long, device=device)], 1)
    cur = np.array(puz_flat, dtype=int)
    states = [cur.copy()]
    for i in range(NUM_CELLS):
        nxt = ar(seq)[:, -1, 1:N + 1].argmax(-1) + 1
        nxt = torch.where(puz[:, i] > 0, puz[:, i], nxt)      # clues are copied
        seq = torch.cat([seq, nxt[:, None]], 1)
        cur[i] = int(nxt.item())
        states.append(cur.copy())
    return states


@torch.no_grad()
def trace_ncm(ncm, puz_flat, device):
    """Returns 4 grids: grid[p] = state after p passes (p = 0..3)."""
    puz = torch.tensor(puz_flat, dtype=torch.long, device=device)[None]
    level = torch.tensor(LEVEL_FLAT, device=device)
    state = torch.where(puz > 0, puz, torch.full_like(puz, MASK))

    def snap(s):
        a = s[0].cpu().numpy().astype(int)
        a[a == MASK] = 0
        return a

    states = [snap(state)]
    for k in range(NUM_LEVELS):
        kk = torch.full((1,), k, dtype=torch.long, device=device)
        pred = ncm(state, kk)[:, :, 1:N + 1].argmax(-1) + 1
        state = torch.where((level[None, :] == k) & (state == MASK), pred, state)
        states.append(snap(state))
    return states