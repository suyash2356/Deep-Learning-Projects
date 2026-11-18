import json, time, random
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.config import N, BOX_R, BOX_C, NUM_CELLS
from src.groups import LEVEL_FLAT, NUM_LEVELS
from src.dataset import load
from src.solvers import load_models, trace_ar, trace_ncm
from src.sudoku import count_violations

st.set_page_config(page_title="Transformer vs NCM-lite on Sudoku", layout="wide")
GROUP_COLORS = ["#fbb4ae", "#fed9a6", "#f2f2f2"]      # same look as results/groups.png


# ---------- cached loading ----------
@st.cache_resource
def get_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ar, ncm = load_models(device)
    return ar, ncm, device


@st.cache_data
def get_test():
    p, s = load("test")
    return p.reshape(len(p), -1).astype(int), s.reshape(len(s), -1).astype(int)


@st.cache_data
def get_traces(idx):
    P, _ = get_test()
    ar, ncm, device = get_models()
    return trace_ar(ar, P[idx], device), trace_ncm(ncm, P[idx], device)


@st.cache_data
def get_outcomes():
    """Which test puzzles each model solved (from saved Phase 3/4 predictions)."""
    try:
        _, S = get_test()
        a = np.load("results/ar_test_preds_forced.npy").reshape(len(S), -1)
        n = np.load("results/ncm_test_preds.npy").reshape(len(S), -1)
        return (a == S).all(1), (n == S).all(1)
    except Exception:
        return None


# ---------- drawing ----------
def grid_html(state, puzzle, truth, new=(), cursor=None, tint=False):
    rows = ""
    for r in range(N):
        row = ""
        for c in range(N):
            i = r * N + c
            v = int(state[i])
            if puzzle[i] > 0:
                bg, fw = "#d0d0d0", "700"
            elif v > 0:
                bg, fw = ("#c8f0c8" if v == truth[i] else "#f5b7b1"), "400"
            else:
                bg, fw = (GROUP_COLORS[LEVEL_FLAT[i]] if tint else "#ffffff"), "400"
            ring = ""
            if i in new:
                ring = "box-shadow: inset 0 0 0 4px #f1c40f;"
            elif cursor == i:
                ring = "box-shadow: inset 0 0 0 4px #3498db;"
            thick, thin = "3px solid #222", "1px solid #888"
            L = thick if c % BOX_C == 0 else thin
            T = thick if r % BOX_R == 0 else thin
            R = thick if c == N - 1 else thin
            B = thick if r == N - 1 else thin
            txt = str(v) if v > 0 else ""
            row += (f'<td style="width:48px;height:48px;text-align:center;font-size:24px;'
                    f'font-family:monospace;color:#111;background:{bg};font-weight:{fw};'
                    f'border-left:{L};border-top:{T};border-right:{R};border-bottom:{B};{ring}">'
                    f'{txt}</td>')
        rows += f"<tr>{row}</tr>"
    return f'<table style="border-collapse:collapse;margin:auto">{rows}</table>'


def draw(t, ar_states, ncm_states, puzzle, truth, tint):
    p = min(t, NUM_LEVELS)
    blank = puzzle == 0

    a_state, n_state = ar_states[t], ncm_states[p]
    a_new = {t - 1} if t >= 1 and puzzle[t - 1] == 0 else set()
    a_cursor = t if t < NUM_CELLS else None
    n_new = set(np.where(ncm_states[p] != ncm_states[p - 1])[0].tolist()) if p >= 1 else set()

    panels = [
        ("Transformer (word by word)", a_state, a_new, a_cursor, t, NUM_CELLS),
        ("NCM-lite (group by group)", n_state, n_new, None, p, NUM_LEVELS),
    ]
    cols = st.columns(2)
    for col, (title, state, new, cur, passes, total) in zip(cols, panels):
        with col:
            st.markdown(f"<h4 style='text-align:center'>{title}</h4>", unsafe_allow_html=True)
            st.markdown(grid_html(state, puzzle, truth, new, cur, tint), unsafe_allow_html=True)
            correct = int(((state == truth) & blank).sum())
            viol = count_violations(state.reshape(N, N))
            st.markdown(
                f"<p style='text-align:center'>Forward pass <b>{passes}/{total}</b> · "
                f"correct blanks <b>{correct}/{int(blank.sum())}</b> · "
                f"rule violations <b>{viol}</b></p>", unsafe_allow_html=True)
            if passes == total:
                if (state == truth).all():
                    st.success("Solved correctly")
                else:
                    st.error(f"Not solved: {int((state != truth).sum())} wrong cells")


# ---------- page ----------
st.title("Transformer vs NCM-lite: solving Sudoku")
st.caption("6x6 Sudoku. Same data, same model size, same rules (given clues are copied, never predicted). "
           "NCM-lite is a simplified version of Next-Concept Modeling: hand-made hierarchy, "
           "whole group predicted in parallel.")

P, S = get_test()
clues = (P > 0).sum(1)
outcomes = get_outcomes()

with st.sidebar:
    st.header("Pick a puzzle")
    lo, hi = st.slider("Clues (fewer = harder)", 12, 24, (12, 24))
    options = ["All puzzles"]
    if outcomes is not None:
        options += ["Transformer fails, NCM-lite solves", "NCM-lite fails, Transformer solves",
                    "Both fail", "Both solve"]
    mode = st.selectbox("Which puzzles?", options)

    mask = (clues >= lo) & (clues <= hi)
    if outcomes is not None:
        a_ok, n_ok = outcomes
        mask &= {"All puzzles": np.ones(len(S), bool),
                 "Transformer fails, NCM-lite solves": ~a_ok & n_ok,
                 "NCM-lite fails, Transformer solves": a_ok & ~n_ok,
                 "Both fail": ~a_ok & ~n_ok,
                 "Both solve": a_ok & n_ok}[mode]
    pool = np.where(mask)[0]
    if len(pool) == 0:
        st.warning("No puzzles match. Widen the filter.")
        st.stop()

    if st.session_state.get("num", 0) > len(pool) - 1:
        st.session_state["num"] = 0
    st.number_input(f"Puzzle number (0 to {len(pool) - 1})", 0, len(pool) - 1, key="num")
    st.button("Random puzzle", on_click=lambda: st.session_state.update(num=random.randrange(len(pool))))

    st.header("Display")
    tint = st.checkbox("Show the 3 groups on empty cells", value=False)
    speed = st.slider("Seconds per step (Play)", 0.05, 1.0, 0.25)

idx = int(pool[st.session_state["num"]])
ar_states, ncm_states = get_traces(idx)
puzzle, truth = P[idx], S[idx]
st.write(f"Test puzzle #{idx} · {int((puzzle > 0).sum())} clues · {int((puzzle == 0).sum())} blanks")

t = st.slider("Forward pass (the transformer needs 36, NCM-lite needs 3)", 0, NUM_CELLS, 0)
play = st.button("Play")
holder = st.empty()

if play:
    for step in range(NUM_CELLS + 1):
        with holder.container():
            draw(step, ar_states, ncm_states, puzzle, truth, tint)
        time.sleep(speed)
else:
    with holder.container():
        draw(t, ar_states, ncm_states, puzzle, truth, tint)

st.caption("Grey = given clue · green = correct · red = wrong · yellow ring = just filled · "
           "blue ring = the cell the transformer fills next")

# ---------- results ----------
st.divider()
st.header("Results on 1000 unseen test puzzles")
try:
    sm = json.load(open("results/phase5_summary.json"))
    names = ["Transformer", "NCM-lite"]
    df = pd.DataFrame({
        "Fully solved %": [round(sm["solved_pct"][n] * 100, 1) for n in names],
        "Cell accuracy %": [round(sm["cell_acc"][n] * 100, 1) for n in names],
        "Avg rule violations": [round(sm["avg_violations"][n], 2) for n in names],
        "Forward passes": [sm["passes"][n] for n in names],
        "ms per puzzle (batch 1)": [round(sm["timing_ms"][n]["batch1"], 1) for n in names],
    }, index=names)
    st.dataframe(df, use_container_width=True)
    st.image("results/comparison.png")
except Exception as e:
    st.info(f"Run the Phase 5 notebook to see results here. ({e})")

with st.expander("What is NCM-lite? What are the limits?"):
    st.markdown(
        "- **Transformer:** reads left to right and fills one cell per forward pass (36 passes).\n"
        "- **NCM-lite:** the 36 cells are split into 3 groups. Each pass fills a whole group at once, "
        "using everything known so far (3 passes).\n"
        "- **Real NCM** learns its coarse-to-fine codes with a VQ-VAE. Here the hierarchy is hand-made.\n"
        "- The two models differ in attention type and fill order as well as parallelism, "
        "so this is a toy comparison, not proof that NCM beats transformers.\n"
        "- The transformer has no KV-cache, so its timing is pessimistic."
    )