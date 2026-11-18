import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import random
import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt

from src.config import N, NUM_CELLS
from src.groups import LEVEL_FLAT, NUM_LEVELS
from src.dataset import load
from src.solvers import load_models, trace_ar, trace_ncm
from src.internals import ar_step, ncm_pass, pick_head
from src.sudoku import count_violations
from src.viz import grid_html, heat, mask_fig, GROUP_COLORS, TRANSFORMER_DOT, NCM_DOT

st.set_page_config(page_title="Inside the models", layout="wide")

CLUE, KNOWN, EMPTY, BLUE = "#d0d0d0", "#dbe9f6", "#ffffff", "#3498db"
GOOD, BAD = "#c8f0c8", "#f5b7b1"


# ---------- loading (cached) ----------
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


def dig(v, blank="?"):
    return str(int(v)) if v > 0 else blank


def norm(a):
    return a / (a.max() + 1e-9)


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


# ---------- sidebar ----------
P, S = get_test()
ar, ncm, device = get_models()

with st.sidebar:
    st.header("Controls")
    if "inside_idx" not in st.session_state:
        st.session_state["inside_idx"] = 0
    st.number_input("Test puzzle number (0-999)", 0, len(P) - 1, key="inside_idx")
    st.button("Random puzzle",
              on_click=lambda: st.session_state.update(inside_idx=random.randrange(len(P))))
    st.subheader("Attention view")
    layer = st.selectbox("Layer (1 = first, 4 = last)", [0, 1, 2, 3], index=3,
                         format_func=lambda x: f"layer {x + 1}")
    head = st.selectbox("Head", ["avg", 0, 1, 2, 3],
                        format_func=lambda x: "average of 4 heads" if x == "avg" else f"head {x + 1}")

idx = int(st.session_state["inside_idx"])
puzzle, truth = P[idx], S[idx]
ar_states, ncm_states = get_traces(idx)

st.title("Inside the models: how each one thinks")
st.caption(f"Test puzzle #{idx}. Everything below is real data from your trained models. "
           "Attention maps show where a model looks. They are a useful hint, not a full explanation "
           "of why it decides what it decides.")


# ---------- transformer ----------
def render_transformer():
    st.subheader("1. The architecture")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.graphviz_chart(TRANSFORMER_DOT)
    with c2:
        show_fig(mask_fig(True))
        st.caption("Causal mask. Green = allowed to look. Token 3 sees tokens 1-3 only, never the future. "
                   "That is why the transformer can only fill cells in order.")

    st.subheader("2. Watch it fill one cell")
    step = st.slider("Which cell is being filled? (1 to 36)", 1, NUM_CELLS, 1, key="ar_slider")
    i = step - 1
    row, col = divmod(i, N)
    answers = ar_states[i][:i]
    probs, attn = ar_step(ar, puzzle, answers, device)

    # Step A: read
    st.markdown(f"#### Step A: Read\nThe model gets the puzzle, then SEP, then the answers it has "
                f"already written. Now it must write answer cell **#{step}** (row {row + 1}, column {col + 1}).")
    cur = ar_states[i]
    bgs = [CLUE if puzzle[j] > 0 else (KNOWN if j < i else EMPTY) for j in range(NUM_CELLS)]
    a1, a2 = st.columns(2)
    with a1:
        st.markdown(grid_html([dig(v, "") for v in cur], bgs, {i: BLUE}), unsafe_allow_html=True)
        st.caption("Grey = clue · light blue = written by the model · blue ring = cell being predicted")
    with a2:
        pz = " ".join(dig(v, ".") for v in puzzle)
        sofar = " ".join(dig(v, ".") for v in answers) if i else "(nothing yet)"
        st.code(f"puzzle ({NUM_CELLS}) : {pz}\nSEP\nanswers ({i})   : {sofar}\n"
                f"--> predict answer cell #{step}", language=None)

    # Step B: look
    st.markdown("#### Step B: Look (attention)\nThe last token looks back at earlier tokens "
                "to decide. Darker orange = more attention.")
    a = pick_head(attn[layer], head)
    top = max(float(a.max()), 1e-9)
    a_puz, a_sep, a_ans = a[:NUM_CELLS], float(a[NUM_CELLS]), a[NUM_CELLS + 1:]
    full_ans = np.zeros(NUM_CELLS)
    full_ans[:i] = a_ans
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("<p style='text-align:center'><b>Looking at the puzzle</b></p>", unsafe_allow_html=True)
        st.markdown(grid_html([dig(v, "") for v in puzzle],
                              [heat(a_puz[j] / top) for j in range(NUM_CELLS)], {i: BLUE}),
                    unsafe_allow_html=True)
    with b2:
        st.markdown("<p style='text-align:center'><b>Looking at its own answers so far</b></p>",
                    unsafe_allow_html=True)
        st.markdown(grid_html([dig(answers[j], "") if j < i else "" for j in range(NUM_CELLS)],
                              [heat(full_ans[j] / top) if j < i else "#e8e8e8" for j in range(NUM_CELLS)],
                              {i: BLUE}), unsafe_allow_html=True)
        st.caption("Grey = not written yet. The causal mask hides these cells from the model.")
    head_txt = "average of 4 heads" if head == "avg" else f"head {head + 1}"
    st.caption(f"Share of attention: puzzle {a_puz.sum() * 100:.0f}% · SEP {a_sep * 100:.0f}% · "
               f"its own answers {a_ans.sum() * 100:.0f}%  (layer {layer + 1}, {head_txt})")

    # Step C: decide
    st.markdown("#### Step C: Decide\nThe output head turns everything into a probability for each digit. "
                "The best digit is written, then the loop repeats for the next cell.")
    d = int(np.argmax(probs[1:N + 1])) + 1
    df = pd.DataFrame({"probability": probs[1:N + 1]}, index=[str(x) for x in range(1, N + 1)])
    d1, d2 = st.columns([2, 1])
    with d1:
        st.bar_chart(df, height=220)
    with d2:
        if puzzle[i] > 0:
            st.info(f"This cell is a given clue ({puzzle[i]}), so the solver copies it. "
                    f"(The model alone would have picked {d}, {probs[d] * 100:.0f}% sure.)")
        else:
            st.metric("Model picks", f"{d}  ({probs[d] * 100:.0f}% sure)")
            if d == truth[i]:
                st.success(f"Correct (true answer: {truth[i]})")
            else:
                st.error(f"Wrong (true answer: {truth[i]})")

    st.markdown("**What to notice**")
    st.markdown("- One forward pass gives **one** cell. Drag the slider to 36 and you have used 36 passes.\n"
                "- A cell written wrongly early stays in the input for every later step, "
                "so one mistake can spread.\n"
                "- Try layer 1 vs layer 4, and different heads. They look at different things.")


# ---------- NCM-lite ----------
def render_ncm():
    st.subheader("1. The architecture")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.graphviz_chart(NCM_DOT)
    with c2:
        show_fig(mask_fig(False))
        st.caption("No mask. Every cell can look at every other cell: left or right, above or below.")

    st.subheader("2. Watch one pass")
    p = st.radio("Pass number", list(range(1, NUM_LEVELS + 1)), horizontal=True, key="ncm_radio")
    k = p - 1
    state_in = ncm_states[k]
    probs, attn = ncm_pass(ncm, state_in, k, device)
    targets = [int(g) for g in np.where(LEVEL_FLAT == k)[0] if puzzle[g] == 0]
    if not targets:
        st.info("Every cell of this group is a clue, so there is nothing to predict in this pass.")
        return

    # Step A: input
    st.markdown(f"#### Step A: Input\nPass {p} starts with the clues and whatever earlier passes wrote. "
                f"`?` = still hidden. The {len(targets)} tinted cells are this pass's group.")
    bgs = []
    for j in range(NUM_CELLS):
        if puzzle[j] > 0:
            bgs.append(CLUE)
        elif state_in[j] > 0:
            bgs.append(KNOWN)
        elif j in targets:
            bgs.append(GROUP_COLORS[k])
        else:
            bgs.append(EMPTY)
    a1, a2 = st.columns(2)
    with a1:
        st.markdown(grid_html([dig(v) for v in state_in], bgs), unsafe_allow_html=True)
    with a2:
        st.caption("Grey = clue · light blue = written by an earlier pass · tinted = cells to predict now · "
                   "white = hidden for later passes")
        st.markdown("The model also receives the **group number** of each cell and the **pass number**, "
                    "so it knows which cells it is responsible for.")

    # Step B: look
    st.markdown("#### Step B: Look (attention)\nPick one cell of this pass and see where it looks. "
                "There is no mask, so it can look anywhere. Darker orange = more attention.")
    q = st.selectbox("Cell to inspect", targets, key=f"ncm_q_{p}",
                     format_func=lambda j: f"row {j // N + 1}, column {j % N + 1}")
    a = norm(pick_head(attn[layer], head)[q])
    b1, b2 = st.columns(2)
    with b1:
        st.markdown(grid_html([dig(v) for v in state_in], [heat(a[j]) for j in range(NUM_CELLS)], {q: BLUE}),
                    unsafe_allow_html=True)
    with b2:
        st.caption("Blue ring = the cell being inspected. Compare with the transformer: here a cell can "
                   "look at cells that come after it too.")

    # Step C: decide
    st.markdown("#### Step C: Decide (all cells at the same time)\nThe output head gives probabilities "
                "for **all 36 cells at once**. We keep only this pass's cells and write their best digits.")
    pred = {t: int(np.argmax(probs[t, 1:N + 1])) + 1 for t in targets}
    texts, cbgs, subs = [], [], []
    for j in range(NUM_CELLS):
        if puzzle[j] > 0:
            texts.append(dig(puzzle[j])); cbgs.append(CLUE); subs.append("")
        elif j in pred:
            d = pred[j]
            texts.append(str(d)); cbgs.append(GOOD if d == truth[j] else BAD)
            subs.append(f"{probs[j, d] * 100:.0f}%")
        elif state_in[j] > 0:
            texts.append(dig(state_in[j])); cbgs.append(KNOWN); subs.append("")
        else:
            texts.append("?"); cbgs.append(EMPTY); subs.append("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(grid_html(texts, cbgs, {q: BLUE}, subs), unsafe_allow_html=True)
        st.caption("Small % = how sure the model is. Green/red compares with the true answer "
                   "(the model never sees it).")
    with c2:
        dfq = pd.DataFrame({"probability": probs[q, 1:N + 1]}, index=[str(x) for x in range(1, N + 1)])
        st.markdown(f"**Probabilities for the inspected cell (row {q // N + 1}, column {q % N + 1})**")
        st.bar_chart(dfq, height=200)

    state_out = state_in.copy()
    for t, d in pred.items():
        state_out[t] = d
    right = sum(d == truth[t] for t, d in pred.items())
    m1, m2, m3 = st.columns(3)
    m1.metric("Cells filled in this ONE pass", len(targets))
    m2.metric("Correct", f"{right}/{len(targets)}")
    m3.metric("Rule violations in grid", count_violations(state_out.reshape(N, N)))

    st.markdown("**What to notice**")
    st.markdown("- One forward pass fills **12 cells**. The whole puzzle takes 3 passes.\n"
                "- Cells in the same pass are predicted **at the same time**, so they cannot see each other's "
                "answers. That is where rule violations can sneak in.\n"
                "- Once written, a digit is final. NCM-lite cannot go back and fix it "
                "(the real NCM refines the whole board at every level).")


tab_ar, tab_ncm = st.tabs(["Transformer (word by word)", "NCM-lite (group by group)"])
with tab_ar:
    render_transformer()
with tab_ncm:
    render_ncm()

st.divider()
st.subheader("The difference in one table")
st.markdown("""
| | Transformer | NCM-lite |
|---|---|---|
| Attention | Causal (past only) | Bidirectional (everything known) |
| Cells per forward pass | 1 | 12 |
| Forward passes | 36 | 3 |
| Order | Left to right, row by row | Group by group (hand-made hierarchy) |
| Weak spot | One early mistake feeds every later step | Cells in the same pass cannot see each other |
""")
