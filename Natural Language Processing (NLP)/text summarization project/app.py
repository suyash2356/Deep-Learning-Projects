import os
import html
import time
import streamlit as st
from summarizer import Summarizer

st.set_page_config(page_title="News Article Summarizer", page_icon="📰", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(HERE, "exports", "summarizer_inference.pt")):
    st.error("Model weights not found. Put `summarizer_inference.pt` into the `exports/` folder "
             "(see the README) or train it with the notebook.")
    st.stop()


@st.cache_resource(show_spinner="Loading model...")
def load_summarizer():
    return Summarizer()


sm = load_summarizer()

EXAMPLES = {
    "City bike lanes (fictional)": (
        "The city council of Riverton voted on Tuesday night to approve a new plan to build a bike lane network "
        "across the downtown area. The plan, which passed by seven votes to two, will add 22 miles of protected "
        "lanes over the next three years at a cost of 14 million dollars. Mayor Helen Carter said the project "
        "would make streets safer and reduce traffic. Supporters, including cycling groups and local shop owners, "
        "spoke in favour of the plan at the meeting. Opponents argued that the lanes would remove parking spaces "
        "and hurt small businesses. Construction is expected to begin in March, starting with the main shopping "
        "street. The council will review the results after the first year and decide whether to expand the "
        "network to the suburbs."
    ),
    "New battery (fictional)": (
        "Scientists at a university research centre have developed a new battery that can be charged to full in "
        "under ten minutes and lasts twice as long as the batteries used in most phones today. The team, led by "
        "Dr Marcus Webb, published the results in a scientific journal on Monday. The battery uses a new material "
        "made from common minerals, which the researchers say could make it cheaper to produce than current "
        "designs. Tests showed the battery kept 90 per cent of its capacity after 2,000 charging cycles. Dr Webb "
        "said the next step is to build larger versions for electric cars and to talk to manufacturers. He "
        "warned that it could take several years before the technology reaches shops."
    ),
}


def box(text):
    return (f"<div style='font-size:1.05rem; line-height:1.7; padding:1rem; border-radius:0.5rem; "
            f"border:1px solid rgba(128,128,128,0.4);'>{html.escape(text)}</div>")


# ---------- sidebar ----------
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Decoding method", ["Greedy", "Beam search"], index=0,
                    help="In our tests beam search was not better than greedy for this model.")
    beam = 1
    if mode == "Beam search":
        beam = st.slider("Beam width", 2, 6, 4)
    min_len = st.slider("Minimum summary length (tokens)", 10, 60, 25)
    block = st.checkbox("Block repeated phrases", value=True)

    st.divider()
    st.subheader("About the model")
    i = sm.info
    st.write(f"Transformer encoder-decoder, **{sm.n_params / 1e6:.1f}M** parameters, built and trained "
             "**from scratch** (no pretrained weights) on CNN/DailyMail articles.")
    if i:
        st.write(f"- Trained on {i.get('train_articles', 0):,} articles for {i.get('epochs')} epochs")
        st.write(f"- ROUGE-1 / 2 / L: **{i.get('rouge1')} / {i.get('rouge2')} / {i.get('rougeL')}**")
        st.write(f"- Lead-3 baseline: {i.get('lead3_rouge1')} / {i.get('lead3_rouge2')} / {i.get('lead3_rougeL')}")
        st.caption(f"Measured on {i.get('eval_articles'):,} test articles. Lead-3 (the first three sentences) "
                   "scores higher than our model.")
    st.write(f"- Running on: `{sm.device}`")

# ---------- main page ----------
st.title("News Article Summarizer")
st.caption("A small Transformer built from scratch in PyTorch and trained on CNN/DailyMail news articles.")
st.warning("This is a learning project. The model can **misspell names, get numbers wrong and invent facts**, "
           "including false statements about real people. Do not rely on its output.")

choice = st.selectbox("Try an example", ["(paste your own)"] + list(EXAMPLES))
default_text = "" if choice == "(paste your own)" else EXAMPLES[choice]
text = st.text_area("Article text (English news works best)", value=default_text, height=280, max_chars=20000,
                    placeholder="Paste a news article here...")

if st.button("Summarize", type="primary"):
    if len(text.split()) < 40:
        st.warning("Please paste a longer article (at least 40 words).")
    else:
        t0 = time.time()
        with st.spinner("Summarizing..."):
            res = sm.summarize(text, beam=beam, no_repeat=3 if block else 0, min_len=min_len)
        elapsed = time.time() - t0

        left, right = st.columns(2)
        with left:
            st.subheader("Model summary")
            st.markdown(box(res["summary"]), unsafe_allow_html=True)
            st.caption(f"{len(res['summary'].split())} words, {elapsed:.1f}s, {mode.lower()}")
        with right:
            st.subheader("Lead-3 baseline")
            st.markdown(box(res["lead3"]), unsafe_allow_html=True)
            st.caption("The first three sentences of the article. No AI involved.")

        if res["truncated"]:
            st.info(f"The model reads only the first {res['words_used']} words of this "
                    f"{res['words_total']}-word article (its input limit is {sm.max_src} tokens).")

with st.expander("Why can the model summary be wrong?"):
    st.markdown(
        "- The model was trained from scratch on about 88,000 articles, which is small for this task.\n"
        "- It cannot copy words directly from the article, so names and numbers are rebuilt piece by piece "
        "and are often garbled.\n"
        "- It writes fluent sentences even when it is unsure, so wrong details can sound convincing.\n"
        "- It only sees the first 400 tokens (about 300 words) of the article.\n"
        "- It was trained mostly on Daily Mail style articles."
    )
