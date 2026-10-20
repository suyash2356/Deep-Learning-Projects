import html
import time
import streamlit as st
import streamlit.components.v1 as components
from translator import Translator
from xray import build_trace, render_html

st.set_page_config(page_title="English to Hindi Translator", page_icon="🌐", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load_translator():
    return Translator()


@st.cache_data(show_spinner="Recording what happens inside the model...")
def xray_page(sentence):
    trace = build_trace(load_translator(), sentence)
    return render_html(trace), trace["translation"]


tr = load_translator()

EXAMPLES = [
    "How are you?",
    "India is a very big country.",
    "The government announced a new education policy.",
    "Students should study hard to achieve success.",
    "The President addressed the nation on Independence Day.",
    "Water is essential for life. The weather is very nice today.",
]
XRAY_EXAMPLES = [
    "The government announced a new education policy.",
    "India is a very big country.",
    "Water is essential for life.",
    "Students should study hard to achieve success.",
    "How are you?",
]

# ---------- sidebar ----------
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Decoding method", ["Beam search", "Greedy"], index=0,
                    help="Beam search is slower but usually a bit more accurate.")
    beam_width, alpha = 4, 0.6
    if mode == "Beam search":
        beam_width = st.slider("Beam width", 2, 8, 4)
        alpha = st.slider("Length penalty (alpha)", 0.0, 1.5, 0.6, 0.1)

    st.divider()
    st.subheader("About the model")
    info = tr.info
    st.write(f"Transformer encoder-decoder, **{tr.n_params / 1e6:.1f}M** parameters, "
             "built and trained **from scratch** (no pretrained weights).")
    if info:
        st.write(f"- Trained on {info.get('train_pairs', 0):,} IITB sentence pairs")
        st.write(f"- Test BLEU: **{info.get('test_bleu')}**, chrF: **{info.get('test_chrf')}**")
    st.write(f"- Running on: `{tr.device}`")

# ---------- header ----------
st.title("English → Hindi Translator")
st.caption("A small Transformer built from scratch in PyTorch, trained on the IITB English-Hindi corpus.")

tab_tr, tab_xr = st.tabs(["Translate", "Model X-ray"])

# ---------- tab 1: translate ----------
with tab_tr:
    _, centre, _ = st.columns([1, 2.4, 1])
    with centre:
        example = st.selectbox("Try an example", ["(type your own)"] + EXAMPLES)
        default_text = "" if example == "(type your own)" else example

        text = st.text_area("English text", value=default_text, height=150, max_chars=1500,
                            placeholder="Type or paste English text here...")

        if st.button("Translate", type="primary"):
            if not text.strip():
                st.warning("Please enter some English text first.")
            else:
                t0 = time.time()
                with st.spinner("Translating..."):
                    result = tr.translate(text,
                                          mode="beam" if mode == "Beam search" else "greedy",
                                          beam=beam_width, alpha=alpha)
                elapsed = time.time() - t0

                st.subheader("Hindi translation")
                safe = html.escape(result["translation"])
                st.markdown(
                    f"<div style='font-size:1.4rem; line-height:1.8; padding:1rem; "
                    f"border-radius:0.5rem; border:1px solid rgba(128,128,128,0.4);'>{safe}</div>",
                    unsafe_allow_html=True,
                )
                st.caption(f"{len(result['pairs'])} sentence(s) translated in {elapsed:.2f}s using {mode.lower()}")

                if result["truncated"]:
                    st.warning("A very long sentence was cut to fit the model's length limit. "
                               "Try splitting it into shorter sentences.")

                if len(result["pairs"]) > 1:
                    with st.expander("Sentence by sentence"):
                        for en, hi in result["pairs"]:
                            st.write(f"**EN:** {en}")
                            st.write(f"**HI:** {hi}")
                            st.divider()

        st.divider()
        st.caption("This is a learning project. It works best on short, everyday sentences. "
                   "Rare words, names and idioms may be translated incorrectly.")

# ---------- tab 2: model x-ray ----------
with tab_xr:
    st.markdown("Watch one sentence travel through the Transformer: words become embeddings, "
                "attention connects the words, neurons light up layer by layer, and the Hindi "
                "sentence is written one word at a time.")

    ex = st.selectbox("Pick an example", ["(type your own)"] + XRAY_EXAMPLES, key="xr_example")
    default_x = "" if ex == "(type your own)" else ex
    xr_text = st.text_input("English sentence (short sentences work best)", value=default_x, max_chars=200)

    if st.button("Show X-ray", type="primary"):
        if not xr_text.strip():
            st.warning("Please enter a sentence first.")
        else:
            st.session_state["xray_sentence"] = xr_text.strip()

    sentence = st.session_state.get("xray_sentence")
    if sentence:
        parts = tr.split_sentences(sentence)
        if len(parts) > 1:
            sentence = st.selectbox("Your text has several sentences. Which one should we trace?", parts)

        if len(tr.sp.encode(sentence)) > 14:
            st.warning("The X-ray shows at most 14 tokens, so the end of this sentence is cut. "
                       "Use a shorter sentence for a clean picture.")

        page, translation = xray_page(sentence)
        st.markdown(f"**Model output (greedy decoding, one word at a time):** {translation}")
        components.html(page, height=880, scrolling=True)

        with st.expander("How to read the animation"):
            st.markdown(
                "- **Yellow / blue glowing dots** are neurons: yellow is positive, blue is negative, brighter is stronger. "
                "Each word's 256 numbers are grouped into 8 dots.\n"
                "- **Gold lines in the encoder** are attention: a thicker line means that word looks more at the other word.\n"
                "- **Click an English word** during the encoder part to follow only its path.\n"
                "- **Decoder:** blue lines look back at the Hindi words written so far, green lines look at the English words. "
                "At the bottom, the 5 candidate words glow by probability and the winner is written at the top.\n"
                "- Use the **attention** dropdown to look at one head instead of the average.\n"
                "- Thin lines between dots inside one word are an illustration; the glow of the dots and the attention lines are real values from the model."
            )
