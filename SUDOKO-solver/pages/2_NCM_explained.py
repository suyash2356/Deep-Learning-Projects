import streamlit as st

st.set_page_config(page_title="NCM explained", layout="wide")

st.title("NCM explained: Next-Concept Modeling")
st.markdown("**In one line:** instead of writing text word by word, NCM writes it in a few passes, "
            "from a rough sketch to fine detail, and each pass predicts many pieces **at once**.")
st.caption("A simplified summary of the paper. For exact details, read the paper (links at the bottom).")

# ------------------------------------------------------------------
st.header("1. Why this idea exists")
st.markdown("""
Normal LLMs (GPT-style) are **autoregressive**:
- They write **one token per forward pass**. A 1000-token answer needs 1000 passes.
- A token, once written, is never revised.
- So long answers are slow, and early mistakes pile up. You saw this in your own project: 
  the transformer got about 90% of cells right but only about 53% of grids fully right.

**Diffusion language models** try to fix this by predicting many tokens at once. 
The paper points out their catches: they need many refinement passes, every pass processes the full sequence, 
and they take a lot of training. NCM is a different route to parallel generation.
""")

# ------------------------------------------------------------------
st.header("2. The core idea: sketch first, details later")
st.markdown("Think of drawing: first a rough outline, then shapes, then fine details. NCM writes text the same way.")
st.code("""
Level 1   (very few codes)   : the overall idea
Level 2   (a few more)       : adds structure
Level 3   ...                : adds more detail
...
Level K   (as long as text)  : the exact words
                  |
                  v
     add up all levels -> decode -> final text

K is much smaller than the text length L  (K << L)
""", language=None)
st.markdown("""
- Each level is a short list of **codes**. A code is just an index into a small learned dictionary (a *codebook*).
- Early levels are **short** and cheap. The last level is as long as the text itself.
- Generation = **K forward passes**, not L. Inside each pass, all codes of that level are predicted in parallel.
""")

# ------------------------------------------------------------------
st.header("3. How it works: two stages")
st.subheader("Stage 1: the tokenizer (multiscale residual VQ-VAE)")
st.markdown("""
This part turns text into the levels of codes, and back. The long name breaks into simple pieces:
- **Encoder:** text becomes a list of vectors.
- **VQ (vector quantization):** each vector is snapped to the nearest entry in a codebook. 
  What you keep is just that entry's index, the *code*.
- **Residual:** after level 1, look at **what is still missing** (the residual). Level 2 encodes that. 
  The reconstruction is the **sum** of all levels so far.
- **Multiscale:** early levels have few codes, later ones more. Level lengths grow until the last one equals the text length.
- **Nested dropout (training trick):** sometimes train with only the first few levels, so the early levels 
  are forced to carry real meaning.
- **Decoder:** turns the final sum back into tokens.

The paper reports very high exact reconstruction (about 99.8-99.99% on most tasks, 100% on Sudoku).
""")
with st.expander("The one formula (optional)"):
    st.markdown("""Level 1 starts from zero. Each new level only has to fix what the earlier levels got wrong.
""")

st.subheader("Stage 2: the generator (a normal transformer)")
st.markdown("""
- The tokenizer is **frozen**. Now a transformer learns to predict the codes of level k, 
  given the reconstruction built from levels 1 to k-1.
- **All codes inside one level are predicted in parallel**, treated as independent of each other 
  once the earlier levels are known.
- Generation: run K passes (coarse to fine), then decode the final reconstruction into text.
- **Trick, conditioning corruption:** during training, some input codes are replaced with random ones from the same level. 
  At generation time the model sees its *own* imperfect codes, so this teaches it to survive its own mistakes.
- **Attention:** each level's block only attends within itself, since its input already summarizes the earlier levels. 
  The paper's test found this faster and better than the alternatives they tried.
""")

# ------------------------------------------------------------------
st.header("4. What the levels look like")
st.markdown("""
The paper shows one German-to-English translation decoded after every level:
- After level 1 the sentence is nearly meaningless (BLEU score about 6).
- By level 4 the rough structure is there (about 16).
- By level 11, the last one, it closely matches the reference (about 78).

Quality is **not perfectly smooth**: one level scored about 52, the next dropped to about 39, 
so later levels can fix things but can also disturb them.

The paper's probing experiments show the coarse levels carry the **gist**:
- Fix the first few levels and resample the rest: same theme, different names and wording.
- Mix the coarse levels of two stories: the result blends both.
- Change just two early codes: characters and events can change.
""")

# ------------------------------------------------------------------
st.header("5. Real NCM vs the NCM-lite in this project")
st.markdown("""
| | Real NCM (paper) | NCM-lite (this app) |
|---|---|---|
| Hierarchy | **Learned** by a VQ-VAE | **Hand-made**: 3 fixed groups of cells |
| What a level holds | Codes that decode into an estimate of the **whole board** | Actual digits for 12 cells |
| Can a later level fix an earlier mistake? | **Yes**, each level refines the full board | **No**, a written digit is final |
| Sudoku size | 9x9 | 6x6 |
| Model size | about 25M parameters | about 0.8M parameters |
| Passes | 12 | 3 |

NCM-lite keeps the core idea (**coarse to fine, many cells in parallel**) but not the learned codes 
or the self-correction. It teaches the concept. It does not reproduce the paper.
""")

# ------------------------------------------------------------------
st.header("6. What the paper reports")
st.markdown("""
| Task | Result (the paper's own numbers) |
|---|---|
| LM1B text generation | Better quality than the strongest diffusion baseline, about **21x faster** with about 11x fewer forward passes |
| OpenWebText | Competitive with the strongest diffusion model, about **10x faster** |
| XSum summarization | On par or better ROUGE, about **31x faster** than the strongest diffusion baseline |
| WMT14 German to English | Second best among the non-autoregressive methods compared; uses about 5.8x fewer passes than the best one |
| Training cost | About **4-5x fewer training tokens** than strong diffusion baselines (LM1B, OpenWebText) |
""")
st.markdown("**Sudoku (9x9, % of boards fully solved):**")
st.markdown("""
| Method | Forward passes | Easy | Medium | Hard |
|---|---|---|---|---|
| Autoregressive (greedy) | 128 | 14.6 | 5.1 | 1.0 |
| Best diffusion/flow baseline | 16 | 97.7 | 92.8 | 81.3 |
| **NCM** | **12** | **99.5** | **96.5** | **86.1** |
""")
st.caption("Easy/medium/hard = 40/35/30 clues. Baseline numbers are taken by the authors from another paper.")
st.warning("Read these with care: models are about 100M parameters (up to 725M in one scaling test), "
           "most comparisons are against diffusion language models, speed was measured on one GPU, "
           "and this is a preprint reporting the authors' own results.")

# ------------------------------------------------------------------
st.header("7. Other applications")
st.markdown("""
**Shown in the paper:**
- **Summarization** and **translation:** the input is encoded as a prefix, and the output is generated coarse to fine.
- **Unconditional generation:** stories, news-style sentences, web text.
- **Reasoning with constraints (Sudoku):** every cell must agree with the others, which suits refining the whole output across levels.
- **Test-time scaling:** because generation is cheap, you can sample many candidates and keep the one most similar 
  to the rest (*minimum Bayes risk* decoding). The paper reports about 7% higher ROUGE-L than the strongest summarization 
  baseline for the same time budget.
- **Steering at the idea level:** the probing experiments above hint at controlling text through its coarse levels.

**Same recipe elsewhere:** the next-scale idea comes from image generation (VAR), and residual quantization is used in neural audio codecs.
""")

# ------------------------------------------------------------------
st.header("8. Limitations")
st.markdown("""
**Listed by the paper:**
- The number of levels K and the length of each level are **fixed**, so even a short output goes through every level.
- The tokenizer is **trained separately**, not together with the generator.
- Tested mostly at about **100M parameters** (largest 725M, on one dataset).
- Occasional artifacts where a coarse level hands over to the next.

**Also visible in the paper:**
- The tokenizer sometimes rebuilds names, numbers or accents wrongly. It is rare (about 0.01% of tokens on web text), 
  but one wrong token can matter.
- Errors can pile up across levels, which is why the corruption trick is needed.
- Codes inside a level are assumed independent given the earlier levels.

**My own caution (not from the paper):**
- It is not shown as a chat or instruction-following model, or at frontier-model scale.
- Autoregressive models have years of tooling (caching, serving, fine-tuning) that NCM does not yet have.
- Our toy results here **cannot** be extrapolated to real language models.
""")

# ------------------------------------------------------------------
st.header("9. Future scope")
st.markdown("""
**Directions the authors name:**
- An **input-dependent schedule**: fewer levels for easy inputs, more for hard ones.
- **End-to-end training** of the tokenizer and generator together.
- **Scaling up** to larger datasets and models.
- Smoother hand-over between coarse levels.

**Ideas that are my speculation, not claims from the paper:**
- Faster long outputs where latency matters.
- Spending the saved time on more candidates and picking the best.
- Editing or steering text by changing only its coarse codes.
""")

# ------------------------------------------------------------------
st.header("10. How it compares with related ideas")
st.markdown("""
| Approach | What it produces per step | Number of steps |
|---|---|---|
| Autoregressive LLM | 1 token | L (text length) |
| Diffusion language model | Refines all positions again and again | Many |
| Block diffusion | One block of tokens, blocks in order | Number of blocks |
| Meta's Large Concept Model | One sentence representation | One per sentence |
| **NCM** | One level of codes (coarse to fine) | K, much smaller than L |

The word "concept" means different things: in Meta's model it is a **sentence**; 
in NCM it is the information carried by the **coarse levels**.
""")

# ------------------------------------------------------------------
with st.expander("Glossary for beginners"):
    st.markdown("""
- **Token:** a piece of text (word or part of a word) the model reads and writes.
- **Autoregressive (AR):** generating one token at a time, each depending on the earlier ones.
- **Forward pass / NFE:** one run of the model. NFE = number of function evaluations = number of passes.
- **Codebook:** a small learned dictionary of vectors. A *code* is the index of an entry.
- **VQ-VAE:** a model that compresses data into codes and rebuilds it from them.
- **Residual:** what is still missing after the earlier levels.
- **Parallel generation:** predicting many outputs in the same forward pass.
- **Diffusion LM:** a model that starts from a rough or masked text and refines it repeatedly.
- **MBR decoding:** generate several candidates, keep the one most similar to the others.
""")

# ------------------------------------------------------------------
st.header("11. The research paper")
st.markdown("""
**Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation**  
Hermann Kumbong, Ishan S. Khare, Iddah Mlauzi, Andrew Woen, Charles Ding, E. Kelly Buchanan, 
Simran Arora, Azalia Mirhoseini, Christopher Ré. Stanford University and Together AI.  
arXiv preprint 2610.14162 (October 2026)

- Paper: https://arxiv.org/abs/2610.14162
- Code (authors): https://github.com/HazyResearch/ncm

**Related:** *Large Concept Models: Language Modeling in a Sentence Representation Space* (Meta, arXiv 2412.08821), 
and *Visual Autoregressive Modeling* (next-scale prediction for images).
""")
st.caption("This demo is an independent learning project. It does not use the authors' code and is not affiliated with them.")