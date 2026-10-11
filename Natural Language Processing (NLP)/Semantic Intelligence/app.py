"""
Semantic News Intelligence Engine
=================================
GloVe (50D) + mean document embeddings + Linear SVM, wrapped in a Streamlit app.

Features
    1. Overview            - what the system does and how to use it
    2. Semantic Search     - live, meaning-based search over the AG News corpus
    3. Article Intelligence- topic classification with confidence, coverage and similar articles
    4. Word Lab            - nearest neighbours and pairwise word similarity
    5. Analogy Lab         - vector arithmetic (A is to B as C is to ?)
    6. Embedding Explorer  - interactive 2D / 3D PCA projection
    7. Model Insights      - live test-set evaluation, confusion matrix, limitations

Project structure
    app.py
    requirements.txt
    artifacts/  glove_50d_matrix.npy, glove_vocab.pkl, train_mean_embeddings.npy,
                test_mean_embeddings.npy, final_classifier.pkl, [project_metadata.pkl]
    data/       train.jsonl, test.jsonl

Run
    pip install -r requirements.txt
    python -m streamlit run app.py
"""

from __future__ import annotations

import html
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "Semantic News Intelligence"
GITHUB_URL = ""  # e.g. "https://github.com/your-name/semantic-news-engine"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"

REQUIRED_FILES = [
    ARTIFACTS_DIR / "glove_50d_matrix.npy",
    ARTIFACTS_DIR / "glove_vocab.pkl",
    ARTIFACTS_DIR / "train_mean_embeddings.npy",
    ARTIFACTS_DIR / "test_mean_embeddings.npy",
    ARTIFACTS_DIR / "final_classifier.pkl",
    DATA_DIR / "train.jsonl",
    DATA_DIR / "test.jsonl",
]

LABEL_MAP = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
CATEGORIES = list(LABEL_MAP.values())
CATEGORY_ICONS = {"World": "🌍", "Sports": "⚽", "Business": "📈", "Sci/Tech": "💻"}
CATEGORY_COLORS = {
    "World": "#3B82C4",
    "Sports": "#3FA66B",
    "Business": "#E9A23B",
    "Sci/Tech": "#8B6BC9",
}

WORD_GROUPS = {
    "Royalty": ["king", "queen", "prince", "princess", "monarch", "throne"],
    "Technology": ["computer", "software", "internet", "digital", "technology", "chip"],
    "Business": ["company", "market", "stock", "investors", "economy", "profit"],
    "Sports": ["football", "soccer", "basketball", "tennis", "olympics", "coach"],
    "Geography": ["paris", "france", "london", "england", "tokyo", "japan"],
}

SEARCH_EXAMPLES = {
    "Tech": "artificial intelligence computer processors",
    "Sports": "football championship team player",
    "Business": "stock market investors company economy",
    "World": "government international country leaders",
}

ARTICLE_EXAMPLES = {
    "Tech": (
        "A technology company announced a new artificial intelligence processor "
        "designed to improve machine learning performance in data centers."
    ),
    "Sports": (
        "The football team reached the championship final after defeating its "
        "opponent in a dramatic match."
    ),
    "Business": (
        "Investors pushed technology stocks higher as the company reported "
        "stronger quarterly earnings and higher revenue."
    ),
    "World": (
        "Government leaders met to discuss international relations and a new "
        "agreement between several countries."
    ),
}

ANALOGY_EXAMPLES = {
    "man → woman, king": ("man", "woman", "king"),
    "france → paris, italy": ("france", "paris", "italy"),
    "big → bigger, small": ("big", "bigger", "small"),
}


# ============================================================
# STYLING
# ============================================================
# Neutral, theme-aware styling: cards use translucent fills and inherited
# text colour so the app reads well in both light and dark Streamlit themes.

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stMarkdown, .stTextInput, .stTextArea, .stButton {
        font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
    }
    .block-container { max-width: 1280px; padding-top: 1.6rem; padding-bottom: 3rem; }

    h1, h2, h3, .hero h1, .serif {
        font-family: 'Newsreader', Georgia, 'Times New Roman', serif !important;
        letter-spacing: -0.01em;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: #0F172A; }
    section[data-testid="stSidebar"] * { color: #E2E8F0; }
    section[data-testid="stSidebar"] hr { border-color: #1E293B; }

    /* Hero */
    .hero {
        background: #0F172A;
        border-radius: 16px;
        padding: 2.2rem 2.4rem;
        margin-bottom: 1.4rem;
        border-left: 6px solid #E9A23B;
    }
    .hero h1 { color: #F8FAFC; font-size: 2.6rem; font-weight: 700; margin: 0 0 .5rem 0; }
    .hero p  { color: #CBD5E1; font-size: 1.04rem; line-height: 1.7; max-width: 780px; margin: 0; }
    .chip {
        display: inline-block; padding: .22rem .7rem; margin: 0 .35rem .8rem 0;
        border-radius: 999px; font-size: .78rem; font-weight: 500;
        color: #E2E8F0; border: 1px solid #334155;
    }

    /* Cards */
    .card {
        border: 1px solid rgba(128,128,128,.28);
        background: rgba(128,128,128,.07);
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        margin-bottom: .85rem;
    }
    .card h4 { margin: 0 0 .35rem 0; font-size: 1.05rem; }
    .card p  { margin: 0; line-height: 1.6; opacity: .85; font-size: .93rem; }

    .metric { border: 1px solid rgba(128,128,128,.28); border-radius: 12px; padding: 1rem 1.2rem; }
    .metric .label { font-size: .82rem; opacity: .7; }
    .metric .value { font-family: 'Newsreader', Georgia, serif; font-size: 2rem; font-weight: 700; line-height: 1.2; }
    .metric .desc  { font-size: .8rem; opacity: .65; }

    /* Search results */
    .result {
        border: 1px solid rgba(128,128,128,.28);
        border-left-width: 5px;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: .8rem;
        background: rgba(128,128,128,.05);
    }
    .result .title { font-family: 'Newsreader', Georgia, serif; font-size: 1.15rem; font-weight: 700; margin-bottom: .35rem; }
    .result .body  { font-size: .92rem; line-height: 1.6; opacity: .85; margin-top: .5rem; }
    .tag {
        display: inline-block; padding: .15rem .6rem; margin-right: .4rem;
        border-radius: 999px; font-size: .76rem; font-weight: 600;
        border: 1px solid rgba(128,128,128,.4);
    }
    .bar { height: 5px; border-radius: 3px; background: rgba(128,128,128,.2); margin-top: .55rem; }
    .bar > div { height: 100%; border-radius: 3px; }

    .verdict { font-family: 'Newsreader', Georgia, serif; font-size: 2.2rem; font-weight: 700; }
    .footer { margin-top: 3rem; padding-top: 1.2rem; border-top: 1px solid rgba(128,128,128,.3);
              text-align: center; font-size: .82rem; opacity: .65; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SMALL UI HELPERS
# ============================================================

def stretch(fn, *args, **kwargs):
    """Call a Streamlit widget full-width on both old and new Streamlit versions."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


def show_df(df: pd.DataFrame, **kwargs):
    return stretch(st.dataframe, df, hide_index=True, **kwargs)


def show_plot(fig: go.Figure):
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return stretch(st.plotly_chart, fig)


def set_state(key: str, value) -> None:
    """Button callback: safely update a widget's value (must run before the widget renders)."""
    st.session_state[key] = value


def esc(value) -> str:
    """Escape untrusted dataset/user text before putting it inside HTML."""
    return html.escape(str(value), quote=True)


def metric_card(label: str, value: str, desc: str = "") -> str:
    return (
        f'<div class="metric"><div class="label">{esc(label)}</div>'
        f'<div class="value">{esc(value)}</div><div class="desc">{esc(desc)}</div></div>'
    )


def info_card(title: str, body: str) -> str:
    return f'<div class="card"><h4>{esc(title)}</h4><p>{esc(body)}</p></div>'


# ============================================================
# RESOURCE LOADING
# ============================================================

def load_jsonl(path: Path) -> pd.DataFrame:
    """Read a JSON Lines file, reporting the exact line on malformed input."""
    records = []
    with open(path, "r", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path.name}, line {number}: {error}") from error
    if not records:
        raise ValueError(f"No records found in {path.name}")
    return pd.DataFrame(records)


def row_text(df: pd.DataFrame) -> pd.Series:
    """Title + description for display/search, tolerant of missing columns."""
    title = df["title"] if "title" in df else pd.Series("", index=df.index)
    desc = df["description"] if "description" in df else pd.Series("", index=df.index)
    if "title" not in df and "description" not in df and "text" in df:
        return df["text"].astype(str)
    return (title.fillna("").astype(str) + ". " + desc.fillna("").astype(str)).str.strip(". ")


@st.cache_resource(show_spinner="Loading semantic engine...")
def load_resources() -> dict:
    matrix = np.load(ARTIFACTS_DIR / "glove_50d_matrix.npy").astype(np.float32)
    with open(ARTIFACTS_DIR / "glove_vocab.pkl", "rb") as f:
        words = list(pickle.load(f))
    train_emb = np.load(ARTIFACTS_DIR / "train_mean_embeddings.npy").astype(np.float32)
    test_emb = np.load(ARTIFACTS_DIR / "test_mean_embeddings.npy").astype(np.float32)
    with open(ARTIFACTS_DIR / "final_classifier.pkl", "rb") as f:
        classifier = pickle.load(f)

    metadata = {}
    meta_path = ARTIFACTS_DIR / "project_metadata.pkl"
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

    train_df = load_jsonl(DATA_DIR / "train.jsonl")
    test_df = load_jsonl(DATA_DIR / "test.jsonl")

    # Some AG News exports use labels 0-3 instead of 1-4.
    offset = 1 if int(train_df["label"].min()) == 0 else 0

    def names(series: pd.Series) -> np.ndarray:
        return np.array([LABEL_MAP.get(int(v) + offset, str(v)) for v in series])

    def unit(a: np.ndarray) -> np.ndarray:
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)

    # Were the classifier's training vectors unit-length? If so, inputs must be too.
    sample_norms = np.linalg.norm(train_emb[:2000], axis=1)
    trained_on_unit_vectors = bool(np.allclose(sample_norms, 1.0, atol=1e-3))

    return {
        "matrix": matrix,
        "words": words,
        "word_to_index": {w: i for i, w in enumerate(words)},
        "norm_glove": unit(matrix),
        "train_df": train_df,
        "test_df": test_df,
        "train_texts": row_text(train_df).to_numpy(),
        "train_categories": names(train_df["label"]),
        "test_categories": names(test_df["label"]),
        "norm_train": unit(train_emb),
        "test_emb": test_emb,
        "classifier": classifier,
        "metadata": metadata,
        "label_offset": offset,
        "unit_inputs": trained_on_unit_vectors,
    }


# ---- missing-file guard -------------------------------------------------
missing = [p for p in REQUIRED_FILES if not p.exists()]
if missing:
    st.error("Some required project files are missing.")
    for path in missing:
        st.code(str(path))
    st.info(
        "Place `app.py` in the project root with the `artifacts/` and `data/` "
        "folders beside it, then rerun the app."
    )
    st.stop()

try:
    R = load_resources()
except Exception as error:  # noqa: BLE001 - show any loading problem to the user
    st.error("The application could not load the project resources.")
    st.exception(error)
    st.stop()

words_list = R["words"]
word_to_index = R["word_to_index"]
matrix = R["matrix"]
norm_glove = R["norm_glove"]
classifier = R["classifier"]
train_df = R["train_df"]
test_df = R["test_df"]


# ============================================================
# CORE NLP FUNCTIONS
# ============================================================

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class TextEmbedding:
    unit: np.ndarray      # L2-normalised mean vector (used for similarity search)
    model_input: np.ndarray  # vector in the form the classifier was trained on
    matched: list[str]
    unknown: list[str]

    @property
    def coverage(self) -> float:
        total = len(self.matched) + len(self.unknown)
        return len(self.matched) / total if total else 0.0


def embed_text(text: str) -> TextEmbedding | None:
    """Mean GloVe embedding of a text; None if no word is in the vocabulary."""
    tokens = clean_text(text).split()
    matched = [t for t in tokens if t in word_to_index]
    unknown = [t for t in tokens if t not in word_to_index]
    if not matched:
        return None
    mean = matrix[[word_to_index[t] for t in matched]].mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm == 0:
        return None
    unit = (mean / norm).astype(np.float32)
    model_input = unit if R["unit_inputs"] else mean.astype(np.float32)
    return TextEmbedding(unit, model_input, matched, unknown)


def top_k(scores: np.ndarray, k: int, mask: np.ndarray | None = None) -> np.ndarray:
    """Indices of the k highest scores (descending), optionally restricted by a boolean mask."""
    idx = np.flatnonzero(mask) if mask is not None else np.arange(len(scores))
    if idx.size == 0:
        return idx
    k = min(k, idx.size)
    part = idx[np.argpartition(-scores[idx], k - 1)[:k]]
    return part[np.argsort(-scores[part])]


def semantic_search(query: str, k: int, category: str, min_sim: float):
    emb = embed_text(query)
    if emb is None:
        return None, None
    scores = R["norm_train"] @ emb.unit
    mask = scores >= min_sim
    if category != "All":
        mask &= R["train_categories"] == category
    idx = top_k(scores, k, mask)
    out = pd.DataFrame(
        {
            "rank": np.arange(1, len(idx) + 1),
            "category": R["train_categories"][idx],
            "similarity": scores[idx],
            "article": R["train_texts"][idx],
            "title": train_df["title"].to_numpy()[idx] if "title" in train_df else "",
            "description": (
                train_df["description"].to_numpy()[idx] if "description" in train_df else ""
            ),
        }
    )
    return out, emb


def nearest_words(word: str, n: int = 10, vocab_limit: int | None = None) -> list[tuple[str, float]]:
    word = word.lower().strip()
    if word not in word_to_index:
        return []
    space = norm_glove if vocab_limit is None else norm_glove[:vocab_limit]
    scores = space @ norm_glove[word_to_index[word]]
    idx = top_k(scores, n + 1)
    return [(words_list[i], float(scores[i])) for i in idx if words_list[i] != word][:n]


def word_similarity(a: str, b: str) -> float | None:
    a, b = a.lower().strip(), b.lower().strip()
    if a not in word_to_index or b not in word_to_index:
        return None
    return float(norm_glove[word_to_index[a]] @ norm_glove[word_to_index[b]])


def solve_analogy(a: str, b: str, c: str, n: int = 10, vocab_limit: int = 50_000):
    """A is to B as C is to ?   ->  B - A + C."""
    a, b, c = (w.lower().strip() for w in (a, b, c))
    missing_words = [w for w in (a, b, c) if w not in word_to_index]
    if missing_words:
        return [], missing_words
    target = matrix[word_to_index[b]] - matrix[word_to_index[a]] + matrix[word_to_index[c]]
    norm = np.linalg.norm(target)
    if norm == 0:
        return [], []
    space = norm_glove[:vocab_limit]
    scores = space @ (target / norm)
    exclude = {a, b, c}
    idx = top_k(scores, n + 3)
    return [(words_list[i], float(scores[i])) for i in idx if words_list[i] not in exclude][:n], []


def classify(text: str):
    """Returns (category, TextEmbedding, scores_df) or (None, None, None)."""
    emb = embed_text(text)
    if emb is None:
        return None, None, None
    x = emb.model_input.reshape(1, -1)
    pred = int(classifier.predict(x)[0])
    category = LABEL_MAP.get(pred + R["label_offset"], str(pred))

    scores_df = None
    if hasattr(classifier, "decision_function"):
        raw = np.atleast_1d(classifier.decision_function(x)).ravel()
        exp = np.exp(raw - raw.max())
        scores_df = pd.DataFrame(
            {
                "Category": [LABEL_MAP.get(int(c) + R["label_offset"], str(c)) for c in classifier.classes_],
                "Decision score": raw,
                "Relative confidence": exp / exp.sum(),
            }
        ).sort_values("Decision score", ascending=False, ignore_index=True)
    return category, emb, scores_df


@st.cache_data(show_spinner=False)
def project_words(word_tuple: tuple[str, ...], dims: int):
    vectors = np.array([matrix[word_to_index[w]] for w in word_tuple])
    pca = PCA(n_components=dims)
    return pca.fit_transform(vectors), pca.explained_variance_ratio_


@st.cache_data(show_spinner="Evaluating on the test split...")
def evaluate_model():
    """Live evaluation on the held-out test split (replaces hard-coded numbers)."""
    try:
        y_true = R["test_categories"]
        y_pred_raw = classifier.predict(R["test_emb"])
        y_pred = np.array([LABEL_MAP.get(int(v) + R["label_offset"], str(v)) for v in y_pred_raw])
        report = classification_report(y_true, y_pred, labels=CATEGORIES, output_dict=True, zero_division=0)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "matrix": confusion_matrix(y_true, y_pred, labels=CATEGORIES),
            "report": report,
        }
    except Exception:  # noqa: BLE001 - shapes/labels may not match; the UI degrades gracefully
        return None


evaluation = evaluate_model()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🧠 Semantic Intelligence")
    st.caption("Search, classify and explore news with word embeddings. No deep learning required.")
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown(
        "1. Every word becomes a 50-number GloVe vector.\n"
        "2. An article is the average of its word vectors.\n"
        "3. Similar meaning means vectors that point the same way.\n"
        "4. A Linear SVM maps the vector to a topic."
    )
    st.markdown("---")
    st.markdown(
        f"**Articles indexed:** {len(train_df):,}  \n"
        f"**Vocabulary:** {len(words_list):,} words  \n"
        f"**Vector size:** {matrix.shape[1]}D"
    )
    if GITHUB_URL:
        st.markdown(f"[View source on GitHub]({GITHUB_URL})")
    st.markdown("---")
    st.caption("Data: AG News. The app searches a fixed dataset, not live news.")


# ============================================================
# HERO + METRICS
# ============================================================

st.markdown(
    """
    <div class="hero">
      <span class="chip">GloVe 50D</span><span class="chip">Semantic search</span>
      <span class="chip">Linear SVM</span><span class="chip">PCA</span>
      <h1>Find the story by its meaning, not its keywords</h1>
      <p>Search 120,000 news articles by idea, classify any article into a topic,
      and see how a machine represents language as geometry.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

acc_text = f"{evaluation['accuracy'] * 100:.2f}%" if evaluation else "n/a"
f1_text = f"{evaluation['macro_f1'] * 100:.2f}%" if evaluation else "n/a"
cols = st.columns(4)
for col, (label, value, desc) in zip(
    cols,
    [
        ("Indexed articles", f"{len(train_df):,}", "Searchable corpus"),
        ("Vocabulary", f"{len(words_list):,}", "Pretrained GloVe words"),
        ("Test accuracy", acc_text, "Held-out AG News split"),
        ("Macro F1", f1_text, "Averaged over 4 topics"),
    ],
):
    col.markdown(metric_card(label, value, desc), unsafe_allow_html=True)

st.write("")

(overview_tab, search_tab, article_tab, word_tab, analogy_tab, embed_tab, insights_tab) = st.tabs(
    [
        "🏠 Overview",
        "🔎 Semantic Search",
        "📰 Article Intelligence",
        "🔤 Word Lab",
        "🔄 Analogy Lab",
        "🗺️ Embedding Explorer",
        "📊 Model Insights",
    ]
)


# ============================================================
# TAB 1 - OVERVIEW
# ============================================================

with overview_tab:
    st.header("From words to meaning")
    st.write(
        "Words that appear in similar contexts get similar vectors. Averaging those vectors "
        "gives every article a position in the same space, so articles about the same thing "
        "land near each other even when they share no words."
    )
    c1, c2, c3 = st.columns(3)
    c1.markdown(info_card("Word space", "GloVe gives each word a 50-dimensional vector. Related words sit close together."), unsafe_allow_html=True)
    c2.markdown(info_card("Document space", "An article is the average of its word vectors, giving one compact vector per article."), unsafe_allow_html=True)
    c3.markdown(info_card("Intelligence layer", "Cosine similarity powers search. A Linear SVM predicts the topic."), unsafe_allow_html=True)

    st.subheader("Try it")
    left, right = st.columns(2)
    left.markdown(
        "**🔎 Semantic Search** finds articles by meaning.  \n"
        "**📰 Article Intelligence** predicts the topic of any text.  \n"
        "**🔤 Word Lab** shows which words sit near a word."
    )
    right.markdown(
        "**🔄 Analogy Lab** solves `woman − man + king`.  \n"
        "**🗺️ Embedding Explorer** plots word vectors in 2D or 3D.  \n"
        "**📊 Model Insights** shows accuracy, errors and limits."
    )
    st.info("Start with Semantic Search, then look up the same words in Word Lab to see why the results came back.")


# ============================================================
# TAB 2 - SEMANTIC SEARCH
# ============================================================

with search_tab:
    st.header("Search by meaning")
    st.caption("Results update as you type. Press Enter to apply.")

    st.session_state.setdefault("search_query", "")
    st.text_input(
        "What are you looking for?",
        key="search_query",
        placeholder="companies developing artificial intelligence processors",
    )

    ex_cols = st.columns(len(SEARCH_EXAMPLES))
    for col, (name, query) in zip(ex_cols, SEARCH_EXAMPLES.items()):
        stretch(col.button, f"Try: {name}", key=f"ex_search_{name}", on_click=set_state, args=("search_query", query))

    f1, f2, f3 = st.columns([1.2, 1.2, 1])
    category_filter = f1.selectbox("Category", ["All", *CATEGORIES], key="search_category")
    result_count = f2.slider("Results", 3, 20, 8, key="search_k")
    min_similarity = f3.slider("Minimum similarity", 0.0, 1.0, 0.0, 0.05, key="search_min")

    query = st.session_state["search_query"].strip()
    if not query:
        st.info("Type a query or pick an example above to search the corpus.")
    else:
        results, q_emb = semantic_search(query, result_count, category_filter, min_similarity)
        if results is None:
            st.warning("None of the words in your query are in the GloVe vocabulary. Try different words.")
        else:
            if q_emb.unknown:
                st.caption("Ignored (not in vocabulary): " + ", ".join(sorted(set(q_emb.unknown))))
            if results.empty:
                st.warning("No articles matched. Try a broader query or lower the minimum similarity.")
            else:
                st.success(f"{len(results)} related articles found")
                for _, row in results.iterrows():
                    color = CATEGORY_COLORS.get(row["category"], "#888")
                    icon = CATEGORY_ICONS.get(row["category"], "📰")
                    body = esc(row["description"])
                    if len(body) > 320:
                        body = body[:320].rsplit(" ", 1)[0] + "…"
                    title = esc(row["title"]) if str(row["title"]).strip() else esc(str(row["article"])[:90])
                    pct = max(0.0, min(1.0, float(row["similarity"]))) * 100
                    st.markdown(
                        f'<div class="result" style="border-left-color:{color}">'
                        f'<div class="title">{title}</div>'
                        f'<span class="tag">{icon} {esc(row["category"])}</span>'
                        f'<span class="tag">Similarity {row["similarity"]:.3f}</span>'
                        f'<div class="bar"><div style="width:{pct:.0f}%;background:{color}"></div></div>'
                        f'<div class="body">{body}</div></div>',
                        unsafe_allow_html=True,
                    )
                export = results[["rank", "category", "similarity", "title", "description"]]
                st.download_button(
                    "Download results (CSV)",
                    export.to_csv(index=False).encode("utf-8"),
                    file_name="semantic_search_results.csv",
                    mime="text/csv",
                )


# ============================================================
# TAB 3 - ARTICLE INTELLIGENCE
# ============================================================

with article_tab:
    st.header("Article Intelligence")
    st.caption("Paste a headline or article. The text becomes a mean GloVe vector and the SVM predicts its topic.")

    st.session_state.setdefault("article_text", "")
    ex_cols = st.columns(len(ARTICLE_EXAMPLES))
    for col, (name, text) in zip(ex_cols, ARTICLE_EXAMPLES.items()):
        stretch(col.button, f"Example: {name}", key=f"ex_article_{name}", on_click=set_state, args=("article_text", text))

    st.text_area("Article text", key="article_text", height=200, placeholder="Paste a news article or short summary...")

    if stretch(st.button, "Analyze article", type="primary", key="analyze_article"):
        text = st.session_state["article_text"].strip()
        if not text:
            st.warning("Paste some article text first.")
        else:
            category, emb, scores_df = classify(text)
            if category is None:
                st.error("None of the words are in the GloVe vocabulary, so the text cannot be classified.")
            else:
                color = CATEGORY_COLORS.get(category, "#888")
                st.markdown(
                    f'<div class="card" style="border-left:5px solid {color}">'
                    f'<div style="opacity:.7;font-size:.85rem">Predicted topic</div>'
                    f'<div class="verdict">{CATEGORY_ICONS.get(category, "📰")} {esc(category)}</div></div>',
                    unsafe_allow_html=True,
                )

                m1, m2, m3 = st.columns(3)
                m1.metric("Words recognised", f"{len(emb.matched)} of {len(emb.matched) + len(emb.unknown)}")
                m2.metric("Vocabulary coverage", f"{emb.coverage * 100:.0f}%")
                m3.metric("Vector dimensions", len(emb.unit))
                if emb.coverage < 0.6:
                    st.warning("Low coverage: many words are outside the vocabulary, so this prediction is less reliable.")
                if emb.unknown:
                    with st.expander("Words that were ignored"):
                        st.write(", ".join(sorted(set(emb.unknown))))

                if scores_df is not None:
                    st.subheader("How the classifier scored each topic")
                    fig = px.bar(
                        scores_df.sort_values("Relative confidence"),
                        x="Relative confidence",
                        y="Category",
                        orientation="h",
                        color="Category",
                        color_discrete_map=CATEGORY_COLORS,
                        text=scores_df.sort_values("Relative confidence")["Relative confidence"].map("{:.0%}".format),
                    )
                    fig.update_layout(showlegend=False, xaxis_tickformat=".0%", height=280)
                    show_plot(fig)
                    st.caption(
                        "Relative confidence is a softmax over SVM decision scores. It shows how the four topics "
                        "compare, but it is not a calibrated probability."
                    )

                st.subheader("Most similar articles in the corpus")
                sims = R["norm_train"] @ emb.unit
                for i in top_k(sims, 5):
                    cat = R["train_categories"][i]
                    st.markdown(
                        f"**{CATEGORY_ICONS.get(cat, '📰')} {cat}** · similarity {sims[i]:.3f}  \n"
                        f"{R['train_texts'][i][:240]}"
                    )

                with st.expander("Inspect the document vector"):
                    fig = px.bar(x=[f"d{i + 1}" for i in range(len(emb.unit))], y=emb.unit,
                                 labels={"x": "Dimension", "y": "Value"})
                    fig.update_layout(height=280)
                    show_plot(fig)


# ============================================================
# TAB 4 - WORD LAB
# ============================================================

with word_tab:
    st.header("Word Lab")
    st.caption("See which words the model places closest to any word, and measure how close two words are.")

    c1, c2 = st.columns([1.5, 1])
    explorer_word = c1.text_input("Word", value="king", key="explorer_word")
    neighbor_count = c2.slider("Neighbours", 5, 20, 10, key="neighbor_count")

    word = explorer_word.lower().strip()
    if word:
        if word not in word_to_index:
            st.error(f"“{explorer_word}” is not in the GloVe vocabulary. Try a lowercase single word.")
        else:
            nn = pd.DataFrame(nearest_words(word, neighbor_count), columns=["Word", "Cosine similarity"])
            t_col, p_col = st.columns([1, 1.4])
            with t_col:
                show_df(nn)
            with p_col:
                fig = px.bar(nn.iloc[::-1], x="Cosine similarity", y="Word", orientation="h",
                             title=f"Closest words to “{word}”")
                fig.update_layout(height=380)
                show_plot(fig)

    st.divider()
    st.subheader("Compare two words")
    s1, s2 = st.columns(2)
    word_a = s1.text_input("First word", value="king", key="sim_a")
    word_b = s2.text_input("Second word", value="queen", key="sim_b")
    if word_a.strip() and word_b.strip():
        score = word_similarity(word_a, word_b)
        if score is None:
            st.error("Both words must be in the GloVe vocabulary.")
        else:
            st.metric("Cosine similarity", f"{score:.4f}")
            st.progress(float(max(0.0, min(1.0, score))))
            if score >= 0.8:
                st.info("Very close: these words are used in almost interchangeable contexts.")
            elif score >= 0.5:
                st.info("Related: these words appear in similar kinds of contexts.")
            elif score >= 0.2:
                st.info("Loosely related.")
            else:
                st.info("Distant: these words rarely share contexts.")


# ============================================================
# TAB 5 - ANALOGY LAB
# ============================================================

with analogy_tab:
    st.header("Analogy Lab")
    st.caption("Vector arithmetic captures relationships. The model computes B − A + C and finds the closest word.")

    for key, default in (("an_a", "man"), ("an_b", "woman"), ("an_c", "king")):
        st.session_state.setdefault(key, default)

    ex_cols = st.columns(len(ANALOGY_EXAMPLES))
    for col, (name, (a, b, c)) in zip(ex_cols, ANALOGY_EXAMPLES.items()):
        def _load(a=a, b=b, c=c):
            st.session_state.update(an_a=a, an_b=b, an_c=c)
        stretch(col.button, name, key=f"ex_an_{name}", on_click=_load)

    a_col, b_col, c_col = st.columns(3)
    an_a = a_col.text_input("A", key="an_a")
    an_b = b_col.text_input("B", key="an_b")
    an_c = c_col.text_input("C", key="an_c")
    pool = st.select_slider("Candidate pool (most frequent words)", [10_000, 20_000, 50_000, 100_000, len(words_list)], value=50_000,
                            help="Smaller pools avoid rare, noisy words.")

    if all(x.strip() for x in (an_a, an_b, an_c)):
        st.markdown(
            f'<div class="card"><h4>{esc(an_a)} is to {esc(an_b)} as {esc(an_c)} is to ?</h4>'
            f'<p>Target vector = {esc(an_b)} − {esc(an_a)} + {esc(an_c)}</p></div>',
            unsafe_allow_html=True,
        )
        found, absent = solve_analogy(an_a, an_b, an_c, 10, pool)
        if absent:
            st.error("Not in vocabulary: " + ", ".join(absent))
        elif not found:
            st.warning("No candidates found.")
        else:
            st.success(f"Best answer: **{found[0][0]}** (similarity {found[0][1]:.3f})")
            show_df(pd.DataFrame(found, columns=["Candidate", "Similarity"]))


# ============================================================
# TAB 6 - EMBEDDING EXPLORER
# ============================================================

with embed_tab:
    st.header("Embedding Explorer")
    st.caption("PCA squeezes 50 dimensions into 2 or 3 so you can see clusters. Some structure is always lost.")

    chosen_groups = st.multiselect("Word groups", list(WORD_GROUPS), default=list(WORD_GROUPS))
    custom_input = st.text_input("Add your own words (comma separated)", placeholder="ocean, river, mountain")
    dims = 3 if st.radio("View", ["2D", "3D"], horizontal=True) == "3D" else 2

    group_of: dict[str, str] = {}
    for group in chosen_groups:
        for w in WORD_GROUPS[group]:
            group_of.setdefault(w, group)
    for w in (x.strip().lower() for x in custom_input.split(",")):
        if w:
            group_of.setdefault(w, "Custom")

    unknown_words = [w for w in group_of if w not in word_to_index]
    valid = [w for w in group_of if w in word_to_index]
    if unknown_words:
        st.caption("Not in vocabulary: " + ", ".join(unknown_words))

    if len(valid) < max(3, dims):
        st.info("Pick at least three words to draw the projection.")
    else:
        coords, variance = project_words(tuple(valid), dims)
        plot_df = pd.DataFrame(coords, columns=[f"PC{i + 1}" for i in range(dims)])
        plot_df["Word"] = valid
        plot_df["Group"] = [group_of[w] for w in valid]
        if dims == 2:
            fig = px.scatter(plot_df, x="PC1", y="PC2", text="Word", color="Group")
            fig.update_traces(textposition="top center", marker=dict(size=11))
        else:
            fig = px.scatter_3d(plot_df, x="PC1", y="PC2", z="PC3", text="Word", color="Group")
            fig.update_traces(marker=dict(size=5))
        fig.update_layout(height=620, title="GloVe embedding space (PCA)")
        show_plot(fig)
        st.caption("Variance explained: " + " · ".join(f"PC{i + 1} {v * 100:.1f}%" for i, v in enumerate(variance))
                   + f" · total {variance.sum() * 100:.1f}%")


# ============================================================
# TAB 7 - MODEL INSIGHTS
# ============================================================

with insights_tab:
    st.header("Model Insights")

    d_col, s_col = st.columns([1.2, 1])
    with d_col:
        st.subheader("Training corpus by topic")
        dist = pd.Series(R["train_categories"]).value_counts().reindex(CATEGORIES).reset_index()
        dist.columns = ["Category", "Articles"]
        fig = px.bar(dist, x="Category", y="Articles", color="Category", color_discrete_map=CATEGORY_COLORS)
        fig.update_layout(showlegend=False, height=320)
        show_plot(fig)
    with s_col:
        st.subheader("Dataset statistics")
        show_df(pd.DataFrame({
            "Metric": ["Training articles", "Test articles", "GloVe vocabulary", "Embedding dimensions", "Categories"],
            "Value": [f"{len(train_df):,}", f"{len(test_df):,}", f"{len(words_list):,}", str(matrix.shape[1]), "4"],
        }))

    st.divider()
    st.subheader("Evaluation on the test split")
    if evaluation is None:
        st.warning("The test split could not be evaluated. Check that the test embeddings match the classifier.")
    else:
        m1, m2 = st.columns(2)
        m1.metric("Accuracy", f"{evaluation['accuracy'] * 100:.2f}%")
        m2.metric("Macro F1", f"{evaluation['macro_f1'] * 100:.2f}%")

        cm_col, rep_col = st.columns([1, 1])
        with cm_col:
            fig = px.imshow(evaluation["matrix"], x=CATEGORIES, y=CATEGORIES, text_auto=True,
                            color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual", color="Articles"),
                            title="Confusion matrix")
            fig.update_layout(height=420)
            show_plot(fig)
        with rep_col:
            rows = [
                {"Category": c, "Precision": evaluation["report"][c]["precision"],
                 "Recall": evaluation["report"][c]["recall"], "F1": evaluation["report"][c]["f1-score"],
                 "Articles": int(evaluation["report"][c]["support"])}
                for c in CATEGORIES
            ]
            st.markdown("**Per-category results**")
            show_df(pd.DataFrame(rows).round(3))
            st.caption("Business and Sci/Tech are the most commonly confused topics: their vocabulary overlaps heavily.")

    st.divider()
    st.subheader("System architecture")
    show_df(pd.DataFrame(
        [
            ["Word representation", "GloVe", "50-dimensional pretrained vectors"],
            ["Document representation", "Mean pooling", "Average of word vectors"],
            ["Semantic retrieval", "Cosine similarity", "Rank documents by vector direction"],
            ["Classification", "Linear SVM", "Four-topic classifier"],
            ["Visualization", "PCA", "50D to 2D / 3D projection"],
        ],
        columns=["Layer", "Method", "Purpose"],
    ))

    st.subheader("Limitations")
    st.markdown(
        "- **Word order is lost.** Averaging vectors ignores sentence structure.\n"
        "- **One vector per word.** Words with several meanings (like *apple*) get a single blended vector.\n"
        "- **Vocabulary coverage.** Words outside GloVe contribute nothing.\n"
        "- **Similarity is not fact-checking.** Close vectors mean similar language, not matching facts.\n"
        "- **PCA is a projection.** Two or three dimensions cannot show every relationship in 50.\n"
        "- **Static corpus.** Results come from the indexed AG News dataset, not live news."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer"><b>Semantic News Intelligence Engine</b><br>'
    "GloVe 50D · Mean document embeddings · Cosine similarity · Linear SVM · PCA</div>",
    unsafe_allow_html=True,
)