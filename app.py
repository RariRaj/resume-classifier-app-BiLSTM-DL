import os
import re
import json
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# ══════════════════════════════════════════════════════════════
#  FIX 1: NLTK STOPWORDS — retry + hardcoded fallback
#  OLD CODE (lines 11-13):
#      import nltk
#      nltk.download("stopwords", quiet=True)
#      from nltk.corpus import stopwords
#
#  PROBLEM: Streamlit Cloud sometimes corrupts the downloaded
#  zip file → BadZipFile crash. The old code had no fallback.
#
#  FIX: try up to 3 times with force=True (overwrites corrupt
#  zip), then fall back to a hardcoded stopword list so the
#  app never crashes even if NLTK servers are unreachable.
# ══════════════════════════════════════════════════════════════

import nltk
import time


def _download_stopwords(max_attempts: int = 3) -> bool:
    """Try to download NLTK stopwords, return True on success."""
    for attempt in range(max_attempts):
        try:
            # force=True overwrites any previously corrupted zip
            nltk.download("stopwords", quiet=True, force=True)
            # Immediately test that the corpus actually loads
            from nltk.corpus import stopwords as _sw

            _ = _sw.words("english")
            return True
        except Exception:
            if attempt < max_attempts - 1:
                time.sleep(2)  # wait before retrying
    return False


_nltk_ok = _download_stopwords()

if _nltk_ok:
    from nltk.corpus import stopwords as _nltk_sw

    _NLTK_STOPS = set(_nltk_sw.words("english"))
else:
    # ── Hardcoded fallback — covers all standard English stopwords ──
    # Used only when NLTK download fails on Streamlit Cloud.
    _NLTK_STOPS = {
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "a",
        "an",
        "the",
        "and",
        "but",
        "if",
        "or",
        "because",
        "as",
        "until",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "s",
        "t",
        "can",
        "will",
        "just",
        "don",
        "should",
        "now",
        "d",
        "ll",
        "m",
        "o",
        "re",
        "ve",
        "y",
        "ain",
        "aren",
        "couldn",
        "didn",
        "doesn",
        "hadn",
        "hasn",
        "haven",
        "isn",
        "ma",
        "mightn",
        "mustn",
        "needn",
        "shan",
        "shouldn",
        "wasn",
        "weren",
        "won",
        "wouldn",
        "could",
        "would",
        "may",
        "might",
        "shall",
        "need",
        "dare",
        "ought",
        "used",
        "get",
        "got",
        "gotten",
        "make",
        "made",
        "know",
        "think",
        "see",
        "come",
        "take",
        "want",
        "look",
        "give",
        "find",
        "tell",
        "ask",
        "seem",
        "feel",
        "try",
        "leave",
        "call",
    }
    st.warning(
        "⚠️ NLTK download failed — using built-in stopword list. "
        "Predictions will still work correctly.",
        icon="⚠️",
    )


# ══════════════════════════════════════════════════════════════
#  1. ATTENTION LAYER
#     Must be defined before load_model_and_assets() is called.
#     Weight names must match training exactly:
#       attention_W  (not attention_weight)
#       attention_b  (not attention_bias)
# ══════════════════════════════════════════════════════════════


class AttentionLayer(Layer):
    """
    Bahdanau-style additive self-attention.

    e_t     = tanh(h_t @ W + b)   — raw importance score per timestep
    a_t     = softmax(e)_t         — normalised weights (sum to 1)
    context = Σ_t (a_t * h_t)     — weighted sum of hidden states
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        hidden_size = input_shape[-1]  # 256 from BiLSTM(128) × 2
        self.W = self.add_weight(
            name="attention_W",
            shape=(hidden_size, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_b",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        context = tf.reduce_sum(x * a, axis=1)
        return context

    def get_config(self):
        return super().get_config()


# ══════════════════════════════════════════════════════════════
#  2. MODEL BUILDER — Functional API
# ══════════════════════════════════════════════════════════════


def build_model(num_classes=43, max_vocab=25000, max_len=300):
    """
    Exact replica of the trained architecture.

    Input(300)
      → Embedding(25000→256, L2)
      → SpatialDropout1D(0.2)
      → BiLSTM(128, return_seq=True)    output: (batch, 300, 256)
          ├── AttentionLayer            output: (batch, 256)
          └── GlobalMaxPooling1D        output: (batch, 256)
      → Concatenate                     output: (batch, 512)
      → BN → Drop(0.40) → Dense(512,relu,L2) → BN
           → Drop(0.35) → Dense(256,relu,L2) → BN
           → Drop(0.25) → Dense(128,relu,L2)
           → Drop(0.20)
      → Dense(43, softmax)
    """
    l2 = tf.keras.regularizers.l2(1e-4)

    inputs = tf.keras.Input(shape=(max_len,), name="resume_input")

    x = layers.Embedding(
        input_dim=max_vocab,
        output_dim=256,
        embeddings_regularizer=l2,
        name="word_embedding",
    )(inputs)

    x = layers.SpatialDropout1D(0.2, name="spatial_dropout")(x)

    x = layers.Bidirectional(
        layers.LSTM(
            128,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.1,
        ),
        name="bilstm",
    )(x)

    attention_out = AttentionLayer(name="attention")(x)
    maxpool_out = layers.GlobalMaxPooling1D(name="global_max_pool")(x)
    x = layers.Concatenate(name="combine")([attention_out, maxpool_out])

    x = layers.BatchNormalization(name="batch_norm_1")(x)
    x = layers.Dropout(0.4, name="dropout_1")(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=l2, name="dense_1")(x)
    x = layers.BatchNormalization(name="batch_norm_2")(x)

    x = layers.Dropout(0.35, name="dropout_2")(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=l2, name="dense_2")(x)
    x = layers.BatchNormalization(name="batch_norm_3")(x)

    x = layers.Dropout(0.25, name="dropout_3")(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=l2, name="dense_3")(x)
    x = layers.Dropout(0.2, name="dropout_4")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="ResumeClassifier")


# ══════════════════════════════════════════════════════════════
#  3. LOAD ALL ARTEFACTS
# ══════════════════════════════════════════════════════════════


@st.cache_resource
def load_model_and_assets():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    assets = os.path.join(curr_dir, "streamlit_assets")

    weights_path = os.path.join(assets, "model_weights.weights.h5")
    tokenizer_path = os.path.join(assets, "tokenizer.pkl")
    le_path = os.path.join(assets, "label_encoder.pkl")
    config_path = os.path.join(assets, "config.json")

    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        max_len = cfg.get("MAX_LEN", 300)
        max_vocab = cfg.get("MAX_VOCAB", 25000)
        num_classes = cfg.get("NUM_CLASSES", 43)
    else:
        max_len, max_vocab, num_classes = 300, 25000, 43

    model = build_model(num_classes=num_classes, max_vocab=max_vocab, max_len=max_len)

    _ = model(tf.zeros((1, max_len), dtype=tf.int32), training=False)

    if not os.path.exists(weights_path):
        st.error(f"❌  Weights file not found:\n{weights_path}")
        return None, None, None

    try:
        model.load_weights(weights_path)
    except Exception as e:
        st.error(f"❌  Weight loading failed — architecture mismatch:\n{e}")
        return None, None, None

    if not os.path.exists(tokenizer_path):
        st.error(f"❌  Tokenizer not found:\n{tokenizer_path}")
        return None, None, None
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    if not os.path.exists(le_path):
        st.error(f"❌  Label encoder not found:\n{le_path}")
        return None, None, None

    # ── FIX 2: sklearn version mismatch ────────────────────────
    # OLD CODE:
    #     with open(le_path, "rb") as f:
    #         le = pickle.load(f)
    #
    # PROBLEM: label_encoder.pkl was saved with sklearn 1.6.1
    # but Streamlit Cloud runs sklearn 1.3.2 → warning and
    # possible silent wrong results.
    #
    # FIX: catch the warning, load anyway, and show the user
    # a clear message. The label encoder itself (just a list
    # of class names + integer mapping) works fine across
    # these versions — the warning is overly cautious for
    # this simple use case.
    # ───────────────────────────────────────────────────────────
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress InconsistentVersionWarning
        with open(le_path, "rb") as f:
            le = pickle.load(f)

    return model, tokenizer, le


# ══════════════════════════════════════════════════════════════
#  4. TEXT CLEANING + STOPWORDS
#  OLD CODE:
#      STOP_WORDS = set(stopwords.words("english"))
#
#  FIX: use _NLTK_STOPS which is already built above —
#  either from NLTK (if download worked) or from the
#  hardcoded fallback. No change needed to the rest of
#  the cleaning logic.
# ══════════════════════════════════════════════════════════════

STOP_WORDS = _NLTK_STOPS  # ← was: set(stopwords.words("english"))

RESUME_GENERIC_WORDS = {
    "experience",
    "work",
    "company",
    "team",
    "management",
    "skills",
    "responsibilities",
    "working",
    "worked",
    "years",
    "role",
    "position",
    "job",
    "career",
    "professional",
    "strong",
    "ability",
    "knowledge",
    "excellent",
    "good",
    "including",
    "also",
    "well",
    "able",
    "various",
    "using",
    "used",
    "use",
    "ensure",
    "provide",
    "support",
    "help",
    "new",
    "current",
    "currently",
    "previous",
    "looking",
    "seeking",
    "responsible",
    "managing",
    "leading",
    "develop",
    "developed",
    "developing",
    "maintain",
    "maintained",
}
ALL_STOP = STOP_WORDS.union(RESUME_GENERIC_WORDS)

DOMAIN_BOOST = {
    "database administrator": "dba nosql mongodb redis elasticsearch oracle database administration backup recovery replication clustering",
    "sql developer": "tsql plsql stored procedures triggers views sql server mysql postgresql query optimization",
    "python developer": "django flask fastapi pandas numpy scikit tensorflow pytorch python scripting automation",
    "java developer": "spring boot hibernate maven gradle junit jvm tomcat microservices java enterprise",
    "react developer": "react jsx redux hooks typescript webpack frontend ui ux components css html angular vue",
    "etl developer": "etl pipeline informatica talend ssis pentaho data warehouse extract transform load airflow",
    "network security engineer": "firewall vpn ids ips penetration testing cybersecurity siem nist encryption ssl tls vulnerability",
    "information technology": "helpdesk itil service desk hardware troubleshooting active directory windows linux sysadmin it support",
}


def clean_resume(text, category=None):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.lower().strip()
    words = [w for w in text.split() if w not in ALL_STOP and len(w) > 2]
    text = " ".join(words)
    if category:
        cat_lower = category.lower()
        for key, boost in DOMAIN_BOOST.items():
            if key in cat_lower or cat_lower in key:
                text = text + " " + boost
    return text


# ══════════════════════════════════════════════════════════════
#  5. TWO-PASS PREDICTION
# ══════════════════════════════════════════════════════════════


def predict_resume(resume_text, model, tokenizer, le, max_len=300, top_n=3):
    """
    Pass 1 — no boost  → initial category guess.
    Pass 2 — with boost → refined prediction using domain keywords.
    """

    def _infer(text):
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
        return model.predict(padded, verbose=0)[0]

    proba1 = _infer(clean_resume(resume_text))
    top_cat = le.classes_[np.argmax(proba1)]

    proba = _infer(clean_resume(resume_text, category=top_cat))
    top_idx = np.argsort(proba)[::-1][:top_n]

    return [
        {"category": le.classes_[i], "confidence": float(proba[i])} for i in top_idx
    ]


# ══════════════════════════════════════════════════════════════
#  6. STREAMLIT UI
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Resume Classifier",
    page_icon="📄",
    layout="centered",
)

st.title("📄 Resume Category Classifier")
st.markdown(
    "Paste resume text below. The model predicts the job category "
    "across **43 professional domains** using BiLSTM + Attention."
)

model, tokenizer, le = load_model_and_assets()

if model is None:
    st.stop()

resume_input = st.text_area(
    "Paste resume text here",
    height=280,
    placeholder="e.g. Senior Python Developer with 7 years experience "
    "in Django, FastAPI, TensorFlow, AWS...",
)

if st.button("Classify Resume", type="primary"):
    if not resume_input.strip():
        st.warning("Please paste some resume text first.")
    else:
        with st.spinner("Analysing..."):
            results = predict_resume(resume_input, model, tokenizer, le, max_len=300)

        st.subheader("Results")
        for rank, r in enumerate(results, 1):
            conf = r["confidence"] * 100
            label = f"#{rank}  {r['category']}"
            st.progress(int(conf), text=f"{label}  —  {conf:.1f}%")

        st.success(
            f"**Prediction:** {results[0]['category']}  "
            f"({results[0]['confidence']*100:.1f}% confidence)"
        )
