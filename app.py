# ============================================================
# Fake News Detector – Streamlit Cloud Safe Version
# WITH CNN Model + URL Mode + Custom LIME-Style Explanations
# ============================================================

import os
import json
import joblib
import numpy as np
import streamlit as st
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =======================================
# OPTIONAL IMPORTS
# =======================================

try:
    from newspaper import Article
except:
    Article = None

try:
    import feedparser
except:
    feedparser = None


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# LOAD MODEL, TOKENIZER, CONFIG
# ============================================================
@st.cache_resource
def load_all():
    model_path = os.path.join(BASE_DIR, "advanced_cnn_model.h5")
    tok_path = os.path.join(BASE_DIR, "tokenizer.pkl")
    cfg_path = os.path.join(BASE_DIR, "model_config.json")

    model = load_model(model_path)
    tokenizer = joblib.load(tok_path)
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    return model, tokenizer, cfg["max_len"]

model, tokenizer, max_len = load_all()


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================
def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding="post")
    return padded

def predict_news(text):
    padded = preprocess_text(text)
    prob = float(model.predict(padded)[0][0])
    is_fake = prob > 0.5
    label = "FAKE News" if is_fake else "REAL News"
    confidence = prob if is_fake else 1 - prob
    return label, confidence, prob


# ============================================================
# URL EXTRACTOR
# ============================================================
def extract_article(url):
    if Article is None:
        return None, "⚠️ newspaper3k is not installed on Streamlit Cloud."

    try:
        article = Article(url)
        article.download()
        article.parse()
        if not article.text.strip():
            return None, "Could not extract article text."
        return article.text, None
    except Exception as e:
        return None, f"❌ Error: {e}"


# ============================================================
# TRENDING FACT CHECK FEED
# ============================================================
def get_trending_items():
    if feedparser:
        try:
            feed = feedparser.parse("https://www.politifact.com/rss/factchecks/")
            items = [{"title": e.title, "link": e.link} for e in feed.entries[:5]]
            if items:
                return items
        except:
            pass

    # fallback
    return [
        {"title": "5G towers cause COVID-19? – FALSE", "link": "https://www.politifact.com/"},
        {"title": "Fake cash giveaway rumor", "link": "https://www.snopes.com/"},
        {"title": "Vaccine side effects claim debunked", "link": "https://www.factcheck.org/"},
    ]


# ============================================================
# CUSTOM LIME-STYLE EXPLAINER
# ============================================================
def get_word_importance(text, model, tokenizer, maxlen):
    """Gradient-based word importance (Streamlit-safe, no LIME needed)."""
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=maxlen)
    input_tensor = tf.convert_to_tensor(padded)

    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        pred = model(input_tensor)
        target_class = tf.argmax(pred[0])

    grads = tape.gradient(pred[:, target_class], input_tensor)
    grads = tf.reduce_mean(tf.abs(grads), axis=2).numpy()[0]

    words = text.split()
    last_tokens = seq[0][-len(words):]

    word_scores = []
    for i, w in enumerate(words):
        try:
            score = float(grads[-len(words) + i])
            word_scores.append((w, score))
        except:
            word_scores.append((w, 0.0))

    return sorted(word_scores, key=lambda x: x[1], reverse=True)


def display_word_importance(word_scores):
    st.write("### 🔍 Top Influential Words")

    top = word_scores[:10]
    median_score = np.median([s for _, s in word_scores])

    for word, score in top:
        color = "green" if score >= median_score else "red"
        st.write(f"**{word}** — {score:.4f}", unsafe_allow_html=True)

    # Highlighted text
    st.write("### ✨ Highlighted Text Explanation")
    highlighted = ""
    scores_dict = dict(word_scores)

    for w, s in word_scores:
        color = "#4CAF50" if s >= median_score else "#FF5252"
        highlighted += f"<span style='background-color:{color}; padding:4px; margin:2px'>{w}</span> "

    st.markdown(highlighted, unsafe_allow_html=True)


# ============================================================
# THEME / CSS
# ============================================================
def inject_css(theme):
    if theme == "Dark":
        bg = "linear-gradient(145deg, #050816, #111827)"
        card = "#111827"
        text = "#e5e7eb"
        sub = "#9ca3af"
        result_bg = "#020617"
        border = "#4f46e5"
    else:
        bg = "linear-gradient(145deg, #f6f9ff, #e9efff)"
        card = "#ffffff"
        text = "#111827"
        sub = "#4b5563"
        result_bg = "#f1f5f9"
        border = "#3b82f6"

    st.markdown(
        f"""
        <style>
            body {{ background: {bg}; }}
            .main-block {{
                background:{card}; padding:2rem; border-radius:18px;
                box-shadow:0 4px 20px rgba(0,0,0,0.12);
            }}
            .result-box {{
                background:{result_bg}; padding:1.4rem; border-radius:14px;
                border-left:6px solid {border}; margin-top:1.2rem;
            }}
            .prediction-badge {{
                font-size:26px; font-weight:700; padding:6px 14px;
                border-radius:999px; background:#0f172a11;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("## 🧭 Navigation")

theme_choice = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)
inject_css(theme_choice)

input_mode = st.sidebar.radio("Input Mode", ["📝 Enter Text Manually", "🌐 Paste URL"])

st.sidebar.markdown("### 📰 Trending fact-checks")
for item in get_trending_items():
    st.sidebar.markdown(f"- [{item['title']}]({item['link']})")


# ============================================================
# HEADER
# ============================================================
try:
    logo = Image.open(os.path.join(BASE_DIR, "logo.png"))
    st.image(logo, width=170)
except:
    pass

st.markdown("<h1 style='text-align:center;'>📡 Fake News Detection Dashboard</h1>", unsafe_allow_html=True)
st.write("<p style='text-align:center;'>AI-powered predictions + word-level explanations</p>", unsafe_allow_html=True)


# ============================================================
# MAIN CONTENT BLOCK
# ============================================================
st.markdown("<div class='main-block'>", unsafe_allow_html=True)

# ----------------- TEXT MODE -----------------------
if input_mode == "📝 Enter Text Manually":

    st.subheader("✍️ Enter News Text")
    user_text = st.text_area("", height=200)

    if st.button("Analyze Text"):
        if not user_text.strip():
            st.warning("Please enter some text.")
        else:
            label, conf, raw = predict_news(user_text)
            emoji = "🔴" if "FAKE" in label else "🟢"

            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.markdown(f"<div class='prediction-badge'>{emoji} {label}</div>", unsafe_allow_html=True)
            st.write(f"**Confidence:** {conf:.3f}")

            st.write("---")
            st.write("### 🔍 Why did the model predict this?")
            scores = get_word_importance(user_text, model, tokenizer, max_len)
            display_word_importance(scores)

            st.markdown("</div>", unsafe_allow_html=True)


# ----------------- URL MODE ------------------------
else:
    st.subheader("🌐 Paste Article URL")
    url = st.text_input("URL")

    if st.button("Fetch & Analyze"):
        if not url.strip():
            st.warning("Please paste a URL.")
        else:
            if Article is None:
                st.error("newspaper3k is not installed.")
            else:
                with st.spinner("Extracting article..."):
                    article, err = extract_article(url)

                if err:
                    st.error(err)
                else:
                    preview = article[:700] + ("..." if len(article) > 700 else "")
                    st.write("### Extracted Preview")
                    st.write(preview)

                    label, conf, raw = predict_news(article)
                    emoji = "🔴" if "FAKE" in label else "🟢"

                    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='prediction-badge'>{emoji} {label}</div>", unsafe_allow_html=True)
                    st.write(f"**Confidence:** {conf:.3f}")

                    st.write("---")
                    st.write("### 🔍 Why did the model predict this?")
                    scores = get_word_importance(article, model, tokenizer, max_len)
                    display_word_importance(scores)

                    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
