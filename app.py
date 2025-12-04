import os
import json
import numpy as np
import joblib
from PIL import Image

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Optional: URL article extraction
try:
    from newspaper import Article
except Exception:
    Article = None

# Optional: live trend feed
try:
    import feedparser
except Exception:
    feedparser = None


# ==============================
# Page config
# ==============================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ==============================
# Load model, tokenizer, config
# ==============================
@st.cache_resource
def load_all():
    model_path = os.path.join(BASE_DIR, "advanced_cnn_model.h5")
    tok_path = os.path.join(BASE_DIR, "tokenizer.pkl")
    cfg_path = os.path.join(BASE_DIR, "model_config.json")

    # Load model (CPU-only, compile disabled for speed/compat)
    model = load_model(model_path, compile=False)

    tokenizer = joblib.load(tok_path)
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    return model, tokenizer, cfg["max_len"]


model, tokenizer, max_len = load_all()


# ==============================
# Helper functions
# ==============================
def preprocess_text(text: str):
    seq = tokenizer.texts_to_sequences([text])
    return pad_sequences(seq, maxlen=max_len, padding="post")


def predict_news(text: str):
    padded = preprocess_text(text)
    prob = float(model.predict(padded)[0][0])
    is_fake = prob > 0.5
    label = "FAKE News" if is_fake else "REAL News"
    confidence = prob if is_fake else 1 - prob
    return label, confidence, prob


def extract_article(url: str):
    if Article is None:
        return None, "⚠️ `newspaper3k` not installed. Run: pip install newspaper3k"

    try:
        article = Article(url)
        article.download()
        article.parse()
        if not article.text.strip():
            return None, "Could not extract article text from this URL."
        return article.text, None
    except Exception as e:
        return None, f"❌ Error extracting article: {e}"


def get_trending_items():
    """Return list of {'title','link'} dicts for sidebar."""
    if feedparser is None:
        # Fallback static examples
        return [
            {
                "title": "Claim that 5G towers cause COVID-19 rated false",
                "link": "https://www.politifact.com/"
            },
            {
                "title": "Rumor of cash giveaway by central bank debunked",
                "link": "https://www.snopes.com/"
            },
            {
                "title": "Viral post misleads about vaccine side effects",
                "link": "https://www.factcheck.org/"
            },
        ]
    try:
        feed = feedparser.parse("https://www.politifact.com/rss/factchecks/")
        items = []
        for entry in feed.entries[:5]:
            items.append({"title": entry.title, "link": entry.link})
        if items:
            return items
    except Exception:
        pass
    # fallback if RSS fails
    return [
        {
            "title": "Latest fact-checks unavailable (offline) – demo headlines shown.",
            "link": "https://www.politifact.com/"
        }
    ]


# ==============================
# EXPLANATION: word-level contributions
# ==============================

def explain_prediction(text: str):
    """
    Occlusion-style explanation:
    For each word, remove it and see how much the model's target
    probability changes. That change = importance score.
    Positive score -> pushes towards the predicted label.
    Negative score -> pushes against it.
    """
    words = text.split()
    if len(words) < 3:
        return []

    # Base prediction
    base_padded = preprocess_text(text)
    base_prob = float(model.predict(base_padded)[0][0])
    is_fake = base_prob > 0.5
    base_target = base_prob if is_fake else (1 - base_prob)

    scores = []
    for i in range(len(words)):
        tmp_words = words[:i] + words[i + 1:]
        if not tmp_words:
            continue
        tmp_text = " ".join(tmp_words)
        padded = preprocess_text(tmp_text)
        prob = float(model.predict(padded)[0][0])
        tmp_target = prob if is_fake else (1 - prob)
        importance = base_target - tmp_target
        scores.append((words[i], importance))

    if not scores:
        return []

    # Normalize for nicer display
    max_abs = max(abs(s) for _, s in scores) or 1.0
    norm_scores = [(w, s / max_abs) for w, s in scores]
    return norm_scores


def display_explanation(text: str):
    scores = explain_prediction(text)
    if not scores:
        st.info("Not enough text for explanation. Try a longer headline or paragraph.")
        return

    st.subheader("🔍 Word-level explanation")

    # Show top 10 strongest contributors
    top = sorted(scores, key=lambda x: abs(x[1]), reverse=True)[:10]
    st.markdown("**Top influential words for this prediction:**")
    for w, s in top:
        direction = "⬆️ pushes towards this label" if s >= 0 else "⬇️ pushes away from this label"
        st.write(f"- **{w}** — contribution: {s:+.2f}  ({direction})")

    # Highlight full text
    st.markdown("**Highlighted text (green = supporting, red = opposing):**")

    html = ""
    for w, s in scores:
        if s >= 0:
            color = "#bbf7d0"  # light green
        else:
            color = "#fecaca"  # light red
        html += f"<span style='background-color:{color}; padding:2px 5px; margin:1px; border-radius:4px;'>{w}</span> "

    st.markdown(html, unsafe_allow_html=True)


# ==============================
# Theming – inject CSS
# ==============================
def inject_css(theme: str):
    if theme == "Dark":
        bg_grad = "linear-gradient(145deg, #050816, #111827)"
        card_bg = "#111827"
        text_color = "#e5e7eb"
        subtext_color = "#9ca3af"
        result_bg = "#020617"
        border_color = "#4f46e5"
    else:
        bg_grad = "linear-gradient(145deg, #f6f9ff, #e9efff)"
        card_bg = "#ffffff"
        text_color = "#111827"
        subtext_color = "#4b5563"
        result_bg = "#f1f5f9"
        border_color = "#3b82f6"

    st.markdown(
        f"""
        <style>
            body {{
                background: {bg_grad};
            }}
            .main-block {{
                background: {card_bg};
                padding: 2rem;
                border-radius: 18px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.12);
            }}
            .intro-box {{
                background: rgba(255,255,255,0.12);
                padding: 1.2rem 1.6rem;
                border-radius: 12px;
                text-align: center;
                color: {subtext_color};
                margin-bottom: 1.5rem;
            }}
            .result-box {{
                background: {result_bg};
                padding: 1.4rem 1.7rem;
                border-radius: 14px;
                border-left: 6px solid {border_color};
                margin-top: 1.2rem;
            }}
            .title-text {{
                text-align: center;
                font-size: 42px;
                font-weight: 700;
                color: {text_color};
                margin-bottom: 0.3rem;
            }}
            .subtitle-text {{
                text-align: center;
                font-size: 18px;
                color: {subtext_color};
            }}
            .confidence-track {{
                width: 100%;
                background: #0f172a33;
                border-radius: 999px;
                height: 16px;
                margin-top: 10px;
                overflow: hidden;
            }}
            .confidence-fill {{
                height: 100%;
                border-radius: 999px;
                background: linear-gradient(90deg,#ef4444,#f97316,#eab308,#22c55e);
                width: 0%;
                transition: width 0.7s ease-out;
            }}
            .prediction-badge {{
                font-size: 26px;
                font-weight: 700;
                display: inline-block;
                padding: 6px 14px;
                border-radius: 999px;
                background: #0f172a11;
                animation: pop-in 0.35s ease-out;
            }}
            @keyframes pop-in {{
                0%   {{ transform: scale(0.7); opacity: 0; }}
                100% {{ transform: scale(1); opacity: 1; }}
            }}
        </style>
        """,
        unsafe_allow_html=True
    )


# ==============================
# Sidebar
# ==============================
st.sidebar.markdown("## 🧭 Navigation")

theme_choice = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)
inject_css(theme_choice)

input_type = st.sidebar.radio(
    "Input Mode",
    ["📝 Enter Text Manually", "🌐 Paste URL"]
)

st.sidebar.markdown("### 📰 Trending fact-checks")
for item in get_trending_items():
    st.sidebar.markdown(f"- [{item['title']}]({item['link']})")


# ==============================
# Logo + Title
# ==============================
try:
    logo = Image.open(os.path.join(BASE_DIR, "logo.png"))
    st.image(logo, width=170)
except Exception:
    st.write("")

st.markdown(
    "<div class='title-text'>📡 Fake News Detection Dashboard</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='subtitle-text'>AI-powered classifier for headlines, articles, and live URLs</div><br>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="intro-box">
        Paste any news headline, paragraph, or article link below.<br/>
        The model will predict whether it is more likely <b>Fake</b> or <b>Real</b>.
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================
# Main content card
# ==============================
with st.container():
    st.markdown("<div class='main-block'>", unsafe_allow_html=True)

    # --------- Manual Text Mode ----------
    if input_type == "📝 Enter Text Manually":
        st.subheader("✍️ Enter News Text")
        user_text = st.text_area(
            "News text",
            placeholder="Type or paste a news headline or short article here...",
            height=200,
            label_visibility="collapsed"  # hides label, fixes accessibility warning
        )

        if st.button("Analyze Text"):
            if not user_text.strip():
                st.warning("Please enter some text first.")
            else:
                label, conf, raw_prob = predict_news(user_text)
                is_fake = "FAKE" in label
                emoji = "🔴" if is_fake else "🟢"

                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='prediction-badge'>{emoji} {label}</div>",
                    unsafe_allow_html=True
                )
                st.write(f"**Confidence:** {conf:.3f}")

                bar_html = f"""
                <div class="confidence-track">
                    <div class="confidence-fill" style="width:{conf*100:.1f}%"></div>
                </div>
                <div style="margin-top:4px; font-size:13px; color:#6b7280;">
                    Model is {conf*100:.1f}% confident.
                </div>
                """
                st.markdown(bar_html, unsafe_allow_html=True)

                # Word-level explanation
                with st.expander("🔍 Why did the model predict this?"):
                    display_explanation(user_text)

                st.markdown("</div>", unsafe_allow_html=True)

    # --------- URL Mode ----------
    else:
        st.subheader("🌐 Paste Article URL")
        url = st.text_input(
            "News article URL",
            placeholder="https://www.example.com/news/article"
        )

        if st.button("Fetch & Analyze"):
            if not url.strip():
                st.warning("Please paste a URL first.")
            else:
                if Article is None:
                    st.error("`newspaper3k` is not installed. Run: pip install newspaper3k")
                else:
                    with st.spinner("Downloading & parsing article..."):
                        article_text, err = extract_article(url)

                    if err:
                        st.error(err)
                    else:
                        st.success("Article extracted successfully!")

                        preview = article_text[:700] + ("..." if len(article_text) > 700 else "")
                        st.markdown("**Preview of extracted text:**")
                        st.write(preview)

                        with st.expander("📄 Show full article text"):
                            st.write(article_text)

                        label, conf, raw_prob = predict_news(article_text)
                        is_fake = "FAKE" in label
                        emoji = "🔴" if is_fake else "🟢"

                        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                        st.markdown(
                            f"<div class='prediction-badge'>{emoji} {label}</div>",
                            unsafe_allow_html=True
                        )
                        st.write(f"**Confidence:** {conf:.3f}")

                        bar_html = f"""
                        <div class="confidence-track">
                            <div class="confidence-fill" style="width:{conf*100:.1f}%"></div>
                        </div>
                        <div style="margin-top:4px; font-size:13px; color:#6b7280;">
                            Model is {conf*100:.1f}% confident.
                        </div>
                        """
                        st.markdown(bar_html, unsafe_allow_html=True)

                        # Word-level explanation for full article
                        with st.expander("🔍 Why did the model predict this?"):
                            display_explanation(article_text)

                        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
