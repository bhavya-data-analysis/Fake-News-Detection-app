import os
import json
import numpy as np
import pandas as pd
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

# Optional: Stable Baselines 3 (for RL agent)
try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except Exception:
    PPO = None
    SB3_AVAILABLE = False

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

    # Load CNN model (CPU only, compile disabled for safety)
    model = load_model(model_path, compile=False)

    tokenizer = joblib.load(tok_path)
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    max_len = cfg.get("max_len", 200)
    return model, tokenizer, max_len


model, tokenizer, max_len = load_all()


def preprocess_text(text: str):
    """Basic tokenizer → padded sequence (used by explainer)."""
    seq = tokenizer.texts_to_sequences([text])
    return pad_sequences(seq, maxlen=max_len, padding="post")


# ============================================
# Load RL Agent (PPO)
# ============================================
rl_agent = None
rl_enabled = False

if SB3_AVAILABLE:
    try:
        RL_AGENT_PATH = os.path.join(BASE_DIR, "rl_agent", "rl_fake_news_agent.zip")
        rl_agent = PPO.load(RL_AGENT_PATH)
        rl_enabled = True
    except Exception:
        rl_agent = None
        rl_enabled = False
        st.warning("⚠️ RL agent could not be loaded. Using standard CNN model only.")
else:
    st.info("ℹ️ Stable-Baselines3 not available. Running in CNN-only mode.")


# ============================================================
# RL Text Refinement (lightweight inference version)
# ============================================================
def rl_refine_text(text, tokenizer, rl_agent, steps=5, max_len=200):
    """
    Runs the trained RL agent to refine the input text before CNN prediction.
    No gym environment needed – this is a lightweight inference loop.
    """
    # Convert text to token IDs
    seq = tokenizer.texts_to_sequences([text])[0]
    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq += [0] * (max_len - len(seq))

    seq = np.array(seq, dtype=np.int32)
    weights = np.ones(max_len, dtype=np.float32)

    # Build observation vector (must match env.py logic)
    def build_obs(token_ids, weights_vec):
        ids = token_ids.astype(np.float32)
        max_v = np.max(ids)
        if max_v > 0:
            ids_scaled = (ids / max_v) * 2.0
        else:
            ids_scaled = ids
        return np.concatenate([ids_scaled, weights_vec]).astype(np.float32)

    obs = build_obs(seq, weights)

    num_ops = 3
    noop_action = max_len * num_ops

    # Execute RL actions
    for _ in range(steps):
        action, _ = rl_agent.predict(obs, deterministic=True)

        if action == noop_action:
            break  # RL chooses to stop

        pos = action // num_ops
        op = action % num_ops

        if 0 <= pos < max_len:
            if op == 0:
                weights[pos] = 0.0
            elif op == 1:
                weights[pos] = min(weights[pos] + 0.5, 2.0)
            elif op == 2:
                weights[pos] = max(weights[pos] - 0.5, 0.0)

        obs = build_obs(seq, weights)

    return seq, weights


# ==============================
# Unified prediction: CNN (+ RL if enabled)
# ==============================
def predict_with_rl(text: str):
    """
    Main prediction function:
    - If RL agent is available → refine tokens with RL
    - Else → normal CNN pipeline
    Returns: label, confidence_for_label, prob_fake, is_fake
    """
    if rl_enabled and rl_agent is not None:
        seq, weights = rl_refine_text(text, tokenizer, rl_agent, steps=5, max_len=max_len)
        seq_weighted = seq * (weights > 0).astype(int)
    else:
        seq = tokenizer.texts_to_sequences([text])[0]
        if len(seq) > max_len:
            seq = seq[:max_len]
        else:
            seq += [0] * (max_len - len(seq))
        seq_weighted = np.array(seq, dtype=np.int32)

    padded = pad_sequences([seq_weighted], maxlen=max_len, padding="post")
    prob_fake = float(model.predict(padded, verbose=0)[0][0])

    is_fake = prob_fake > 0.5
    label = "FAKE News" if is_fake else "REAL News"
    confidence = prob_fake if is_fake else (1 - prob_fake)

    return label, confidence, prob_fake, is_fake


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
# CUSTOM EXPLANATION ENGINE
# (occlusion-based, premium UI)
# ==============================
def compute_token_importances(text: str, base_prob_fake: float, is_fake: bool, max_words: int = 60):
    """
    For each word, remove it and see how the predicted probability
    for the predicted class changes.
    Positive score = pushes towards the predicted label.
    Negative score = pushes against the predicted label.
    """
    words = text.split()
    if not words:
        return []

    # Limit words for speed
    words = words[:max_words]
    base_class_score = base_prob_fake if is_fake else (1.0 - base_prob_fake)

    scores = []
    for idx, w in enumerate(words):
        occluded = " ".join(words[:idx] + words[idx + 1:])
        if not occluded.strip():
            continue
        padded = preprocess_text(occluded)
        prob_fake_occ = float(model.predict(padded, verbose=0)[0][0])
        class_score_occ = prob_fake_occ if is_fake else (1.0 - prob_fake_occ)
        impact = base_class_score - class_score_occ
        scores.append((w, impact))

    # Sort by absolute impact
    scores.sort(key=lambda x: abs(x[1]), reverse=True)
    return scores


def render_explanation(text: str, base_prob_fake: float, is_fake: bool):
    scores = compute_token_importances(text, base_prob_fake, is_fake)
    if not scores:
        st.info("Not enough text to generate an explanation.")
        return

    # Take top 12 for display
    top_scores = scores[:12]
    words = [w for w, s in top_scores]
    vals = [float(s) for _, s in top_scores]

    # Summary bullets
    st.markdown("### 🧠 Deep Explanation")
    if is_fake:
        st.write(
            "The model is **more confident this is FAKE** when certain words are present. "
            "Words with higher positive impact push the prediction towards FAKE."
        )
    else:
        st.write(
            "The model is **more confident this is REAL** when certain words are present. "
            "Words with higher positive impact push the prediction towards REAL."
        )

    # --- Bar chart (top influential words) ---
    df = pd.DataFrame(
        {"word": words, "importance": vals}
    ).set_index("word")

    st.markdown("#### 🔝 Most influential words")
    st.bar_chart(df)

    # --- Highlighted text chips ---
    st.markdown("#### 🎨 Highlighted sentence")

    all_words = text.split()
    # Build a quick lookup dict
    impact_map = {w: s for w, s in scores}

    if impact_map:
        max_abs = max(abs(v) for v in impact_map.values()) + 1e-8
    else:
        max_abs = 1.0

    def word_to_span(w):
        raw_score = impact_map.get(w, 0.0)
        norm = raw_score / max_abs

        # Positive = supports predicted class
        if norm > 0:
            if is_fake:
                base_color = "#f97373"  # reddish for fake
            else:
                base_color = "#4ade80"  # green for real
        else:
            # Opposes predicted class = faded blue/grey
            base_color = "#e5e7eb"

        intensity = min(0.8, 0.2 + abs(norm))  # 0.2–0.8
        bg = base_color
        opacity = intensity

        style = (
            f"display:inline-block; padding:2px 6px; margin:2px; "
            f"border-radius:999px; background-color:{bg}; "
            f"opacity:{opacity}; font-size:14px;"
        )
        score_str = f"{raw_score:+.3f}"
        return f"<span style='{style}'>{w} ({score_str})</span>"

    chips_html = " ".join(word_to_span(w) for w in all_words)
    st.markdown(chips_html, unsafe_allow_html=True)

    # --- Textual reason summary ---
    st.markdown("#### 📝 Key drivers")
    strong = [w for w, s in top_scores if s > 0]
    against = [w for w, s in top_scores if s < 0]

    if strong:
        st.write(
            "- Main words **supporting** this prediction: "
            + ", ".join(f"`{w}`" for w in strong[:6])
        )
    if against:
        st.write(
            "- Words that **pull in the opposite direction**: "
            + ", ".join(f"`{w}`" for w in against[:6])
        )


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

if rl_enabled:
    st.sidebar.success("🤖 RL refinement: **ON**")
else:
    st.sidebar.info("📡 RL refinement: **OFF** (CNN-only)")

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
            label_visibility="collapsed",
        )

        if st.button("Analyze Text"):
            if not user_text.strip():
                st.warning("Please enter some text first.")
            else:
                if rl_enabled:
                    st.info("🤖 RL refinement enabled: optimizing text representation before prediction...")
                label, conf, prob_fake, is_fake = predict_with_rl(user_text)
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

                with st.expander("🔍 Why did the model predict this? (advanced explanation)"):
                    render_explanation(user_text, prob_fake, is_fake)

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

                        if rl_enabled:
                            st.info("🤖 RL refinement enabled: optimizing text representation before prediction...")
                        label, conf, prob_fake, is_fake = predict_with_rl(article_text)
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

                        with st.expander("🔍 Why did the model predict this? (advanced explanation)"):
                            render_explanation(article_text, prob_fake, is_fake)

                        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
