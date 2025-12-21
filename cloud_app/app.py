# Fake News Detector 

import streamlit as st
import pickle
import re
import numpy as np
from pathlib import Path
import os

# PAGE CONFIG

st.set_page_config(
    page_title="Fake News Detector (Cloud)",
    layout="wide"
)

# Hero Section

st.markdown("""
<style>
.hero {
    background-color: #ffffff;
    padding: 40px 50px;
    border-radius: 18px;
    margin-bottom: 20px;
    border: 1px solid #e8e8e8;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.05);
    text-align: center;
}
.hero-title {
    font-size: 36px;
    font-weight: 700;
    color: #1f1f1f;
}
.hero-sub {
    font-size: 17px;
    color: #676b73;
}
.card {
    background-color: #ffffff;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}
.badge-real {
    background-color: #d4fbe8;
    color: #0a8754;
    border: 1px solid #0a8754;
    padding: 7px 14px;
    border-radius: 12px;
    font-weight: 600;
}
.badge-fake {
    background-color: #ffe0e0;
    color: #d7263d;
    border: 1px solid #d7263d;
    padding: 7px 14px;
    border-radius: 12px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-title">📰 Fake News Detector (Cloud Version)</div>
    <div class="hero-sub">Lightweight model using Logistic Regression + TF-IDF</div>
</div>
""", unsafe_allow_html=True)


# LOAD MODEL + VECTOR

@st.cache_resource
def load_model_and_vectorizer():
    """
    Robust loader that checks:
      - file next to this script (preferred)
      - fallback to current working directory
    If not found, displays a helpful Streamlit error listing checked paths and files present.
    """
    base_dir = Path(__file__).resolve().parent
    tfidf_path = base_dir / "tfidf_vectorizer.pkl"
    logreg_path = base_dir / "log_reg.pkl"

    alt_tfidf = Path.cwd() / "tfidf_vectorizer.pkl"
    alt_logreg = Path.cwd() / "log_reg.pkl"

    # Try preferred location (script folder)
    if tfidf_path.exists() and logreg_path.exists():
        with open(tfidf_path, "rb") as f:
            tfidf = pickle.load(f)
        with open(logreg_path, "rb") as f:
            logreg = pickle.load(f)
        return tfidf, logreg

    # Try current working directory
    if alt_tfidf.exists() and alt_logreg.exists():
        with open(alt_tfidf, "rb") as f:
            tfidf = pickle.load(f)
        with open(alt_logreg, "rb") as f:
            logreg = pickle.load(f)
        return tfidf, logreg

    # If we get here, files not found — produce helpful diagnostics
    try:
        base_files = sorted([p.name for p in base_dir.iterdir()])
    except Exception:
        base_files = "N/A"

    msg = (
        "Model files not found. Checked the following locations:\n\n"
        f" - {tfidf_path}\n"
        f" - {logreg_path}\n"
        f" - {alt_tfidf}\n"
        f" - {alt_logreg}\n\n"
        f"Current working directory: {Path.cwd()}\n"
        f"Files in script directory ({base_dir}): {base_files}\n\n"
        "Common causes:\n"
        "- The .pkl files are not committed to the repo or are on a different branch.\n"
        "- Files were committed with Git LFS but LFS isn't enabled in the deploy environment.\n"
        "- Filename/case mismatch (Linux is case-sensitive).\n"
        "- The app is executed from a different folder on the cloud host.\n"
    )
    st.error(msg)
    st.stop()


tfidf, logreg = load_model_and_vectorizer()


# CLEANING FUNCTION

def clean_text(text):
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).lower().strip()
    return text


# MAIN WORKSPACE

col1, col2 = st.columns([1, 1])

# INPUT SECTION

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("1️⃣ Enter Text")
    user_input = st.text_area("Paste article text here:", height=200)
    analyze = st.button("🚀 Analyze")
    st.markdown("</div>", unsafe_allow_html=True)

# ANALYSIS

if analyze:
    cleaned = clean_text(user_input)

    if len(cleaned) < 5:
        st.error("Please enter at least one sentence.")
        st.stop()

    probs = logreg.predict_proba(tfidf.transform([cleaned]))[0]
    p_real, p_fake = probs
    label = "Fake" if p_fake >= 0.5 else "Real"
    badge = "badge-fake" if label == "Fake" else "badge-real"

   
    # RESULTS SECTION
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔍 Prediction")
        st.markdown(f'<span class="{badge}">Overall Prediction: {label.upper()}</span>',
                    unsafe_allow_html=True)

        st.write(f"Fake Probability: **{p_fake:.3f}**")
        st.write(f"Real Probability: **{p_real:.3f}**")

        st.markdown("</div>", unsafe_allow_html=True)

   
    # EXPLANATION SECTION
   
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧠 Model Notes")
        st.write("""
        - This lightweight model uses TF-IDF + Logistic Regression  
        - Cloud-friendly → no TensorFlow  
        - Local version includes CNN + LIME  
        - This is a fast, simplified web demo  
        """)
        st.markdown("</div>", unsafe_allow_html=True)
