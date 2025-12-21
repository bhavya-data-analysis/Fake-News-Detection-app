# 📰 Fake News Detection App

This project is a **Fake News Detection system** built using Natural Language Processing and Machine Learning.  
It contains **two implementations** in a single repository:

- A **cloud-based lightweight demo** (Logistic Regression)
- A **local full model version** (CNN + Logistic Regression + LIME)

---

## 📁 Repository Structure
```
Fake-News-Detection-app/
│
├── README.md
│
├── local_app/ # Full local version (for evaluation)
│ ├── app.py # Streamlit app (CNN + LR + LIME)
│ ├── models/
│ │ ├── advanced_cnn_model.h5
│ │ ├── log_reg.pkl
│ │ ├── tfidf_vectorizer.pkl
│ │ └── tokenizer.pkl
│ └── notebook/ # Training / experimentation notebooks
│
└── cloud_app/ # Cloud demo version
├── app.py # Streamlit app (Logistic Regression only)
├── log_reg.pkl
├── tfidf_vectorizer.pkl
└── requirements.txt
```

---

## ☁️ Cloud Version (Public Demo)

🔗 **Live App:**  
https://fake-news-detection-app-adptswkkruuf4keteyadn6.streamlit.app

**Model Used**
- Logistic Regression + TF-IDF

**Why this version**
- Cloud platforms do not reliably support TensorFlow-based CNN models
- This version is lightweight, fast, and stable for public access

**Use Case Remember**
- Class demo
- Sharing link with others
- Quick testing

---

## 🖥️ Local Version (Full Model)

**Models Used**
- CNN (Convolutional Neural Network)
- Logistic Regression
- LIME (Explainability)

This version demonstrates the **complete deep learning pipeline** and is intended for **academic evaluation**.
---
### ▶️ How to Run Locally
```bash
streamlit run app.py
```
---

## 📦 Requirements
---
```
Python 3.10
TensorFlow 2.10
Streamlit
scikit-learn
LIME
```
---

## 📌 Notes
---
The cloud version is a simplified deployment due to platform limitations.
The local version contains the full CNN architecture and explainability features.
Predictions are designed for news-style text; casual text may be classified as fake.

---

## 👤 Author
---
- Bhavya Pandya
- Graduate Student – Data Analytics
