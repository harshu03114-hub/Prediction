import streamlit as st
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

# =====================================================
# Streamlit Page Config
# =====================================================
st.set_page_config(
    page_title="Predicted You vs Real You",
    layout="wide"
)

st.title("🧠 Predicted You vs Real You")

# =====================================================
# Load models ONCE (important for memory)
# =====================================================
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    models = {
        "Extraversion": joblib.load("E_label_classifier.pkl"),
        "Openness": joblib.load("J_label_classifier.pkl"),
        "Conscientiousness": joblib.load("N_label_classifier.pkl"),
        "Agreeableness": joblib.load("T_label_classifier.pkl"),
    }
    return embedder, models

embedder, models = load_models()

# =====================================================
# Prediction Logic (pure ML)
# =====================================================
def predict_personality(text: str):
    embedding = embedder.encode([text])

    predictions = {}
    confidence = {}

    for trait, model in models.items():
        pred = model.predict(embedding)[0]
        prob = model.predict_proba(embedding)[0][pred]

        predictions[trait] = "High" if pred == 1 else "Low"
        confidence[trait] = round(float(prob), 3)

    return predictions, confidence

# =====================================================
# Question Bank (Real You)
# =====================================================
QUESTION_BANK = {
    "Extraversion": [
        "Do you feel energized in social gatherings?",
        "How often do you seek out conversations?"
    ],
    "Openness": [
        "Do you enjoy exploring new ideas?",
        "How do you feel about unconventional opinions?"
    ],
    "Conscientiousness": [
        "How do you plan your daily tasks?",
        "How do you handle deadlines?"
    ],
    "Agreeableness": [
        "How do you deal with conflicts?",
        "How important is empathy to you?"
    ]
}

# =====================================================
# Session State Initialization
# =====================================================
if "real_text" not in st.session_state:
    st.session_state.real_text = ""
    st.session_state.q_index = 0
    st.session_state.questions = sum(QUESTION_BANK.values(), [])

# =====================================================
# UI Layout
# =====================================================
col1, col2 = st.columns(2)

# =====================================================
# PREDICTED YOU
# =====================================================
with col1:
    st.subheader("📄 Predicted You (Essay-based)")

    predicted_text = st.text_area(
        "Paste a long text / essay about yourself:",
        height=250
    )

    if st.button("Analyze Predicted You"):
        if len(predicted_text.strip()) < 100:
            st.warning("Please enter a longer text for better accuracy.")
        else:
            pred, conf = predict_personality(predicted_text)

            st.markdown("### 🔍 Prediction")
            for trait in pred:
                st.write(
                    f"**{trait}**: {pred[trait]} "
                    f"(confidence: {conf[trait]})"
                )

            st.session_state.predicted_result = (pred, conf)

# =====================================================
# REAL YOU (CHATBOT STYLE)
# =====================================================
with col2:
    st.subheader("💬 Real You (Conversational)")

    if st.session_state.q_index < len(st.session_state.questions):
        question = st.session_state.questions[st.session_state.q_index]
        st.markdown(f"**❓ {question}**")

        answer = st.text_input("Your answer:", key=st.session_state.q_index)

        if st.button("Next"):
            st.session_state.real_text += " " + answer
            st.session_state.q_index += 1
            st.experimental_rerun()
    else:
        st.success("All questions answered!")

        pred, conf = predict_personality(st.session_state.real_text)

        st.markdown("### 🧠 Real You Profile")
        for trait in pred:
            st.write(
                f"**{trait}**: {pred[trait]} "
                f"(confidence: {conf[trait]})"
            )

        st.session_state.real_result = (pred, conf)

# =====================================================
# COMPARISON SECTION
# =====================================================
st.markdown("---")
st.subheader("📊 Comparison")

if "predicted_result" in st.session_state and "real_result" in st.session_state:
    p_pred, p_conf = st.session_state.predicted_result
    r_pred, r_conf = st.session_state.real_result

    for trait in p_pred:
        st.write(
            f"**{trait}** → "
            f"Predicted You: {p_pred[trait]} ({p_conf[trait]}) | "
            f"Real You: {r_pred[trait]} ({r_conf[trait]})"
        )
else:
    st.info("Complete both Predicted You and Real You to see comparison.")
