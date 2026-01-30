# =====================================================
# Predicted You vs Real You — Streamlit App
# =====================================================

import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer

# =====================================================
# Page Config
# =====================================================
st.set_page_config(
    page_title="Predicted You vs Real You",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Predicted You vs Real You")

# =====================================================
# Load Models (cached to prevent reload + OOM)
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
# Inference Function
# =====================================================
def predict_personality(text: str):
    embedding = embedder.encode([text])

    prediction = {}
    confidence = {}

    for trait, model in models.items():
        pred = model.predict(embedding)[0]
        prob = model.predict_proba(embedding)[0][pred]

        prediction[trait] = "High" if pred == 1 else "Low"
        confidence[trait] = round(float(prob), 3)

    return prediction, confidence

# =====================================================
# Question Bank (Real You)
# =====================================================
QUESTION_BANK = [
    ("Extraversion", "Do you feel energized in social gatherings?"),
    ("Openness", "Do you enjoy exploring new ideas or philosophies?"),
    ("Conscientiousness", "How do you usually plan your daily tasks?"),
    ("Agreeableness", "How do you handle disagreements with others?")
]

# =====================================================
# Session State Initialization
# =====================================================
if "real_text" not in st.session_state:
    st.session_state.real_text = ""

if "q_index" not in st.session_state:
    st.session_state.q_index = 0

if "real_result" not in st.session_state:
    st.session_state.real_result = None

# =====================================================
# Layout
# =====================================================
col1, col2 = st.columns(2)

# =====================================================
# Predicted You (Essay-Based)
# =====================================================
with col1:
    st.subheader("📄 Predicted You (Essay-based)")

    essay_text = st.text_area(
        "Paste a long text / essay about yourself:",
        height=300
    )

    if st.button("Analyze Predicted You"):
        if len(essay_text.strip()) < 100:
            st.warning("Please provide a longer text for accurate prediction.")
        else:
            pred, conf = predict_personality(essay_text)

            st.markdown("### 🔍 Predicted You Results")
            for trait in pred:
                st.write(
                    f"**{trait}**: {pred[trait]} "
                    f"(confidence: {conf[trait]})"
                )

# =====================================================
# Real You (Conversational)
# =====================================================
with col2:
    st.subheader("💬 Real You (Conversational)")

    if st.session_state.q_index < len(QUESTION_BANK):
        trait, question = QUESTION_BANK[st.session_state.q_index]

        st.markdown(f"❓ **{question}**")
        answer = st.text_input("Your answer:", key=f"answer_{st.session_state.q_index}")

        if st.button("Next", key=f"next_{st.session_state.q_index}"):
            if answer.strip():
                st.session_state.real_text += " " + answer
                st.session_state.q_index += 1
                st.rerun()
            else:
                st.warning("Please enter an answer before continuing.")

    else:
        if st.session_state.real_result is None:
            pred, conf = predict_personality(st.session_state.real_text)
            st.session_state.real_result = (pred, conf)

        st.markdown("### 🎯 Real You Results")
        pred, conf = st.session_state.real_result

        for trait in pred:
            st.write(
                f"**{trait}**: {pred[trait]} "
                f"(confidence: {conf[trait]})"
            )

# =====================================================
# Comparison Section
# =====================================================
if st.session_state.real_result and essay_text.strip():
    st.markdown("---")
    st.subheader("🆚 Comparison: Predicted You vs Real You")

    pred_pred, pred_conf = predict_personality(essay_text)
    real_pred, real_conf = st.session_state.real_result

    for trait in pred_pred:
        st.write(
            f"**{trait}** → "
            f"Predicted You: {pred_pred[trait]} ({pred_conf[trait]}) | "
            f"Real You: {real_pred[trait]} ({real_conf[trait]})"
        )
