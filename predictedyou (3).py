# =====================================================
# Predicted You vs Real You — Advanced Streamlit App
# =====================================================

import streamlit as st
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go

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
# Load Models (cached = memory safe)
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
# Question Bank (3 per trait)
# =====================================================
QUESTION_BANK = [
    ("Extraversion", "Do you feel energized in social gatherings?"),
    ("Extraversion", "How often do you seek conversations with new people?"),
    ("Extraversion", "Do you prefer working in teams or alone?"),

    ("Openness", "Do you enjoy exploring abstract ideas?"),
    ("Openness", "How do you react to unfamiliar experiences?"),
    ("Openness", "Do you like creative or artistic activities?"),

    ("Conscientiousness", "How do you usually plan your daily tasks?"),
    ("Conscientiousness", "How important are deadlines to you?"),
    ("Conscientiousness", "Do you stay organized under pressure?"),

    ("Agreeableness", "How do you handle disagreements with others?"),
    ("Agreeableness", "Do you empathize easily with people?"),
    ("Agreeableness", "Do you prioritize harmony in relationships?")
]

# =====================================================
# Session State
# =====================================================
if "real_text" not in st.session_state:
    st.session_state.real_text = ""

if "q_index" not in st.session_state:
    st.session_state.q_index = 0

if "real_result" not in st.session_state:
    st.session_state.real_result = None

# =====================================================
# Dynamic Confidence Stabilization
# =====================================================
def stabilize_confidence(raw_conf, text_length):
    """
    Penalize short text, stabilize confidence for longer inputs
    """
    length_factor = min(1.0, text_length / 600)
    stabilized = 0.5 + (raw_conf - 0.5) * length_factor
    return round(stabilized, 3)

# =====================================================
# Inference
# =====================================================
def predict_personality(text: str):
    embedding = embedder.encode([text])

    prediction = {}
    confidence = {}

    for trait, model in models.items():
        pred = model.predict(embedding)[0]
        raw_prob = model.predict_proba(embedding)[0][pred]

        stabilized = stabilize_confidence(raw_prob, len(text))

        prediction[trait] = "High" if pred == 1 else "Low"
        confidence[trait] = stabilized

    return prediction, confidence

# =====================================================
# Radar Chart
# =====================================================
def personality_radar(pred_conf, real_conf):
    traits = list(pred_conf.keys())

    pred_values = list(pred_conf.values())
    real_values = list(real_conf.values())

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=pred_values,
        theta=traits,
        fill='toself',
        name='Predicted You'
    ))

    fig.add_trace(go.Scatterpolar(
        r=real_values,
        theta=traits,
        fill='toself',
        name='Real You'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=500
    )

    return fig

# =====================================================
# Layout
# =====================================================
col1, col2 = st.columns(2)

# =====================================================
# Predicted You (Essay)
# =====================================================
with col1:
    st.subheader("📄 Predicted You (Essay-based)")

    essay = st.text_area(
        "Write an essay / long text about yourself:",
        height=320
    )

    predicted_result = None

    if st.button("Analyze Predicted You"):
        if len(essay.strip()) < 150:
            st.warning("Please write at least 150 characters.")
        else:
            predicted_result = predict_personality(essay)
            pred_pred, pred_conf = predicted_result

            st.markdown("### 🔍 Predicted You")
            for trait in pred_pred:
                st.write(f"**{trait}**: {pred_pred[trait]} (conf: {pred_conf[trait]})")

# =====================================================
# Real You (Conversational)
# =====================================================
with col2:
    st.subheader("💬 Real You (Conversational)")

    if st.session_state.q_index < len(QUESTION_BANK):
        trait, question = QUESTION_BANK[st.session_state.q_index]

        st.markdown(f"❓ **{question}**")
        answer = st.text_input("Your answer:", key=f"a_{st.session_state.q_index}")

        if st.button("Next"):
            if answer.strip():
                st.session_state.real_text += " " + answer
                st.session_state.q_index += 1
                st.rerun()
            else:
                st.warning("Please answer before continuing.")
    else:
        if st.session_state.real_result is None:
            st.session_state.real_result = predict_personality(st.session_state.real_text)

        real_pred, real_conf = st.session_state.real_result

        st.markdown("### 🎯 Real You")
        for trait in real_pred:
            st.write(f"**{trait}**: {real_pred[trait]} (conf: {real_conf[trait]})")

# =====================================================
# Comparison + Radar
# =====================================================
if st.session_state.real_result and predicted_result:
    st.markdown("---")
    st.subheader("🆚 Personality Comparison")

    real_pred, real_conf = st.session_state.real_result
    pred_pred, pred_conf = predicted_result

    for trait in pred_pred:
        st.write(
            f"**{trait}** → "
            f"Predicted: {pred_pred[trait]} ({pred_conf[trait]}) | "
            f"Real: {real_pred[trait]} ({real_conf[trait]})"
        )

    st.markdown("### 📊 Personality Radar")
    st.plotly_chart(personality_radar(pred_conf, real_conf), use_container_width=True)
