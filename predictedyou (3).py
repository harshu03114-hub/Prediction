import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Predicted You vs Real You",
    layout="wide"
)

st.title("🧠 Predicted You vs Real You")
st.caption("Essay-based prediction vs conversational personality discovery")

# --------------------------------------------------
# LOAD MODELS (SAFE)
# --------------------------------------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    models = {
        "E": pickle.load(open("E_label_classifier.pkl", "rb")),
        "I": pickle.load(open("I_label_classifier.pkl", "rb")),
        "N": pickle.load(open("N_label_classifier.pkl", "rb")),
        "T": pickle.load(open("T_label_classifier.pkl", "rb")),
    }
    return embedder, models

embedder, models = load_models()

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def predict_personality(text):
    vec = embedder.encode([text])
    scores = {}

    for trait, model in models.items():
        prob = model.predict_proba(vec)[0][1]
        scores[trait] = prob

    return scores


def stabilize_scores(scores):
    """Confidence stabilisation to avoid extreme jumps"""
    stabilized = {}
    for k, v in scores.items():
        stabilized[k] = 0.7 * v + 0.3 * 0.5
    return stabilized


def radar_chart(predicted, real):
    labels = ["Extroversion", "Introversion", "Intuition", "Thinking"]
    pred_values = [
        predicted["E"],
        1 - predicted["E"],
        predicted["N"],
        predicted["T"]
    ]
    real_values = [
        real["E"],
        1 - real["E"],
        real["N"],
        real["T"]
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=pred_values,
        theta=labels,
        fill='toself',
        name="Predicted You"
    ))

    fig.add_trace(go.Scatterpolar(
        r=real_values,
        theta=labels,
        fill='toself',
        name="Real You"
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True
    )

    return fig


def explain_personality(pred, real):
    explanation = []

    if pred["E"] > real["E"]:
        explanation.append(
            "You **appear more outgoing in writing** than you feel internally. "
            "You may express confidence intellectually but conserve social energy."
        )
    else:
        explanation.append(
            "You are **more socially expressive in real interactions** than your writing suggests."
        )

    if pred["N"] > 0.6 and real["N"] > 0.6:
        explanation.append(
            "You are strongly **intuitive and future-oriented**, preferring concepts over routines."
        )

    if pred["T"] > real["T"]:
        explanation.append(
            "You rely on **structured logic when reflecting**, but emotionally calibrate decisions in real life."
        )

    explanation.append(
        "Overall, you show a **high self-awareness gap** — you understand yourself deeply, "
        "but regulate how much of it you externally project."
    )

    return "\n\n".join(explanation)

# --------------------------------------------------
# UI LAYOUT
# --------------------------------------------------
left, right = st.columns(2)

# ----------------- PREDICTED YOU ------------------
with left:
    st.subheader("📄 Predicted You (Essay-based)")
    essay = st.text_area(
        "Paste a long text / essay about yourself:",
        height=280
    )

    analyze_essay = st.button("Analyze Predicted You")

# ----------------- REAL YOU -----------------------
with right:
    st.subheader("💬 Real You (Conversational)")

    questions = [
        "Do you feel energized in social gatherings?",
        "Do you rely more on intuition than facts?",
        "Do you prefer logic over emotions when deciding?",
        "Do you enjoy exploring abstract ideas?",
    ]

    answers = []
    for q in questions:
        answers.append(st.text_input(q, key=q))

    analyze_real = st.button("Analyze Real You")

# --------------------------------------------------
# PROCESS RESULTS
# --------------------------------------------------
if analyze_essay and analyze_real:
    if essay.strip() == "" or any(a.strip() == "" for a in answers):
        st.warning("Please complete both sections.")
    else:
        predicted_scores = stabilize_scores(
            predict_personality(essay)
        )

        real_text = " ".join(answers)
        real_scores = stabilize_scores(
            predict_personality(real_text)
        )

        st.subheader("📊 Personality Radar Comparison")
        fig = radar_chart(predicted_scores, real_scores)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🧠 Who You Are (Interpretation)")
        st.markdown(explain_personality(predicted_scores, real_scores))
