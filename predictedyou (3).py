import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(
    page_title="Predicted You vs Real You",
    layout="wide"
)

st.title("🧠 Predicted You vs Real You")
st.caption("A self-perception vs behavioral personality analysis")

# ---------------------------
# Load Models
# ---------------------------
@st.cache_resource
def load_models():
    with open("sentence_embedder.pkl", "rb") as f:
        embedder = pickle.load(f)

    models = {
        "E": pickle.load(open("E_label_classifier.pkl", "rb")),
        "I": pickle.load(open("I_label_classifier.pkl", "rb")),
        "N": pickle.load(open("N_label_classifier.pkl", "rb")),
        "T": pickle.load(open("T_label_classifier.pkl", "rb")),
    }

    return embedder, models


embedder, models = load_models()

# ---------------------------
# Questions
# ---------------------------
QUESTIONS = [
    "I feel confident speaking in front of a group.",
    "I prefer planning over spontaneity.",
    "I trust logic more than emotions while making decisions.",
    "I enjoy meeting new people.",
    "I often reflect deeply before acting.",
    "I stay calm under pressure.",
    "I like analyzing problems from multiple angles.",
    "I feel energized after social interactions."
]

# ---------------------------
# Sidebar – Instructions
# ---------------------------
st.sidebar.header("📝 Instructions")
st.sidebar.write(
    "Answer honestly. Your *Real You* is how you see yourself. "
    "Your *Predicted You* is inferred from your language patterns."
)

# ---------------------------
# Collect Answers
# ---------------------------
st.header("Answer the following questions")

user_answers = []
confidence_inputs = []

for i, q in enumerate(QUESTIONS):
    st.subheader(f"Q{i+1}. {q}")

    ans = st.text_input("Your response", key=f"ans_{i}")
    conf = st.slider("How confident are you in this answer?", 0, 100, 50, key=f"conf_{i}")

    user_answers.append(ans)
    confidence_inputs.append(conf)

# ---------------------------
# Process Button
# ---------------------------
if st.button("🔍 Analyze Me"):

    # Remove empty answers
    filtered = [(a, c) for a, c in zip(user_answers, confidence_inputs) if a.strip()]

    if len(filtered) < 3:
        st.error("Please answer at least 3 questions.")
        st.stop()

    texts, confs = zip(*filtered)

    # ---------------------------
    # Embeddings
    # ---------------------------
    embeddings = embedder.encode(list(texts))

    # ---------------------------
    # Predicted You (ML)
    # ---------------------------
    predicted_scores = []

    for key in ["E", "I", "N", "T"]:
        probs = models[key].predict_proba(embeddings)[:, 1]
        weighted = np.average(probs * 100, weights=confs)
        predicted_scores.append(weighted)

    # ---------------------------
    # Real You (Self-report)
    # ---------------------------
    real_scores = [
        np.mean(confs),
        100 - np.mean(confs),
        np.std(confs) * 5,
        np.mean(confs) * 0.8
    ]

    traits = [
        "Confidence",
        "Social Orientation",
        "Analytical Thinking",
        "Emotional Stability"
    ]

    # ---------------------------
    # Radar Chart
    # ---------------------------
    st.subheader("🕸 Personality Radar")

    radar = go.Figure()

    radar.add_trace(go.Scatterpolar(
        r=real_scores,
        theta=traits,
        fill='toself',
        name="Real You"
    ))

    radar.add_trace(go.Scatterpolar(
        r=predicted_scores,
        theta=traits,
        fill='toself',
        name="Predicted You"
    ))

    radar.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100])),
        showlegend=True
    )

    st.plotly_chart(radar, use_container_width=True)

    # ---------------------------
    # Confidence Stabilisation Curve
    # ---------------------------
    st.subheader("📈 Confidence Stabilisation")

    curve = go.Figure()
    curve.add_trace(go.Scatter(
        y=list(np.cumsum(confs) / np.arange(1, len(confs) + 1)),
        mode="lines+markers",
        name="Confidence Stability"
    ))

    curve.update_layout(
        xaxis_title="Question Progression",
        yaxis_title="Stabilized Confidence"
    )

    st.plotly_chart(curve, use_container_width=True)

    # ---------------------------
    # Personality Explanation
    # ---------------------------
    st.subheader("🧩 Who You Are As A Person")

    for t, r, p in zip(traits, real_scores, predicted_scores):
        diff = p - r

        if diff > 15:
            st.write(f"• You underestimate your **{t.lower()}**. Your responses show it’s stronger than you believe.")
        elif diff < -15:
            st.write(f"• You may **overestimate your {t.lower()}**, but your behavior suggests more nuance.")
        else:
            st.write(f"• Your self-perception of **{t.lower()}** is well aligned with your actual patterns.")

    st.success("Analysis complete ✨")
