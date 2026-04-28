import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# 1. PAGE CONFIG — must be first streamlit call
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Kerala Flood Risk Predictor",
    page_icon="🌧️",
    layout="centered"
)

# ─────────────────────────────────────────
# 2. LOAD MODEL — @st.cache_resource means
#    it loads ONCE and reuses every time
#    the user interacts. Without this,
#    it reloads the model on every click.
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("flood_risk_model.pkl")
    with open("features.json") as f:
        features = json.load(f)
    return model, features

model, FEATURES = load_model()

# ─────────────────────────────────────────
# 3. LOAD HISTORICAL DATA for context
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("kerala_features_v2.csv")

kerala = load_data()

# ─────────────────────────────────────────
# 4. HEADER
# ─────────────────────────────────────────
st.title("🌧️ Kerala Flood Risk Predictor")
st.markdown(
    "Enter seasonal rainfall values to predict flood risk level. "
    "Model trained on **115 years** of Kerala meteorological data (1901–2015)."
)
st.divider()

# ─────────────────────────────────────────
# 5. INPUT SECTION
#    st.columns splits the page into columns
#    st.number_input creates a number field
# ─────────────────────────────────────────
st.subheader("Enter Rainfall Values (mm)")

col1, col2 = st.columns(2)

with col1:
    monsoon = st.number_input(
        "Monsoon total (Jun–Sep)",
        min_value=0.0, max_value=5000.0,
        value=2200.0, step=10.0,
        help="Sum of June, July, August, September rainfall"
    )
    pre_monsoon = st.number_input(
        "Pre-Monsoon total (Mar–May)",
        min_value=0.0, max_value=1000.0,
        value=300.0, step=10.0
    )
    post_monsoon = st.number_input(
        "Post-Monsoon total (Oct–Nov)",
        min_value=0.0, max_value=1000.0,
        value=280.0, step=10.0
    )
    winter = st.number_input(
        "Winter total (Dec–Feb)",
        min_value=0.0, max_value=500.0,
        value=70.0, step=5.0
    )

with col2:
    peak_month = st.number_input(
        "Peak month rainfall (mm)",
        min_value=0.0, max_value=2000.0,
        value=700.0, step=10.0,
        help="Highest single month rainfall"
    )
    monsoon_std = st.number_input(
        "Monsoon variability (std dev)",
        min_value=0.0, max_value=500.0,
        value=100.0, step=5.0,
        help="How unevenly spread the monsoon rain was"
    )
    prev_year = st.number_input(
        "Previous year total (mm)",
        min_value=0.0, max_value=6000.0,
        value=2900.0, step=10.0
    )

# ─────────────────────────────────────────
# 6. DERIVED FEATURES
#    Calculate monsoon ratio automatically
# ─────────────────────────────────────────
annual_total = monsoon + pre_monsoon + post_monsoon + winter
monsoon_ratio = monsoon / annual_total if annual_total > 0 else 0

st.caption(f"📊 Calculated annual total: **{annual_total:.0f} mm** "
           f"| Monsoon ratio: **{monsoon_ratio:.1%}**")

st.divider()

# ─────────────────────────────────────────
# 7. PREDICTION BUTTON
# ─────────────────────────────────────────
if st.button("🔍 Predict Flood Risk", type="primary", use_container_width=True):

    # Build input in the exact order the model expects
    input_data = pd.DataFrame([{
        "MONSOON":          monsoon,
        "PRE_MONSOON":      pre_monsoon,
        "POST_MONSOON":     post_monsoon,
        "WINTER":           winter,
        "MONSOON_RATIO":    monsoon_ratio,
        "PREV_YEAR_RAIN":   prev_year,
        "PEAK_MONTH_RAIN":  peak_month,
        "MONSOON_STD":      monsoon_std
    }])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_

    # ─────────────────────────────────────
    # 8. DISPLAY RESULT
    # ─────────────────────────────────────
    if prediction == "High":
        st.error(f"🔴 Flood Risk: **HIGH**")
    elif prediction == "Medium":
        st.warning(f"🟡 Flood Risk: **MEDIUM**")
    else:
        st.success(f"🟢 Flood Risk: **LOW**")

    # Confidence bar chart
    st.subheader("Model Confidence")
    prob_df = pd.DataFrame({
        "Risk Level": classes,
        "Probability": probabilities
    }).sort_values("Probability", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 2.5))
    colors = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"}
    bars = ax.barh(
        prob_df["Risk Level"],
        prob_df["Probability"],
        color=[colors[c] for c in prob_df["Risk Level"]],
        edgecolor="white", height=0.4
    )
    for bar, val in zip(bars, prob_df["Probability"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.1%}", va="center", fontsize=11)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Confidence")
    ax.spines[["top","right"]].set_visible(False)
    st.pyplot(fig)
    plt.close()

    # ─────────────────────────────────────
    # 9. HISTORICAL CONTEXT
    #    Show where this year sits in history
    # ─────────────────────────────────────
    st.subheader("How does this compare to history?")

    fig2, ax2 = plt.subplots(figsize=(10, 3))
    risk_colors = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"}

    for risk, group in kerala.groupby("FLOOD_RISK"):
        ax2.scatter(group["YEAR"], group["MONSOON"],
                    color=risk_colors[risk], label=risk,
                    alpha=0.6, s=30, zorder=3)

    # Plot the user's input as a star
    ax2.scatter(2025, monsoon, color="black",
                marker="*", s=250, zorder=5,
                label="Your input")

    ax2.set_title("Your monsoon total vs 115 years of Kerala history")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Monsoon rainfall (mm)")
    ax2.legend(title="Flood Risk", bbox_to_anchor=(1.01, 1))
    ax2.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ─────────────────────────────────────────
# 10. FOOTER
# ─────────────────────────────────────────
st.divider()
st.caption(
    "Built with Scikit-learn & Streamlit · "
    "Data: India Meteorological Department (1901–2015) · "
    "Model: Gradient Boosting (F1: 0.847)"
)