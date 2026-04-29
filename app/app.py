import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import time

# ─────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Kerala Flood Risk Prediction System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────
# 2. CUSTOM CSS (Apple/Google Style, Kerala Theme)
# ─────────────────────────────────────────
st.markdown("""
<style>
/* Import Inter Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #f8fafc;
}

/* Hide default streamlit header/footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main App Container */
.block-container {
    padding-top: 0rem !important;
    padding-bottom: 0rem !important;
    padding-left: 0rem !important;
    padding-right: 0rem !important;
    max-width: 100% !important;
}

/* Hero Section */
.hero-container {
    background: linear-gradient(rgba(17, 94, 89, 0.85), rgba(2, 132, 199, 0.75)), url("https://images.unsplash.com/photo-1593693397690-362cb9666fc2?q=80&w=2069&auto=format&fit=crop") center/cover;
    padding: 7rem 2rem;
    text-align: center;
    color: white;
    margin-bottom: 3rem;
}
.hero-title {
    font-size: 3.5rem;
    font-weight: 700;
    margin-bottom: 1rem;
    letter-spacing: -0.02em;
}
.hero-subtitle {
    font-size: 1.2rem;
    font-weight: 300;
    max-width: 800px;
    margin: 0 auto;
    opacity: 0.95;
    line-height: 1.6;
}

/* Inputs */
.stNumberInput > div > div > input {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    color: #334155;
    padding: 0.5rem 1rem;
    font-weight: 500;
}
.stNumberInput > div > div > input:focus {
    border-color: #0d9488;
    box-shadow: 0 0 0 2px rgba(13, 148, 136, 0.2);
}

/* Button */
.stButton > button {
    width: 100%;
    background-color: #0d9488 !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 0.8rem 1.5rem !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    border: none !important;
    box-shadow: 0 4px 6px -1px rgba(13, 148, 136, 0.3) !important;
    transition: all 0.2s ease-in-out !important;
    margin-top: 1rem;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(13, 148, 136, 0.4) !important;
    background-color: #0f766e !important;
}

/* Subheaders */
h2, h3 {
    color: #0f172a !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
}
p, .stMarkdown {
    color: #475569;
}

/* Info Box */
.stAlert {
    border-radius: 12px;
    border: none;
    background-color: #f0fdf4;
    color: #166534;
}

/* Custom Result Card Styles */
.result-card {
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    color: white;
    margin-top: 1rem;
    margin-bottom: 2rem;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
    animation: fadeIn 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}
.result-low { background: linear-gradient(135deg, #10b981, #059669); }
.result-med { background: linear-gradient(135deg, #f59e0b, #d97706); }
.result-high { background: linear-gradient(135deg, #ef4444, #dc2626); }

.result-title { font-size: 1.2rem; font-weight: 500; margin-bottom: 0.5rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.05em;}
.result-value { font-size: 3.5rem; font-weight: 700; letter-spacing: -0.02em; }

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Footer */
.custom-footer {
    text-align: center;
    padding: 4rem 2rem;
    margin-top: 5rem;
    background-color: #f1f5f9;
    color: #64748b;
    font-size: 0.95rem;
}
.custom-footer a {
    color: #0d9488;
    text-decoration: none;
    font-weight: 500;
}
.custom-footer a:hover {
    text-decoration: underline;
}

/* Expander Tweaks */
.streamlit-expanderHeader {
    font-weight: 600;
    color: #0f172a;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# 3. LOAD MODEL & DATA
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("flood_risk_model.pkl")
    with open("features.json") as f:
        features = json.load(f)
    return model, features

model, FEATURES = load_model()

@st.cache_data
def load_data():
    return pd.read_csv("kerala_features_v2.csv")

kerala = load_data()

# ─────────────────────────────────────────
# 4. HERO SECTION
# ─────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Kerala Flood Risk Prediction System</div>
    <div class="hero-subtitle">An advanced machine learning platform leveraging 115 years of meteorological data to provide highly accurate flood vulnerability forecasts for Kerala, India.</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# 5. MAIN CONTENT WRAPPER
# ─────────────────────────────────────────
# Use Streamlit columns to constrain the width for desktop, while remaining responsive for mobile
spacer_left, main_content, spacer_right = st.columns([1, 6, 1])

with main_content:
    
    # --- INPUT PANEL ---
    st.markdown("### 📊 Environmental Parameters", unsafe_allow_html=True)
    st.markdown("Adjust the seasonal rainfall parameters below to simulate different meteorological conditions.", unsafe_allow_html=True)
    st.write("") # spacing
    
    # Inputs using a 2-column layout
    col1, col2 = st.columns(2)
    
    with col1:
        monsoon = st.number_input("🌧️ Monsoon (Jun–Sep) mm", min_value=0.0, max_value=5000.0, value=2200.0, step=10.0, help="Total rainfall during the core monsoon season")
        pre_monsoon = st.number_input("⛅ Pre-Monsoon (Mar–May) mm", min_value=0.0, max_value=1000.0, value=300.0, step=10.0)
        post_monsoon = st.number_input("🌩️ Post-Monsoon (Oct–Nov) mm", min_value=0.0, max_value=1000.0, value=280.0, step=10.0)
        winter = st.number_input("❄️ Winter (Dec–Feb) mm", min_value=0.0, max_value=500.0, value=70.0, step=5.0)

    with col2:
        peak_month = st.number_input("📈 Peak Month Rainfall mm", min_value=0.0, max_value=2000.0, value=700.0, step=10.0, help="Highest single month rainfall recorded")
        monsoon_std = st.number_input("📉 Monsoon Variability (Std Dev)", min_value=0.0, max_value=500.0, value=100.0, step=5.0, help="Standard deviation indicating how unevenly rain was spread")
        prev_year = st.number_input("📅 Previous Year Total mm", min_value=0.0, max_value=6000.0, value=2900.0, step=10.0)
        
        # Derived values calculated automatically
        annual_total = monsoon + pre_monsoon + post_monsoon + winter
        monsoon_ratio = monsoon / annual_total if annual_total > 0 else 0
        
        st.write("")
        st.info(f"**Annual Total:** {annual_total:.0f} mm &nbsp; • &nbsp; **Monsoon Ratio:** {monsoon_ratio:.1%}")

    st.write("")
    st.write("")
    
    # --- PREDICTION ACTION ---
    predict_clicked = st.button("Generate Flood Risk Assessment")
    
    if predict_clicked:
        # UX Improvement: Show spinner to simulate processing time
        with st.spinner("Analyzing historical patterns and running ML inference..."):
            time.sleep(1.2)
            
            # Prepare data
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

            # Run prediction
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            classes = model.classes_
            
            # --- RESULT DISPLAY ---
            if prediction == "High":
                css_class = "result-high"
                icon = "⚠️"
            elif prediction == "Medium":
                css_class = "result-med"
                icon = "⚡"
            else:
                css_class = "result-low"
                icon = "✅"
                
            st.markdown(f"""
            <div class="result-card {css_class}">
                <div class="result-title">Predicted Risk Level</div>
                <div class="result-value">{icon} {prediction.upper()}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # --- VISUALIZATIONS ---
            st.write("---")
            st.markdown("### 📈 Analytics & Context", unsafe_allow_html=True)
            st.write("")
            
            v_col1, v_col2 = st.columns(2)
            
            # Chart 1: Probabilities
            with v_col1:
                st.markdown("##### Model Confidence")
                prob_df = pd.DataFrame({
                    "Risk Level": classes,
                    "Probability": probabilities
                }).sort_values("Probability", ascending=True)

                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_facecolor('#f8fafc')
                ax.set_facecolor('#f8fafc')
                
                chart_colors = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}
                bars = ax.barh(
                    prob_df["Risk Level"],
                    prob_df["Probability"],
                    color=[chart_colors[c] for c in prob_df["Risk Level"]],
                    edgecolor="none", height=0.5
                )
                
                # Add percentage labels
                for bar, val in zip(bars, prob_df["Probability"]):
                    ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                            f"{val:.1%}", va="center", fontsize=11, color="#334155", fontweight="bold")
                
                ax.set_xlim(0, 1.15)
                ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
                ax.tick_params(axis='both', which='both', length=0)
                ax.set_xticks([])
                ax.yaxis.set_tick_params(labelsize=11, labelcolor="#334155")
                st.pyplot(fig)
                plt.close()

            # Chart 2: Historical Scatter
            with v_col2:
                st.markdown("##### Historical Context (1901-2015)")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                fig2.patch.set_facecolor('#f8fafc')
                ax2.set_facecolor('#ffffff')
                
                # Plot historical data
                for risk, group in kerala.groupby("FLOOD_RISK"):
                    ax2.scatter(group["YEAR"], group["MONSOON"],
                                color=chart_colors[risk], label=risk,
                                alpha=0.5, s=40, edgecolors='none')

                # Plot user input
                ax2.scatter(2025, monsoon, color="#0f172a",
                            marker="*", s=250, zorder=5,
                            label="Current Input")

                # Clean up axes
                ax2.spines[["top", "right"]].set_visible(False)
                ax2.spines[["bottom", "left"]].set_color('#cbd5e1')
                ax2.tick_params(colors="#64748b")
                ax2.set_xlabel("Year", color="#64748b", fontsize=10)
                ax2.set_ylabel("Monsoon Rain (mm)", color="#64748b", fontsize=10)
                
                legend = ax2.legend(title="Risk", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, fontsize=9)
                legend.get_title().set_color("#64748b")
                
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()

    st.write("---")
    st.write("")
    
    # --- ABOUT SECTION ---
    with st.expander("ℹ️ About the Prediction Model"):
        st.markdown("""
        **Methodology**  
        This system utilizes a **Gradient Boosting Classifier** trained on historical meteorological data from the India Meteorological Department, covering 115 years (1901–2015). It evaluates seasonal rainfall patterns, distribution variability, and multi-year cumulative effects to accurately assess flood vulnerability.
        
        **Technology Stack**
        - **Machine Learning**: Scikit-learn (Gradient Boosting)
        - **Frontend**: Streamlit with custom CSS overriding
        - **Data Processing**: Pandas & NumPy
        
        **Model Performance Metrics**
        - **F1 Score**: 0.847
        - **Cross-Validation**: 5-fold Stratified
        """)

# ─────────────────────────────────────────
# 6. FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div class="custom-footer">
    <div style="max-width: 800px; margin: 0 auto;">
        <h4 style="color: #334155; margin-bottom: 1rem;">Kerala Flood Risk Assessment Initiative</h4>
        <p style="margin-bottom: 0.5rem;">Providing actionable insights for disaster preparedness and climate resilience.</p>
        <p style="margin-bottom: 2rem;">Powered by <a href="https://streamlit.io" target="_blank">Streamlit</a> & Machine Learning.</p>
        <div>
            <a href="https://github.com/Karthik-Unni/kerala-flood-predictor" target="_blank">GitHub Repository</a> &nbsp;&bull;&nbsp; 
            <a href="#">Technical Documentation</a> &nbsp;&bull;&nbsp; 
            <a href="#">Contact Support</a>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)