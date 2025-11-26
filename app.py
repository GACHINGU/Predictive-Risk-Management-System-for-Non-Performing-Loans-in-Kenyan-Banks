# =========================================
# STREAMLIT APP — FAINCRE: FUTURISTIC CREDIT DEFAULT DASHBOARD
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Faincre - Credit Default Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# STYLING: DARK BLUE FUTURISTIC
# ==========================
st.markdown(
    """
    <style>
    /* Dark background */
    .main {background-color: #0B1D51; color: #FFFFFF;}
    /* Sidebar */
    .css-18e3th9 {background-color: #0A1B48;}
    /* Headers */
    h1, h2, h3, h4, h5, h6 {color: #00E5FF;}
    /* Text */
    .stText, .stMarkdown {color: #FFFFFF;}
    /* Buttons */
    div.stButton > button {background-color: #005BBB; color: #FFFFFF; border-radius:10px;}
    div.stButton > button:hover {background-color: #00E5FF; color: #0B1D51;}
    /* Tables */
    .dataframe td, .dataframe th {color: #FFFFFF; background-color: #0A1B48;}
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================
# LOAD MODEL
# ==========================
@st.cache_data
def load_model():
    model = joblib.load("logistic_regression_best_model.pkl")  # change if top model differs
    return model

model = load_model()

# ==========================
# APP HEADER
# ==========================
st.title("💳 Faincre: Futuristic Credit Default Predictor")
st.markdown("""
Welcome to **Faincre**, your **next-generation credit risk dashboard**.  
Predict default probability instantly and explore interactive insights.
""")

# ==========================
# SIDEBAR: USER INPUT
# ==========================
st.sidebar.header("Faincre: Input Loan Applicant Features")

# Load features from a template csv
feature_template = pd.read_csv("cleaned_preprocessed_loans_data.csv").drop(columns=["loan_status_binary"]).head(1)

input_data = {}
for col in feature_template.columns:
    # Automatic numeric sliders
    min_val = float(feature_template[col].min())
    max_val = float(feature_template[col].max())
    mean_val = float(feature_template[col].mean())
    input_data[col] = st.sidebar.slider(f"{col}", min_val, max_val, mean_val)

# Convert to DataFrame for model
input_df = pd.DataFrame([input_data])

# ==========================
# PREDICTION
# ==========================
st.markdown("### Prediction 🔮")
predict_button = st.button("Predict Default Probability")

if predict_button:
    pred_proba = model.predict_proba(input_df)[:, 1][0]
    pred_class = model.predict(input_df)[0]

    # Display futuristic gauge
    st.markdown(f"""
    <div style="background-color:#00112A; padding:15px; border-radius:15px; text-align:center;">
        <h2 style="color:#00E5FF;">Default Probability</h2>
        <h1 style="color:#FF4D4D; font-size:48px;">{pred_proba*100:.2f}%</h1>
        <p style="color:#FFFFFF; font-size:20px;">Predicted Class: {"Default" if pred_class==1 else "Safe"}</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================
# DATA EXPLORATION (optional futuristic dashboard)
# ==========================
st.markdown("### 📊 Feature Distribution Insights")
expander = st.expander("Click to Explore Feature Distributions")
with expander:
    fig, ax = plt.subplots(figsize=(12,6))
    sample_data = feature_template.copy()
    sample_data = pd.concat([sample_data]*100)  # make it bigger for visualization
    sns.kdeplot(data=sample_data, fill=True, palette="cool", alpha=0.7)
    plt.title("Feature Distributions (Sample)", color="#00E5FF")
    plt.xlabel("Values", color="#FFFFFF")
    plt.ylabel("Density", color="#FFFFFF")
    plt.tick_params(colors="#FFFFFF")
    st.pyplot(fig)

# ==========================
# FOOTER
# ==========================
st.markdown("""
---
<div style="text-align:center; color:#00E5FF;">
Made with ❤️ using **Faincre** & Streamlit | Futuristic Dashboard
</div>
""", unsafe_allow_html=True)