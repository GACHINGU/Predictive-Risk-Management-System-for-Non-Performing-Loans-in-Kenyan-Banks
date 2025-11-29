# =========================================
# STREAMLIT APP — FAINCRE: FUTURISTIC CREDIT DEFAULT DASHBOARD (V2.0)
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shap 

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Faincre - Credit Default Predictor (V2.0)",
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
    div.stButton > button {background-color: #005BBB; color: #FFFFFF; border-radius:10px; border: 1px solid #005BBB;}
    div.stButton > button:hover {background-color: #00E5FF; color: #0B1D51; border: 1px solid #00E5FF;}
    /* Prediction Box Styling */
    .prediction-box {
        background-color:#00112A; 
        padding:20px; 
        border-radius:15px; 
        text-align:center; 
        border: 2px solid #00E5FF;
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.5);
    }
    .risk-high {color: #FF4D4D;}
    .risk-low {color: #4DFF9A;}
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================
# LOAD MODEL and DATA (Fixing Type Errors for SHAP)
# ==========================
@st.cache_resource
def load_resources():
    model_path = os.path.join("Trained_Models", "logistic_regression_best_model.pkl")
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'logistic_regression_best_model.pkl' is in the 'Trained_Models' directory.")
        return None, None, None

    try:
        # Load template data
        data = pd.read_csv("cleaned_preprocessed_loans_data.csv")
        feature_template = data.drop(columns=["loan_status_binary"], errors='ignore').copy()
        feature_template.columns = [c.replace(" ", "_").replace("-", "_") for c in feature_template.columns]
        
        # *** FIX 1: ENFORCE NUMERIC TYPES FOR SHAP BACKGROUND DATA ***
        # Convert all columns to numeric, coercing errors (non-numbers) to NaN, then fill NaNs with 0.
        feature_template = feature_template.apply(pd.to_numeric, errors='coerce').fillna(0)
        
    except FileNotFoundError:
        st.error("Template data file ('cleaned_preprocessed_loans_data.csv') not found.")
        return None, None, None
    
    # Use 100 random rows for the SHAP background data
    n_background_samples = min(100, len(feature_template))
    background_data = feature_template.head(n_background_samples)
    
    # Use only the first row for slider calculation
    input_template = feature_template.head(1)
        
    # Initialize SHAP Explainer (Corrected for Logistic Regression output)
    explainer = None
    try:
        explainer = shap.Explainer(
            model.predict_proba, 
            background_data, 
            output_names=['Safe (0)', 'Default (1)'] 
        )
        st.success("SHAP Explainer initialized successfully! XAI features are enabled.")
    except Exception as e:
        st.warning(f"Could not initialize SHAP Explainer: {e}. XAI features will be disabled.")
        
    return model, input_template, explainer

model, feature_template_base, explainer = load_resources()

if model is None or feature_template_base is None:
    st.stop()

# ==========================
# APP HEADER
# ==========================
st.title("💳 Faincre: Futuristic Credit Default Predictor")
st.markdown("""
Welcome to **Faincre**, your **next-generation credit risk dashboard**. 
Predict default probability instantly, understand the risk drivers, and simulate 'what-if' scenarios.
""")



## Input Loan Applicant Features

# ==========================
# SIDEBAR: USER INPUT
# ==========================
st.sidebar.header("Input Loan Applicant Features")

input_data = {}
for col in feature_template_base.columns:
    min_val = float(feature_template_base[col].min())
    max_val = float(feature_template_base[col].max())
    mean_val = float(feature_template_base[col].mean())
    
    # Handle case when min == max
    if min_val == max_val:
        input_data[col] = st.sidebar.number_input(f"**{col}**", value=min_val, format="%.2f")
    else:
        # Use mean_val as default unless it's out of bounds due to rounding/float issues
        default_val = max(min_val, min(max_val, mean_val))
        input_data[col] = st.sidebar.slider(f"**{col}**", min_val, max_val, default_val, format="%.2f")

# Convert to DataFrame
initial_input_df = pd.DataFrame([input_data])


# ==========================
# RISK THRESHOLD SETTING
# ==========================
st.sidebar.markdown("---")
st.sidebar.header("Risk Management")
# New Feature: Risk Threshold Selector
default_threshold = 0.5
risk_threshold = st.sidebar.slider(
    "Decision Threshold (P > Threshold = Default)", 
    min_value=0.01, max_value=0.99, value=default_threshold, step=0.01
)
st.sidebar.info(f"Applicants with Default Probability > **{risk_threshold*100:.1f}%** are flagged as 'Default'.")



## Prediction and Explainability

# ==========================
# PREDICTION LOGIC
# ==========================
st.markdown("### Prediction & Explainability")
predict_button = st.button("Analyze Loan Applicant Risk")

if 'prediction_run' not in st.session_state:
    st.session_state.prediction_run = False
    st.session_state.current_proba = None
    st.session_state.current_input = initial_input_df.copy()

# Use the initial input from the sidebar if the button is pressed
if predict_button:
    st.session_state.prediction_run = True
    st.session_state.current_input = initial_input_df.copy()

if st.session_state.prediction_run:
    try:
        # Use the stored/updated input_df from session state for analysis
        current_input = st.session_state.current_input.copy()
        
        # *** FIX 2: ENFORCE NUMERIC TYPES FOR LIVE INPUT DATA ***
        current_input = current_input.apply(pd.to_numeric, errors='coerce').fillna(0)

        pred_proba = model.predict_proba(current_input)[:, 1][0]
        st.session_state.current_proba = pred_proba
        
        pred_class = 1 if pred_proba > risk_threshold else 0
        risk_color = "risk-high" if pred_proba > risk_threshold else "risk-low"

        # Display Prediction
        st.markdown(f"""
        <div class="prediction-box">
            <h2 style="color:#00E5FF;">Calculated Default Probability</h2>
            <h1 class="{risk_color}" style="font-size:52px;">{pred_proba*100:.2f}%</h1>
            <p style="color:#FFFFFF; font-size:20px;">
                **Predicted Status:** <span class="{risk_color}">**{'DEFAULT' if pred_class==1 else 'SAFE'}**</span> 
                (Threshold: {risk_threshold*100:.1f}%)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # --- SHAP EXPLAINABILITY ---
        if explainer is not None:
            st.markdown("### Feature Impact Breakdown (XAI)")
            st.markdown("The following chart shows how each feature value contributes to the final prediction.")

            # Calculate SHAP values for the current input
            shap_values = explainer(current_input)
            
            # Use the SHAP plot method (selecting class 1, which is 'Default')
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.waterfall_plot(shap_values[0, :, 1], max_display=10, show=False)
            

            # Customize plot for futuristic theme
            plt.title("Loan Applicant Risk Factors", color="#00E5FF")
            plt.xlabel("SHAP Value (Risk Contribution)", color="#FFFFFF")
            plt.tick_params(colors="#FFFFFF")
            ax.set_facecolor('#0B1D51')
            fig.patch.set_facecolor('#0B1D51')
            
            st.pyplot(fig)
            st.markdown("""
            > **Interpretation:** Features pushing the value **right** increase the probability of **Default** (Risk). 
            Features pushing the value **left** decrease the probability of Default (Safety).
            """)
        
    except ValueError as e:
        st.error(f"Prediction failed: {e}")



## Interactive Risk Simulation

# ==========================
# INTERACTIVE RISK SIMULATION
# ==========================
if st.session_state.prediction_run and st.session_state.current_proba is not None:
    st.markdown("### Interactive Risk Simulation (What-If Analysis)")
    st.markdown("Modify key features below to see how the default probability changes *instantly*.")
    
    current_input = st.session_state.current_input.copy()
    
    # Identify top 5 most important features for the simulation
    top_features = feature_template_base.columns[:5].tolist() # Fallback

    if explainer is not None:
        try:
            # Recalculate SHAP for current input to ensure accuracy for sorting
            # FIX: Ensure current_input is numeric before passing to explainer
            current_input_clean = current_input.apply(pd.to_numeric, errors='coerce').fillna(0)

            shap_values_sim = explainer(current_input_clean)
            # Get absolute SHAP values for class 1 (default) and sort
            feature_impact = pd.Series(np.abs(shap_values_sim[0, :, 1].values), index=current_input_clean.columns)
            top_features = feature_impact.sort_values(ascending=False).head(5).index.tolist()
        except Exception:
            pass

    
    # Create columns for simulation inputs
    sim_cols = st.columns(len(top_features))
    simulated_data = st.session_state.current_input.copy()
    
    # Collect simulation inputs
    for i, col in enumerate(top_features):
        min_val = float(feature_template_base[col].min())
        max_val = float(feature_template_base[col].max())
        # Use current value from session state
        current_val = float(st.session_state.current_input[col].iloc[0]) 

        with sim_cols[i]:
            new_val = st.number_input(
                f"**{col}**", 
                min_value=min_val, 
                max_value=max_val, 
                value=current_val, 
                key=f"sim_input_{col}", 
                format="%.2f"
            )
            simulated_data[col] = new_val

    # Run Simulation
    # *** FIX 3: ENFORCE NUMERIC TYPES FOR SIMULATION DATA ***
    simulated_data_clean = simulated_data.apply(pd.to_numeric, errors='coerce').fillna(0)

    sim_proba = model.predict_proba(simulated_data_clean)[:, 1][0]
    
    # Display Simulation Results
    original_proba = st.session_state.current_proba
    change_in_risk = sim_proba - original_proba
    
    # Display comparison
    col_orig, col_sim = st.columns(2)
    
    with col_orig:
        st.metric(
            label="Original Default Probability", 
            value=f"{original_proba*100:.2f}%",
            delta_color="off"
        )
    
    with col_sim:
        st.metric(
            label="Simulated Default Probability", 
            value=f"{sim_proba*100:.2f}%",
            delta=f"{change_in_risk*100:.2f}%",
            # delta_color="inverse" means green for negative change (risk decrease)
            delta_color="inverse" if change_in_risk < 0 else "normal"
        )
    
    if st.button("Lock-in Simulation Values"):
        st.session_state.current_input = simulated_data
        st.session_state.current_proba = sim_proba
        st.rerun() # Rerun to update the main prediction box and SHAP plot



## Feature Distribution Insights

# ==========================
# DATA EXPLORATION (Kept from original)
# ==========================
st.markdown("### Feature Distribution Insights")
expander = st.expander("Click to Explore Feature Distributions (Sample Data)")
with expander:
    fig, ax = plt.subplots(figsize=(12,6))
    sample_data = feature_template_base.head(100).melt() 
    
    sns.kdeplot(data=sample_data, x='value', hue='variable', fill=False, palette="cool", alpha=0.7)
    
    plt.title("Feature Distributions (Sample)", color="#00E5FF")
    plt.xlabel("Values", color="#FFFFFF")
    plt.ylabel("Density", color="#FFFFFF")
    plt.tick_params(colors="#FFFFFF")
    plt.legend(title='Feature', labels=feature_template_base.columns.tolist())
    ax.set_facecolor('#0B1D51')
    fig.patch.set_facecolor('#0B1D51')
    st.pyplot(fig)


# ==========================
# FOOTER
# ==========================
st.markdown("""
---
<div style="text-align:center; color:#00E5FF;">
Made with using **Faincre** & Streamlit | Futuristic Dashboard V2.0
</div>
""", unsafe_allow_html=True)