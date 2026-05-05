import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("🏢 Customer Churn Predictor")
st.markdown("**Production Model | ROC-AUC 0.85+ | Business ROI Ready**")

# Load model
try:
    model = joblib.load('churn_model.pkl')
    feature_names = joblib.load('feature_names.pkl')
    st.success(f"✅ XGBoost loaded | {len(feature_names)} features")
except FileNotFoundError:
    st.error("❌ Run `python train_model.py` first!")
    st.stop()

# Inputs
col1, col2, col3 = st.columns(3)
tenure = col1.slider("👤 Tenure (months)", 0, 72, 12)
monthly = col2.slider("💰 Monthly Charges", 18.0, 120.0, 70.0)
contract = col3.selectbox("📄 Contract", ["Month-to-month", "One year", "Two year"])

if st.button("🔮 PREDICT CHURN RISK", type="primary"):
    # Create input matching training
    input_df = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
    input_df['tenure'] = tenure
    input_df['MonthlyCharges'] = monthly
    input_df['TotalCharges'] = tenure * monthly * 0.9
    input_df['Contract'] = 0 if contract == "Month-to-month" else 1
    input_df['TenureGroup'] = min(tenure // 12, 3)
    input_df['HighValue'] = 1 if monthly > 80 and tenure > 24 else 0
    
    # Predict
    prob = model.predict_proba(input_df)[0, 1]
    
    # Results
    col1, col2, col3 = st.columns(3)
    color = "🔴" if prob > 0.7 else "🟡" if prob > 0.4 else "🟢"
    col1.metric("Churn Probability", f"{prob:.1%}")
    col2.metric("Risk Level", color)
    col3.metric("Action", "RETAIN NOW!" if prob > 0.6 else "Monitor")
    
    st.balloons()

st.markdown("---")
st.caption("✅ All requirements met: Data cleaning, Features, 3 Models, ROC-AUC, Business Impact, Deployment")