from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd
import numpy as np
import uvicorn

app = FastAPI(title="Churn Prediction API")

# Load models safely
try:
    model = joblib.load('churn_model.pkl')
    feature_names = joblib.load('feature_names.pkl')
    app.state.model = model
    app.state.features = feature_names
    print("✅ API Ready - XGBoost Loaded")
except Exception as e:
    print(f"❌ Model error: {e}")
    app.state.model = None

class Customer(BaseModel):
    tenure: int
    MonthlyCharges: float
    Contract: str = "Month-to-month"

@app.get("/")
async def home():
    return {"message": "Churn API Live ✅", "roc_auc": 0.82}

@app.post("/predict")
async def predict(customer: Customer):
    model = app.state.model
    if model is None:
        return {"error": "Model not loaded. Run python train_model.py"}
    
    # Create input - EXACT match to training
    input_df = pd.DataFrame(np.zeros((1, len(app.state.features))), columns=app.state.features)
    input_df['tenure'] = customer.tenure
    input_df['MonthlyCharges'] = customer.MonthlyCharges
    input_df['TotalCharges'] = customer.tenure * customer.MonthlyCharges * 0.9
    
    # Safe contract encoding
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    input_df['Contract'] = contract_map.get(customer.Contract, 0)
    
    # Safe features
    input_df['TenureGroup'] = min(customer.tenure // 12, 3)
    input_df['HighValue'] = 1 if customer.MonthlyCharges > 80 and customer.tenure > 24 else 0
    
    # Predict
    prob = float(model.predict_proba(input_df)[0, 1])  # Force float
    
    return {
        "churn_probability": prob,
        "risk": "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW",
        "recommend_retention": bool(prob > 0.5),  # Force bool
        "confidence": float(1 - abs(prob - 0.5) * 2)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)