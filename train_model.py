import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("🚀 BULLETPROOF TRAINING (100% Windows Safe)...")

# 1. LOAD DATA
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"📊 Loaded: {df.shape}")

# 2. CLEAN
df = df.drop('customerID', axis=1)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 3. ENCODE ALL OBJECT COLUMNS (Prevents XGBoost error)
object_cols = df.select_dtypes(include=['object']).columns
for col in object_cols:
    df[col] = df[col].astype(str).astype('category').cat.codes

# 4. SAFE FEATURE ENGINEERING (No NaN issues)
df['TenureGroup'] = np.minimum(df['tenure'] // 12, 3).astype(int)  # Safe integer
df['HighValue'] = ((df['MonthlyCharges'] > 80) & (df['tenure'] > 24)).astype(int)

print("✅ Features ready:", df.shape)

# 5. TRAIN/TEST SPLIT
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")

# 6. TRAIN 3 MODELS
print("🤖 Training models...")
models = {
    'XGBoost': xgb.XGBClassifier(n_estimators=50, random_state=42, n_jobs=1),
    'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
}

results = {}
for name, model in models.items():
    print(f"  Training {name}...")
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_proba)
    results[name] = score
    print(f"    {name}: ROC-AUC = {score:.4f}")

# 7. SELECT & SAVE BEST
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
joblib.dump(best_model, 'churn_model.pkl')
joblib.dump(X.columns.tolist(), 'feature_names.pkl')

print(f"\n🎉 BEST MODEL: {best_model_name}")
print(f"   ROC-AUC: {results[best_model_name]:.4f}")
print(f"   Features: {len(X.columns)}")

# 8. BUSINESS IMPACT
y_pred = best_model.predict(X_test)
tp = sum((y_pred == 1) & (y_test == 1))
fp = sum((y_pred == 1) & (y_test == 0))
profit = tp * 1000 * 0.3 - fp * 20  # Retention success 30%

print(f"\n💰 BUSINESS IMPACT (per {len(y_test)} customers):")
print(f"   True Positives (saved): {tp}")
print(f"   False Positives (cost): {fp}")
print(f"   Revenue Saved: ${tp * 1000 * 0.3:,.0f}")
print(f"   Retention Cost: ${fp * 20:,.0f}")
print(f"   NET PROFIT: ${profit:,.0f}")
print(f"\n✅ MODEL SAVED! Run: streamlit run app.py")