# ~/loan_app/step_5_2b.py
import pandas as pd
import os
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

fname = os.path.expanduser("~/loan_app/data/MOTHERFILE.xlsx")
df = pd.read_excel(fname, sheet_name=0, engine="openpyxl")

# --- Map loan_status into 3 buckets ---
def map_status(x):
    if pd.isna(x): return "Maybe"
    s = str(x).lower()
    if "fully paid" in s: return "Will Repay"
    if "charged off" in s or "default" in s: return "Will Not Repay"
    if "current" in s or "late" in s or "grace" in s: return "Maybe"
    return "Maybe"

df["risk_bucket"] = df["loan_status"].apply(map_status)

# Derived features
if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
    df["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
if "dti" in df.columns:
    df["dti_computed"] = pd.to_numeric(df["dti"], errors="coerce")
if "loan_amnt" in df.columns and "annual_inc" in df.columns:
    df["loan_to_income"] = pd.to_numeric(df["loan_amnt"], errors="coerce") / pd.to_numeric(df["annual_inc"], errors="coerce")
if "emp_length" in df.columns:
    def parse_emp_length(x):
        if pd.isna(x): return None
        s = str(x).lower()
        if "10" in s: return 10
        if "<" in s: return 0
        digits = re.findall(r"\d+", s)
        return int(digits[0]) if digits else None
    df["emp_length_years"] = df["emp_length"].apply(parse_emp_length)
if "delinq_2yrs" in df.columns:
    df["has_delinquency"] = (df["delinq_2yrs"] > 0).astype(int)

# Keep only repay vs not repay
df_bin = df[df["risk_bucket"].isin(["Will Repay","Will Not Repay"])].copy()
df_bin["target"] = (df_bin["risk_bucket"] == "Will Not Repay").astype(int)

# Select features (expanded)
features = [c for c in [
    "fico_score","dti_computed","loan_to_income","annual_inc","emp_length_years",
    "has_delinquency","int_rate","installment","grade","purpose"
] if c in df_bin.columns]

X = df_bin[features].copy()
y = df_bin["target"]

# Encode categorical features
for col in ["grade","purpose"]:
    if col in X.columns:
        X[col] = X[col].astype("category").cat.codes

# Clean data
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = X.fillna(X.median())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Class imbalance handling
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos
print(f"Class balance: {neg} Will Repay, {pos} Will Not Repay → scale_pos_weight={scale_pos_weight:.2f}")

# Train XGBoost with imbalance handling
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("\nUsing features:", features)
print("\nClassification report (0=Will Repay, 1=Will Not Repay):\n")
print(classification_report(y_test, y_pred, zero_division=0))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

