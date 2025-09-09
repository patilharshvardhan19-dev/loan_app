# ~/loan_app/step_5_1.py  (fixed: clean infinities / clip outliers before training)
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
import re

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

print("Bucket counts:\n", df["risk_bucket"].value_counts(), "\n")

# === Derived features (ensure they exist) ===
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

# For baseline model → make binary target: 1 = Will Not Repay, 0 = Will Repay (ignore 'Maybe')
df_bin = df[df["risk_bucket"].isin(["Will Repay","Will Not Repay"])].copy()
df_bin["target"] = (df_bin["risk_bucket"] == "Will Not Repay").astype(int)

print("Training set size after dropping 'Maybe':", df_bin.shape)

# Select features
features = [c for c in ["fico_score","dti_computed","loan_to_income","annual_inc","emp_length_years","has_delinquency"] if c in df_bin.columns]
print("Using features:", features)

X = df_bin[features].copy()
y = df_bin["target"]

# ------------------- CLEAN & CLIP -------------------
# Replace infinite values with NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Clip realistic ranges:
# - fico_score: [300, 850]
if "fico_score" in X.columns:
    X["fico_score"] = pd.to_numeric(X["fico_score"], errors="coerce").clip(lower=300, upper=850)

# - dti_computed: clip to [0, 200] (percentage or ratio depending on dataset). Values >200 are extremely unlikely.
if "dti_computed" in X.columns:
    X["dti_computed"] = pd.to_numeric(X["dti_computed"], errors="coerce")
    X.loc[X["dti_computed"] < 0, "dti_computed"] = np.nan
    X["dti_computed"] = X["dti_computed"].clip(upper=200)

# - loan_to_income: clip to [0, 10]
if "loan_to_income" in X.columns:
    X["loan_to_income"] = pd.to_numeric(X["loan_to_income"], errors="coerce")
    X.loc[X["loan_to_income"] < 0, "loan_to_income"] = np.nan
    X["loan_to_income"] = X["loan_to_income"].clip(upper=10)

# - annual_inc: negative -> NaN; cap at a high value to avoid huge influence (e.g., 10M)
if "annual_inc" in X.columns:
    X["annual_inc"] = pd.to_numeric(X["annual_inc"], errors="coerce")
    X.loc[X["annual_inc"] <= 0, "annual_inc"] = np.nan
    X["annual_inc"] = X["annual_inc"].clip(upper=10_000_000)

# - emp_length_years: clip to [0, 50]
if "emp_length_years" in X.columns:
    X["emp_length_years"] = pd.to_numeric(X["emp_length_years"], errors="coerce")
    X.loc[X["emp_length_years"] < 0, "emp_length_years"] = np.nan
    X["emp_length_years"] = X["emp_length_years"].clip(lower=0, upper=50)

# Fill remaining NaNs with median (per column)
X = X.fillna(X.median())

# Final check for infinities / huge values
if not np.isfinite(X.to_numpy()).all():
    raise ValueError("Data still contains non-finite values after cleaning.")

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train baseline Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("\nClassification report (0=Will Repay, 1=Will Not Repay):\n")
print(classification_report(y_test, y_pred, zero_division=0))

print("ROC AUC:", roc_auc_score(y_test, y_prob))















