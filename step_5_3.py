# ~/loan_app/step_5_3.py
import pandas as pd, numpy as np, re, os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import shap
import matplotlib.pyplot as plt

# load data (same derived features used previously)
fname = os.path.expanduser("~/loan_app/data/MOTHERFILE.xlsx")
df = pd.read_excel(fname, sheet_name=0, engine="openpyxl")

def map_status(x):
    if pd.isna(x): return "Maybe"
    s = str(x).lower()
    if "fully paid" in s: return "Will Repay"
    if "charged off" in s or "default" in s: return "Will Not Repay"
    if "current" in s or "late" in s or "grace" in s: return "Maybe"
    return "Maybe"

df["risk_bucket"] = df["loan_status"].apply(map_status)
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

# Binary target
df_bin = df[df["risk_bucket"].isin(["Will Repay","Will Not Repay"])].copy()
df_bin["target"] = (df_bin["risk_bucket"] == "Will Not Repay").astype(int)

# Features (same as last run)
features = [c for c in [
    "fico_score","dti_computed","loan_to_income","annual_inc","emp_length_years",
    "has_delinquency","int_rate","installment","grade","purpose"
] if c in df_bin.columns]

X = df_bin[features].copy()
y = df_bin["target"]

# Encode categories
for col in ["grade","purpose"]:
    if col in X.columns:
        X[col] = X[col].astype("category").cat.codes

# Clean
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = X.fillna(X.median())

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# compute scale_pos_weight
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

# Train same XGBoost (retrain)
model = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
    scale_pos_weight=scale_pos_weight, use_label_encoder=False
)
model.fit(X_train, y_train)

# Print feature importance (gain)
fi = model.get_booster().get_score(importance_type="gain")
# sort and display top 10 features
fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
print("Top feature importances (gain):")
for k, v in fi_sorted[:10]:
    print(f"{k:20s} -> {v:.4f}")

# SHAP summary plot (global)
os.makedirs(os.path.expanduser("~/loan_app/outputs"), exist_ok=True)
explainer = shap.TreeExplainer(model)
# Use a sample for speed
X_sample = X_test.sample(min(2000, len(X_test)), random_state=42)
shap_values = explainer.shap_values(X_sample)
plt.figure()
shap.summary_plot(shap_values, X_sample, show=False, plot_type="bar")
plt.title("SHAP feature importance (bar)")
plt.savefig(os.path.expanduser("~/loan_app/outputs/shap_summary_bar.png"), bbox_inches="tight")
plt.close()

# Also save the full dot summary (may take longer)
plt.figure()
shap.summary_plot(shap_values, X_sample, show=False)
plt.title("SHAP summary (dot)")
plt.savefig(os.path.expanduser("~/loan_app/outputs/shap_summary_dot.png"), bbox_inches="tight")
plt.close()

# Per-applicant top-3 contributing features on test set (save csv)
X_test_idx = X_test.reset_index(drop=True)
sv = explainer.shap_values(X_test_idx)
top_reasons = []
for i in range(len(X_test_idx)):
    arr = sv[i]
    idxs = np.argsort(-np.abs(arr))[:3]
    feats = [(X_test_idx.columns[j], float(arr[j])) for j in idxs]
    top_reasons.append([i] + [f"{f}:{round(val,4)}" for f, val in feats])

out_df = pd.DataFrame(top_reasons, columns=["test_index","reason1","reason2","reason3"])
out_df["pred_proba"] = model.predict_proba(X_test_idx)[:,1]
out_df["true"] = y_test.reset_index(drop=True)
out_df.to_csv(os.path.expanduser("~/loan_app/outputs/test_top_reasons.csv"), index=False)

print("\nSaved SHAP plots to ~/loan_app/outputs/ (shap_summary_bar.png, shap_summary_dot.png)")
print("Saved per-applicant top-3 reasons to ~/loan_app/outputs/test_top_reasons.csv")
print("ROC AUC on test set:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

