# ~/loan_app/app.py
"""
Robust Streamlit app for home-loan credit risk assessment with:
- flexible column detection
- derived features
- training/loading/saving XGBoost model; force retrain and manual save (password-protected)
- alignment to model features to avoid mismatch
- rule-based fallback scoring
- SHAP explanations (optional)
- Repayment Score (0-10)
- Dashboard showing parameter weightage and decision process steps
"""
import streamlit as st
import pandas as pd
import numpy as np
import os, re, zipfile, time, json
import joblib
from xgboost import XGBClassifier

# optional shap
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# Paths & dirs
BASE = os.path.expanduser("~/loan_app")
MODELS_DIR = os.path.join(BASE, "models")
OUTPUTS_DIR = os.path.join(BASE, "outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.joblib")
EXPORT_ZIP = os.path.join(BASE, "final_submission.zip")

st.set_page_config(layout="wide")
st.title("🏦 Home Loan Credit Risk Assessment — Full (Dashboard + Passworded Actions)")

# ---------------- Sidebar controls & admin password ----------------
st.sidebar.header("Controls")
enable_pruning = st.sidebar.checkbox("Aggressive column pruning (drop unused columns)", value=False)

# Admin password box (protects retrain/save)
ADMIN_PW = "UBS@AI"
admin_password = st.sidebar.text_input("Admin password (required for retrain/save)", type="password", placeholder="Enter admin password")

# allow model fallback as before
use_rule_fallback_default = st.sidebar.checkbox("Enable rule-based fallback (if no model)", value=True)

# action buttons (visible to all, but action requires correct password)
force_retrain_btn = st.sidebar.button("Force retrain model on uploaded labels and save")
manual_save_btn = st.sidebar.button("Manually save last trained model to disk")
export_package = st.sidebar.button("Export deliverable ZIP (app, models, outputs, report)")

# ---------------- helper functions ----------------
def normalize_colname(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def find_column(cols, patterns):
    cols_norm = {c: normalize_colname(c) for c in cols}
    for pat in patterns:
        p = normalize_colname(pat)
        for orig, norm in cols_norm.items():
            if p in norm:
                return orig
    return None

def standardize_columns(df):
    cols = list(df.columns)
    mapping = {}
    candidates = {
        "loan_amnt": ["loan_amnt","loanamount","loan_amount","loanamt","amount","principal"],
        "annual_inc": ["annual_inc","annualincome","annual_income","income","gross_income","salary"],
        "installment": ["installment","monthlypayment","monthly_payment","monthlypymt","monthlypmt","monthly_installment","payment","payment_amt","paymentamount","monthly_pymt","total_pymnt"],
        "int_rate": ["int_rate","interestrate","interest_rate","interest%","rate","interest_perc","interest_pct"],
        "fico_range_low": ["fico_range_low","fico_low"],
        "fico_range_high": ["fico_range_high","fico_high"],
        "fico_score": ["fico_score","fico","credit_score","creditscore","score","cibil","cibil_score","cibilscore"],
        "dti": ["dti","debttoincome","debt_to_income","debtratio","debt_ratio","debt_percent"],
        "emp_length": ["emp_length","employment_length","employ_length","emp_len","work_experience","yrs_employed","employmentyears","yrsemployed","years_employed"],
        "delinq_2yrs": ["delinq_2yrs","delinquencies","delinq","num_delinq","numdelinq"],
        "grade": ["grade","loan_grade","risk_grade","grade_code"],
        "purpose": ["purpose","loan_purpose","purposedesc","loan_purpose_desc","reason","purpose_code","loanuse"],
        "home_ownership": ["home_ownership","homeownership","home_owner"]
    }
    for std, pats in candidates.items():
        found = find_column(cols, pats)
        if found:
            mapping[std] = found
    rename_map = {}
    for std, src in mapping.items():
        if src and src != std and std not in df.columns:
            rename_map[src] = std
    if rename_map:
        df = df.rename(columns=rename_map)
    if "fico_score" not in df.columns:
        for c in df.columns:
            n = normalize_colname(c)
            if "cibil" in n or ("credit" in n and "score" in n):
                df["fico_score"] = pd.to_numeric(df[c], errors="coerce")
                break
    if "fico_score" not in df.columns and "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["fico_score"] = (pd.to_numeric(df["fico_range_low"], errors="coerce") + pd.to_numeric(df["fico_range_high"], errors="coerce"))/2
    if "int_rate" in df.columns:
        try:
            df["int_rate"] = df["int_rate"].astype(str).str.replace("%","").str.replace(",","").astype(float)
        except Exception:
            df["int_rate"] = pd.to_numeric(df["int_rate"], errors="coerce")
    if "emp_length" in df.columns and "emp_length_years" not in df.columns:
        def parse_emp(x):
            if pd.isna(x): return np.nan
            s = str(x).lower()
            if "10" in s and "+" in s: return 10
            if "<" in s: return 0
            m = re.findall(r"\d+", s)
            return int(m[0]) if m else np.nan
        df["emp_length_years"] = df["emp_length"].apply(parse_emp)
    if "loan_to_income" not in df.columns:
        if "loan_amnt" in df.columns and "annual_inc" in df.columns:
            df["loan_to_income"] = pd.to_numeric(df["loan_amnt"], errors="coerce") / pd.to_numeric(df["annual_inc"], errors="coerce")
    if "dti" in df.columns and "dti_computed" not in df.columns:
        df["dti_computed"] = pd.to_numeric(df["dti"], errors="coerce")
    if "dti_computed" not in df.columns:
        if "installment" in df.columns and "annual_inc" in df.columns:
            inst = pd.to_numeric(df["installment"], errors="coerce")
            ann = pd.to_numeric(df["annual_inc"], errors="coerce")
            df["dti_computed"] = inst / (ann/12.0)
    if "delinq_2yrs" in df.columns and "has_delinquency" not in df.columns:
        df["has_delinquency"] = (pd.to_numeric(df["delinq_2yrs"], errors="coerce") > 0).astype(int)
    for k in ["loan_amnt","annual_inc","installment","int_rate","fico_score","loan_to_income","dti_computed","emp_length_years"]:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k].astype(str).str.replace("%","").str.replace(",",""), errors="coerce")
    detected = {
        "loan_amnt": ("loan_amnt" if "loan_amnt" in df.columns else None),
        "annual_inc": ("annual_inc" if "annual_inc" in df.columns else None),
        "installment": ("installment" if "installment" in df.columns else None),
        "int_rate": ("int_rate" if "int_rate" in df.columns else None),
        "fico_score": ("fico_score" if "fico_score" in df.columns else None),
        "dti_computed": ("dti_computed" if "dti_computed" in df.columns else None),
        "loan_to_income": ("loan_to_income" if "loan_to_income" in df.columns else None),
        "emp_length_years": ("emp_length_years" if "emp_length_years" in df.columns else None),
        "has_delinquency": ("has_delinquency" if "has_delinquency" in df.columns else None),
        "grade": ("grade" if "grade" in df.columns else None),
        "purpose": ("purpose" if "purpose" in df.columns else None),
    }
    return df, detected

def clean_feature_matrix(X):
    X = X.copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    if "fico_score" in X.columns:
        X["fico_score"] = pd.to_numeric(X["fico_score"], errors="coerce").clip(300,850)
    if "dti_computed" in X.columns:
        X["dti_computed"] = pd.to_numeric(X["dti_computed"], errors="coerce")
        X.loc[X["dti_computed"] < 0, "dti_computed"] = np.nan
        X["dti_computed"] = X["dti_computed"].clip(upper=200)
    if "loan_to_income" in X.columns:
        X["loan_to_income"] = pd.to_numeric(X["loan_to_income"], errors="coerce")
        X.loc[X["loan_to_income"] < 0, "loan_to_income"] = np.nan
        X["loan_to_income"] = X["loan_to_income"].clip(upper=10)
    if "annual_inc" in X.columns:
        X["annual_inc"] = pd.to_numeric(X["annual_inc"], errors="coerce")
        X.loc[X["annual_inc"] <= 0, "annual_inc"] = np.nan
        X["annual_inc"] = X["annual_inc"].clip(upper=10_000_000)
    if "emp_length_years" in X.columns:
        X["emp_length_years"] = pd.to_numeric(X["emp_length_years"], errors="coerce")
        X.loc[X["emp_length_years"] < 0, "emp_length_years"] = np.nan
        X["emp_length_years"] = X["emp_length_years"].clip(lower=0, upper=50)
    X = X.fillna(X.median())
    return X

def rule_based_score(row):
    w = {"fico_score": -0.35, "dti_computed": 0.30, "loan_to_income": 0.20, "annual_inc": -0.10, "emp_length_years": 0.05, "has_delinquency": 0.40, "int_rate": 0.25, "installment": 0.02}
    s = 0.0; denom = 0.0
    if "fico_score" in row and not pd.isna(row["fico_score"]):
        f = (row["fico_score"] - 300) / (850 - 300)
        s += w["fico_score"] * f; denom += abs(w["fico_score"])
    if "dti_computed" in row and not pd.isna(row["dti_computed"]):
        d = min(row["dti_computed"]/100.0, 1.0); s += w["dti_computed"] * d; denom += abs(w["dti_computed"])
    if "loan_to_income" in row and not pd.isna(row["loan_to_income"]):
        l = min(row["loan_to_income"]/1.0, 1.0); s += w["loan_to_income"] * l; denom += abs(w["loan_to_income"])
    if "annual_inc" in row and not pd.isna(row["annual_inc"]):
        a = min(row["annual_inc"]/100000.0, 1.0); s += w["annual_inc"] * a; denom += abs(w["annual_inc"])
    if "emp_length_years" in row and not pd.isna(row["emp_length_years"]):
        e = min(row["emp_length_years"]/40.0, 1.0); s += w["emp_length_years"] * e; denom += abs(w["emp_length_years"])
    if "has_delinquency" in row and not pd.isna(row["has_delinquency"]):
        hd = 1.0 if int(row["has_delinquency"]) else 0.0; s += w["has_delinquency"] * hd; denom += abs(w["has_delinquency"])
    if "int_rate" in row and not pd.isna(row["int_rate"]):
        try: ir = float(str(row["int_rate"]).strip().strip("%"))
        except: ir = 0.0
        irn = min(ir/50.0,1.0); s += w["int_rate"] * irn; denom += abs(w["int_rate"])
    if "installment" in row and not pd.isna(row["installment"]):
        inst = min(row["installment"]/5000.0, 1.0); s += w["installment"] * inst; denom += abs(w["installment"])
    if denom == 0: return 0.5
    raw = s/denom
    prob = 1.0/(1.0+np.exp(-3.0*raw))
    return float(np.clip(prob, 0.0, 1.0))

def humanize_reason(feat, val):
    name_map = {"fico_score":"FICO score","dti_computed":"DTI (debt-to-income)","loan_to_income":"Loan-to-Income","annual_inc":"Annual income","emp_length_years":"Employment length (years)","has_delinquency":"Past delinquencies","int_rate":"Interest rate","installment":"Monthly installment","grade":"Loan grade","purpose":"Loan purpose"}
    pretty = name_map.get(feat, feat.replace("_"," ").title())
    direction = "increased" if val>0 else "reduced"
    mag = abs(val)
    mag_s = f"{mag:.4f}" if abs(mag) < 1 else f"{mag:.2f}"
    return f"{pretty} ({mag_s}) {direction} default risk."

# ---------------- Upload ----------------
uploaded_file = st.file_uploader("Upload dataset (CSV / XLS / XLSX) — training file or applicants file", type=["csv","xls","xlsx"])
if uploaded_file is None:
    st.info("Upload your dataset (e.g., MotherFile.xlsx or teacher applicants file) to proceed.")
    st.stop()

try:
    if str(uploaded_file.name).lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, sheet_name=0, engine="openpyxl")
except Exception as e:
    st.error(f"Failed to read uploaded file: {e}")
    st.stop()

st.write("Preview (first 5 rows):")
st.dataframe(df.head())

# ---------------- Optional pruning ----------------
if enable_pruning:
    KEEP_FEATURES = ["fico_score","dti_computed","loan_to_income","annual_inc","emp_length_years","has_delinquency","int_rate","installment","grade","purpose"]
    KEEP_EXTRA = []
    id_patterns = ["id","member","app","loan","cust","account"]
    id_cols = [c for c in df.columns if any(p in normalize_colname(c) for p in id_patterns)]
    label_col = ["loan_status"] if "loan_status" in df.columns else []
    keep_cols = []
    for k in KEEP_FEATURES:
        if k in df.columns:
            keep_cols.append(k)
    for x in KEEP_EXTRA:
        if x in df.columns:
            keep_cols.append(x)
    for c in id_cols:
        if c in df.columns:
            keep_cols.append(c)
    keep_cols += label_col
    if len(keep_cols) == 0:
        st.warning("Pruning would remove all columns. Keeping first 20 columns instead.")
        keep_cols = list(df.columns[:20])
    st.caption(f"Pruning columns: keeping {len(keep_cols)} columns.")
    df = df[keep_cols].copy()

# ---------------- Standardize columns ----------------
df, detected = standardize_columns(df)
st.write("Detected / created canonical columns (None = not found):")
st.write(json.dumps(detected, indent=2))

# ---------------- Features list ----------------
features = [c for c in ["fico_score","dti_computed","loan_to_income","annual_inc","emp_length_years","has_delinquency","int_rate","installment","grade","purpose"] if c in df.columns]
st.write("Modeling features available:", features)

# ---------------- Prepare labels if present ----------------
y = None; df_bin = None
if "loan_status" in df.columns:
    def _map_status(x):
        if pd.isna(x): return "Maybe"
        s = str(x).lower()
        if "fully paid" in s: return "Will Repay"
        if "charged off" in s or "default" in s: return "Will Not Repay"
        if "current" in s or "late" in s or "grace" in s: return "Maybe"
        return "Maybe"
    df["risk_bucket"] = df["loan_status"].apply(_map_status)
    df_bin = df[df["risk_bucket"].isin(["Will Repay","Will Not Repay"])].copy()
    if len(df_bin)>0:
        df_bin["target"] = (df_bin["risk_bucket"]=="Will Not Repay").astype(int)
        y = df_bin["target"]

# ---------------- Load saved model ----------------
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        st.sidebar.success("Loaded saved model from disk.")
    except Exception as e:
        st.sidebar.warning(f"Saved model exists but failed to load: {e}")

# ---------------- Force retrain (password-protected) ----------------
if force_retrain_btn:
    if admin_password != ADMIN_PW:
        st.sidebar.error("Incorrect admin password — force retrain aborted.")
    else:
        if y is not None and len(y) >= 50:
            st.sidebar.info("Force retrain: training on uploaded labeled data...")
            X_bin = df_bin[features].copy()
            for col in ["grade","purpose"]:
                if col in X_bin.columns:
                    X_bin[col] = X_bin[col].astype("category").cat.codes
            X_bin = clean_feature_matrix(X_bin)
            neg_pos = np.bincount(df_bin["target"])
            if len(neg_pos) == 1:
                neg = int(neg_pos[0]); pos = 0
            else:
                neg, pos = int(neg_pos[0]), int(neg_pos[1])
            scale_pos_weight = neg/pos if pos>0 else 1.0
            try:
                model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", scale_pos_weight=scale_pos_weight)
                model.fit(X_bin, df_bin["target"])
                joblib.dump(model, MODEL_PATH)
                st.sidebar.success("Force retrain complete and model saved.")
            except Exception as e:
                st.sidebar.error(f"Force retrain failed: {e}")
        else:
            st.sidebar.warning("Not enough labeled rows to retrain (need >=50).")

# ---------------- Manual save (password-protected) ----------------
if manual_save_btn:
    if admin_password != ADMIN_PW:
        st.sidebar.error("Incorrect admin password — save aborted.")
    else:
        if model is not None:
            try:
                joblib.dump(model, MODEL_PATH)
                st.sidebar.success("Model saved to models/xgb_model.joblib")
            except Exception as e:
                st.sidebar.error(f"Save failed: {e}")
        else:
            st.sidebar.warning("No model in memory to save. Train first or upload labeled file.")

# ---------------- Build X ----------------
if len(features) == 0:
    st.error("No modeling features detected in uploaded file — add fields like annual_inc, loan_amnt, fico_range_low/high, dti, installment, etc.")
    st.stop()

X = df[features].copy()
for col in ["grade","purpose"]:
    if col in X.columns:
        X[col] = X[col].astype("category").cat.codes
X = clean_feature_matrix(X)

# ---------------- Repayment Score logic (same as before) ----------------
feature_directions = {"fico_score": True, "annual_inc": True, "emp_length_years": True, "dti_computed": False, "loan_to_income": False, "int_rate": False, "installment": False, "has_delinquency": False}
score_features = [f for f in ["fico_score","dti_computed","loan_to_income","annual_inc","emp_length_years","has_delinquency","int_rate","installment"] if f in X.columns]

weights = {}
if model is not None:
    try:
        fi = getattr(model, "feature_importances_", None)
        names = list(getattr(model, "feature_names_in_", []))
        if fi is not None and len(fi)>0 and len(names)>0:
            fi_map = {n: float(v) for n,v in zip(names, fi)}
            total = sum(abs(fi_map.get(f, 0.0)) for f in score_features)
            if total > 0:
                for f in score_features:
                    weights[f] = abs(fi_map.get(f, 0.0)) / total
    except Exception:
        weights = {}
if not weights:
    rule_w = {"fico_score": 0.25, "dti_computed": 0.20, "loan_to_income": 0.15, "annual_inc": 0.15, "emp_length_years": 0.05, "has_delinquency": 0.15, "int_rate": 0.03, "installment": 0.02}
    total = sum(rule_w.get(f,0) for f in score_features)
    if total == 0:
        for f in score_features:
            weights[f] = 1.0/len(score_features)
    else:
        for f in score_features:
            weights[f] = rule_w.get(f,0)/total

st.subheader("Key parameters & weightage used for scoring")
w_table = pd.DataFrame([{"feature": f, "weight_pct": round(weights.get(f,0)*100,2)} for f in score_features]).sort_values("weight_pct", ascending=False)
st.table(w_table)

norm_info = {}
for f in score_features:
    if df_bin is not None and f in df_bin.columns and df_bin[f].dropna().size > 10:
        col_vals = pd.to_numeric(df_bin[f], errors="coerce").dropna()
    else:
        col_vals = pd.to_numeric(df[f], errors="coerce").dropna() if f in df.columns else pd.Series(dtype=float)
    if col_vals.size == 0:
        if f == "fico_score":
            vmin, vmax = 300.0, 850.0
        elif f in ["dti_computed","loan_to_income"]:
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.nanpercentile(col_vals, 1)) if col_vals.size>1 else float(col_vals.min())
        vmax = float(np.nanpercentile(col_vals, 99)) if col_vals.size>1 else float(col_vals.max())
        if vmin == vmax:
            vmin -= 1.0; vmax += 1.0
    norm_info[f] = (vmin, vmax)

def norm_value(f, v):
    if pd.isna(v):
        return 0.5
    vmin, vmax = norm_info.get(f, (0.0,1.0))
    try:
        val = float(v)
    except:
        return 0.5
    if vmax == vmin:
        scaled = 0.5
    else:
        scaled = (val - vmin) / (vmax - vmin)
        scaled = max(0.0, min(1.0, scaled))
    if not feature_directions.get(f, True):
        scaled = 1.0 - scaled
    return scaled

scores = []
per_row_contribs = []
for _, row in X.iterrows():
    s = 0.0
    contribs = {}
    for f in score_features:
        raw = row.get(f, np.nan)
        nv = norm_value(f, raw)
        w = weights.get(f, 0.0)
        contribs[f] = (nv, w, nv*w)
        s += nv * w
    repay_score = float(round(s * 10.0, 3))
    scores.append(repay_score)
    per_row_contribs.append(contribs)

st.subheader("Repayment score distribution & historical mapping")
st.write("Repayment score is 0..10 (higher = more likely to repay on time).")
try:
    st.bar_chart(pd.Series(scores).value_counts(bins=10).sort_index())
except Exception:
    pass

if df_bin is not None and "target" in df_bin.columns and len(df_bin) > 50:
    hist_scores = []
    for _, row in df_bin.iterrows():
        s = 0.0
        for f in score_features:
            raw = row.get(f, np.nan)
            nv = norm_value(f, raw)
            w = weights.get(f,0.0)
            s += nv*w
        hist_scores.append(s*10.0)
    df_bin = df_bin.copy()
    df_bin["repay_score"] = hist_scores
    if "risk_bucket" not in df_bin.columns and "loan_status" in df_bin.columns:
        def status_map(s):
            s = str(s).lower()
            if "fully paid" in s: return "Will Repay"
            if "charged off" in s or "default" in s: return "Will Not Repay"
            return "Maybe"
        df_bin["risk_bucket"] = df_bin["loan_status"].apply(status_map)
    grp = df_bin.groupby("risk_bucket")["repay_score"].agg(["count","mean","std","min","max"]).reset_index()
    st.write("Historical average repay_score by bucket (from labeled data):")
    st.dataframe(grp)
    if set(["Will Repay","Will Not Repay"]).issubset(set(grp["risk_bucket"])):
        mean_repay = float(grp.loc[grp["risk_bucket"]=="Will Repay","mean"].values[0])
        mean_non = float(grp.loc[grp["risk_bucket"]=="Will Not Repay","mean"].values[0])
        midpoint = (mean_repay + mean_non)/2.0
        st.write(f"Suggested decision midpoint between classes: **{midpoint:.2f}** on 0-10 scale.")
        st.info("Rule: score >= midpoint => likely to repay; score < midpoint => likely not to repay.")
else:
    st.info("No labeled historical data available in this upload to compute mapping. Upload MotherFile (labeled) to compute mapping.")

# ---------------- Prediction: attempt model, otherwise rule fallback ----------------
use_rule_fallback = use_rule_fallback_default
probs = None
if (model is not None):
    try:
        expected = None
        try:
            expected = list(model.feature_names_in_)
        except Exception:
            expected = None
        if expected:
            for col in expected:
                if col not in X.columns:
                    X[col] = X.median().get(col, 0)
                    X[col] = X[col].fillna(0)
            extra = [c for c in X.columns if c not in expected]
            if extra:
                X = X.drop(columns=extra)
            X = X[expected]
        probs = model.predict_proba(X)[:,1]
    except Exception as e:
        st.warning(f"Model predict failed: {e}. Falling back to rule-based scoring.")
        probs = None

if probs is None:
    st.info("Using rule-based scoring (fallback).")
    probs = X.apply(rule_based_score, axis=1).values

low_thr = st.sidebar.slider("Low threshold (Will Repay if p < low)", 0.0, 0.5, 0.35, 0.01)
high_thr = st.sidebar.slider("High threshold (Will Not Repay if p > high)", 0.5, 1.0, 0.65, 0.01)
def assign_bucket(p):
    if p < low_thr: return "Will Repay"
    if p > high_thr: return "Will Not Repay"
    return "Maybe"
pred_bucket = [assign_bucket(p) for p in probs]

# ---------------- SHAP explanations ----------------
REASON_CAP = 2000
reason1 = [""] * len(X); reason2 = [""] * len(X); reason3 = [""] * len(X)
if HAS_SHAP and model is not None:
    try:
        n_explain = min(REASON_CAP, len(X))
        X_explain = X.iloc[:n_explain]
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_explain)
        for i in range(n_explain):
            arr = shap_vals[i]
            idxs = np.argsort(-np.abs(arr))[:3]
            for j, idx in enumerate(idxs):
                feat = X_explain.columns[idx]; val = float(arr[idx])
                if j==0: reason1[i] = humanize_reason(feat, val)
                if j==1: reason2[i] = humanize_reason(feat, val)
                if j==2: reason3[i] = humanize_reason(feat, val)
    except Exception as e:
        st.warning(f"SHAP explanation failed or too slow: {e}")

if probs is not None and model is None:
    for i, row in X.iterrows():
        contribs = []
        if "fico_score" in row.index and not pd.isna(row["fico_score"]):
            f = (row["fico_score"] - 300) / (850 - 300); contribs.append(("fico_score", -0.35 * f))
        if "dti_computed" in row.index and not pd.isna(row["dti_computed"]):
            d = min(row["dti_computed"]/100.0, 1.0); contribs.append(("dti_computed", 0.30 * d))
        if "loan_to_income" in row.index and not pd.isna(row["loan_to_income"]):
            l = min(row["loan_to_income"]/1.0, 1.0); contribs.append(("loan_to_income", 0.20 * l))
        if "int_rate" in row.index and not pd.isna(row["int_rate"]):
            try: ir = float(str(row["int_rate"]).strip().strip("%"))
            except: ir = 0.0
            contribs.append(("int_rate", 0.25 * min(ir/50.0,1.0)))
        if "has_delinquency" in row.index and not pd.isna(row["has_delinquency"]):
            hd = 1.0 if int(row["has_delinquency"]) else 0.0; contribs.append(("has_delinquency", 0.4 * hd))
        contribs = sorted(contribs, key=lambda x: -abs(x[1]))[:3]
        for j, (feat, val) in enumerate(contribs):
            if j==0: reason1[i] = humanize_reason(feat, val)
            if j==1: reason2[i] = humanize_reason(feat, val)
            if j==2: reason3[i] = humanize_reason(feat, val)

# ---------------- Results ----------------
results = df.copy()
results["pred_proba"] = probs
results["pred_bucket"] = pred_bucket
results["reason1"] = reason1
results["reason2"] = reason2
results["reason3"] = reason3
results["combined_reason"] = results[["reason1","reason2","reason3"]].fillna("").agg(" ".join, axis=1).str.strip()
results["repay_score"] = scores
repay_reasons = []
for contribs in per_row_contribs:
    items = sorted(contribs.items(), key=lambda x: -abs(x[1][2]))[:3]
    txts = []
    for feat, tup in items:
        nv, w, c = tup
        txts.append(f"{feat} (norm {nv:.2f}, w {w:.2f})")
    repay_reasons.append("; ".join(txts))
results["repay_score_reason"] = repay_reasons

st.write("### Prediction counts")
st.write(results["pred_bucket"].value_counts())

st.write("### Sample predictions (first 20 rows)")
st.dataframe(results[["pred_proba","pred_bucket","repay_score","repay_score_reason","combined_reason"]].head(20))

# ---------------- Download + Dashboard ----------------
csv_out = results.to_csv(index=False).encode("utf-8")
st.download_button("Download predictions CSV", data=csv_out, file_name="predictions.csv", mime="text/csv")

# --- Dashboard block (your requested UI) ---
st.markdown("## 📊 Dashboard — How the model reached decisions")
col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Feature weightage used for scoring")
    st.table(w_table)
    try:
        # bar chart of weights
        w_plot = w_table.set_index("feature")["weight_pct"]
        st.bar_chart(w_plot)
    except Exception:
        pass

with col2:
    st.subheader("Decision steps / pipeline (summary)")
    steps = [
        "1) Data ingestion: file upload (CSV / XLSX).",
        "2) Column standardization: detect FICO / CIBIL / income / DTI / etc.",
        "3) Optional pruning: drop irrelevant columns (configurable).",
        "4) Imputation & cleaning: numeric coercion, replace inf, median-fill.",
        "5) Feature engineering: loan_to_income, dti_computed, emp_length_years, has_delinquency, repay_score.",
        "6) Model alignment: align uploaded features to saved model's expected features (fill missing with medians).",
        "7) Prediction: model.predict_proba used when model available; else rule-based fallback.",
        "8) Explainability: SHAP or rule-based contributions → human-readable reasons.",
        "9) Scoring & bucket: map probabilities to Will Repay / Maybe / Will Not Repay (thresholds configurable).",
        "10) Export: download CSV or create final_submission.zip including app, models, outputs, report."
    ]
    for s in steps:
        st.write(s)
    st.markdown("**Key thresholds & counts**")
    st.write(f"- Low threshold (Will Repay if p <): **{low_thr:.2f}**")
    st.write(f"- High threshold (Will Not Repay if p >): **{high_thr:.2f}**")
    st.write(f"- Total uploaded rows: **{len(df)}**")
    st.write(f"- Using saved model: **{'Yes' if model is not None else 'No (rule fallback)'}**")
    if df_bin is not None:
        st.write(f"- Labeled rows available for retrain/eval: **{len(df_bin)}**")

# ---------------- Export package ----------------
if export_package:
    report_md = []
    report_md.append("# Final submission package — Credit Risk App")
    report_md.append(f"- Generated: {time.ctime()}")
    report_md.append("- Contents: app.py, models/ (if any), outputs/ (if any), this report.")
    report_md.append("\n## Detected modeling features")
    report_md.append(str(features))
    if "loan_status" in df.columns:
        report_md.append("\n## Label distribution (if present)")
        report_md.append(str(df["loan_status"].value_counts().to_dict()))
    report_md.append("\n## Notes")
    report_md.append("- Model (if trained) saved at models/xgb_model.joblib inside package.")
    report_path = os.path.join(BASE, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_md))
    with zipfile.ZipFile(EXPORT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(os.path.join(BASE, "app.py"), arcname="app.py")
        if os.path.exists(MODELS_DIR):
            for root, _, files in os.walk(MODELS_DIR):
                for fn in files:
                    zf.write(os.path.join(root, fn), arcname=os.path.join("models", fn))
        if os.path.exists(OUTPUTS_DIR):
            for root, _, files in os.walk(OUTPUTS_DIR):
                for fn in files:
                    zf.write(os.path.join(root, fn), arcname=os.path.join("outputs", fn))
        zf.write(report_path, arcname="report.md")
    st.success(f"Created package: {EXPORT_ZIP}")
    with open(EXPORT_ZIP, "rb") as f:
        st.download_button("Download final_submission.zip", data=f, file_name="final_submission.zip", mime="application/zip")

st.markdown("---")
st.write("If you want any of these dashboard pieces changed (different step wording, extra charts, or an 'audit CSV' containing dropped columns), tell me which one and I'll provide the one-line change or full patch.")

