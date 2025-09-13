# ~/loan_app/app.py
"""
Credit Risk Assessment Model for Decision Making
- Attractive landing + full-screen loading overlay (always cleared)
- Compact dashboard layout
- Score-bands table placed NEXT TO the repayment-score pie (aligned)
- Clear explainers for repayment score distribution & applicant table
- Larger headers + larger data labels on “Prediction buckets — overall split”
- Thresholds explainer with explicit rules using the current slider values
- Detailed Reason Cards with filters (First 200 / All / bucket-wise)
- Backend math unchanged
"""
import streamlit as st
import pandas as pd
import numpy as np
import os, re, time, json, io, zipfile, gzip
import joblib
import plotly.express as px

# Optional XGBoost + SHAP (handled gracefully if not available)
try:
    from xgboost import XGBClassifier  # noqa: F401
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import shap  # noqa: F401
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# ---------- Page settings ----------
st.set_page_config(layout="wide")

# ---------- Theme / CSS ----------
st.markdown("""
<style>
/* Banking-themed background + tighter content width */
.stApp {
  background: radial-gradient(1200px 800px at 30% 10%, #e8f1ff 0%, #f6fbff 40%, #f8fbff 60%, #fdfefe 100%);
}
.block-container { padding-top: 0.8rem; padding-bottom: 1.2rem; }

/* Center hero title */
.hero { text-align:center; margin: 1.6rem 0 0.6rem 0; }
.hero h1 { font-size: 2.8rem; margin: 0; color: #b30000; font-weight: 900; }

/* Soft cards for sections */
.card {
  background: #ffffff; border: 1px solid #eaeff6; border-radius: 14px;
  padding: 12px 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); margin-bottom: 10px;
}

/* Unified section title look — one line height and alignment */
.section-title { margin: 0 0 8px 0; }
.section-title h3 {
  margin: 0; padding: 0;
  line-height: 32px; min-height: 32px;
  font-size: 24px; color: #12263a; font-weight: 900;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

/* Dataframe/table headers: darker & bold */
div[data-testid="stDataFrame"] thead th, table thead th {
  background: #f0f4fa !important; color: #1b263b !important; font-weight: 800 !important;
  border-bottom: 1px solid #d9e3f1 !important;
}

/* Status chip for mode */
.mode-chip {
  display:inline-block; padding: 6px 10px; border-radius: 999px; font-weight:800;
  background:#eef6ff; border:1px solid #d6e6ff; color:#0b5ed7; font-size: 14px;
}

/* Full-screen loading overlay */
.loading-overlay {
  position: fixed; inset: 0; background: rgba(255,255,255,0.85); z-index: 9999;
  display: flex; align-items: center; justify-content: center; flex-direction: column;
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial;
  color: #0b3d91;
}
.spinner {
  width: 56px; height: 56px; border: 6px solid #cfe2ff; border-top-color: #0b5ed7;
  border-radius: 50%; animation: spin 0.9s linear infinite; margin-bottom: 14px;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* Tighten vertical gaps */
.element-container { margin-bottom: 0.5rem !important; }

/* Reason cards */
.rcard{border:1px solid #eaeff6;border-radius:12px;padding:12px 14px;margin:10px 0;background:#fff}
.rhead{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.rtitle{font-weight:800;font-size:16px}
.rpill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;color:#fff;margin-left:6px}
.rpill.green{background:#2ecc71}.rpill.orange{background:#f39c12}.rpill.red{background:#e74c3c}
.frow{display:flex;align-items:center;gap:10px;margin:6px 0;flex-wrap:wrap}
.fname{min-width:200px;font-weight:700}
.fsub{font-size:12px;color:#666}
.mini{font-size:12px;color:#444}
.meter{display:inline-block;width:120px;height:10px;background:#ecf0f1;border-radius:6px;overflow:hidden}
.fill{height:100%}
.fill.good{background:#2ecc71}.fill.ok{background:#f39c12}.fill.bad{background:#e74c3c}
.wchip{display:inline-block;background:#eef6ff;border:1px solid #d6e6ff;color:#185adb;padding:1px 6px;border-radius:10px;font-size:11px}
</style>
""", unsafe_allow_html=True)

# ---------- Paths ----------
BASE = os.path.expanduser("~/loan_app")
MODELS_DIR = os.path.join(BASE, "models")
OUTPUTS_DIR = os.path.join(BASE, "outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.joblib")

# ---------- Helpers ----------
def normalize_colname(s): return re.sub(r'[^a-z0-9]', '', str(s).lower())

def find_column(cols, patterns):
    cols_norm = {c: normalize_colname(c) for c in cols}
    for pat in patterns:
        p = normalize_colname(pat)
        for orig, norm in cols_norm.items():
            if p in norm:
                return orig
    return None

def pick_applicant_id_col(df: pd.DataFrame):
    candidates = [
        "applicant_id","application_id","user_id","customer_id","client_id","member_id",
        "loan_id","account_id","id","uid","pan","aadhaar","ssn",
        "applicant_name","borrower_name","name","full_name"
    ]
    for c in candidates:
        if c in df.columns: return c
    fuzzy_tokens = ["applicant","application","borrower","customer","user","client","name","id","loan"]
    for col in df.columns:
        n = normalize_colname(col)
        if any(tok in n for tok in fuzzy_tokens):
            return col
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
        if found: mapping[std] = found
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
    if "fico_score" not in df.columns and {"fico_range_low","fico_range_high"}.issubset(df.columns):
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

    if "loan_to_income" not in df.columns and {"loan_amnt","annual_inc"}.issubset(df.columns):
        df["loan_to_income"] = pd.to_numeric(df["loan_amnt"], errors="coerce") / pd.to_numeric(df["annual_inc"], errors="coerce")
    if "dti" in df.columns and "dti_computed" not in df.columns:
        df["dti_computed"] = pd.to_numeric(df["dti"], errors="coerce")
    if "dti_computed" not in df.columns and {"installment","annual_inc"}.issubset(df.columns):
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
    w = {"fico_score": -0.35, "dti_computed": 0.30, "loan_to_income": 0.20, "annual_inc": -0.10,
         "emp_length_years": 0.05, "has_delinquency": 0.40, "int_rate": 0.25, "installment": 0.02}
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
    name_map = {"fico_score":"FICO score","dti_computed":"DTI (debt-to-income)","loan_to_income":"Loan-to-Income",
                "annual_inc":"Annual income","emp_length_years":"Employment length (years)","has_delinquency":"Past delinquencies",
                "int_rate":"Interest rate","installment":"Monthly installment","grade":"Loan grade","purpose":"Loan purpose"}
    pretty = name_map.get(feat, feat.replace("_"," ").title())
    direction = "increased" if val>0 else "reduced"
    mag = abs(val)
    mag_s = f"{mag:.4f}" if abs(mag) < 1 else f"{mag:.2f}"
    return f"{pretty} ({mag_s}) {direction} default risk."

def pretty_label(f):
    return {
        "fico_score":"FICO score",
        "dti_computed":"DTI (debt-to-income)",
        "loan_to_income":"Loan-to-Income",
        "annual_inc":"Annual income",
        "emp_length_years":"Employment length",
        "has_delinquency":"Past delinquencies",
        "int_rate":"Interest rate",
        "installment":"Monthly installment",
    }.get(f, f.replace("_"," ").title())

def apply_fig_style(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#12263a", size=18)  # bigger global chart font
    )
    return fig

def status_from_nv(nv: float) -> str:
    if nv >= 0.60: return "good"
    if nv >= 0.40: return "ok"
    return "bad"

def target_raw_for_feature(f, norm_info, TARGET_NORM=0.60):
    vmin, vmax = norm_info.get(f, (0.0, 1.0))
    higher_good = f not in ["dti_computed","loan_to_income","int_rate","installment","has_delinquency"]
    if higher_good:
        thr = vmin + TARGET_NORM * (vmax - vmin); sign = "≥"
    else:
        thr = vmin + (1 - TARGET_NORM) * (vmax - vmin); sign = "≤"
    return thr, sign

def format_value(f, val):
    if pd.isna(val): return "—"
    if f in ["int_rate","dti_computed"]:
        v = float(val); v = v*100.0 if v <= 1.0 else v; return f"{v:.1f}%"
    if f == "loan_to_income": return f"{float(val):.2f}×"
    if f in ["annual_inc","installment","loan_amnt"]:
        try: return f"{float(val):,.0f}"
        except: return str(val)
    if f == "fico_score": return f"{float(val):.0f}"
    if f == "emp_length_years": return f"{float(val):.0f} yrs"
    if f == "has_delinquency": return "Yes" if int(val)==1 else "No"
    return str(val)

# ---------- Landing ----------
st.markdown('<div class="hero"><h1>Credit Risk Assessment Model for Decision Making</h1></div>', unsafe_allow_html=True)

# ---------- Upload (CSV/XLS/XLSX/CSV.GZ/ZIP + optional URL) ----------
uploaded_file = st.file_uploader(
    "Upload dataset (CSV / XLS / XLSX / CSV.GZ / ZIP)",
    type=["csv","xls","xlsx","csv.gz","zip"],
    help="Tip: For large files, upload a compressed CSV (.csv.gz) or a .zip containing a CSV/XLS/XLSX. Or paste a direct URL below."
)
alt_url = st.text_input("Or paste a file URL (optional)").strip()

if not uploaded_file and not alt_url:
    st.info("Upload your dataset (e.g., MotherFile.xlsx) or paste a direct file URL to proceed.")
    st.stop()

overlay = st.empty()
overlay.markdown("""
<div class="loading-overlay">
  <div class="spinner"></div>
  <div><b>Processing your file...</b><br/>parsing, cleaning & aligning columns</div>
</div>
""", unsafe_allow_html=True)

def _read_uploaded_any(fobj, name_lower: str) -> pd.DataFrame:
    if name_lower.endswith(".csv"):
        return pd.read_csv(fobj)
    if name_lower.endswith(".csv.gz"):
        return pd.read_csv(fobj, compression="gzip")
    if name_lower.endswith(".xls") or name_lower.endswith(".xlsx"):
        return pd.read_excel(fobj, sheet_name=0, engine="openpyxl")
    if name_lower.endswith(".zip"):
        with zipfile.ZipFile(fobj) as zf:
            inner = [n for n in zf.namelist() if n.lower().endswith((".csv",".csv.gz",".xls",".xlsx"))]
            if not inner:
                raise ValueError("ZIP contains no CSV/XLS/XLSX file.")
            first = inner[0]
            with zf.open(first) as inner_file:
                if first.lower().endswith(".csv"):
                    return pd.read_csv(inner_file)
                if first.lower().endswith(".csv.gz"):
                    with gzip.open(inner_file) as gf:
                        return pd.read_csv(gf)
                return pd.read_excel(inner_file, sheet_name=0, engine="openpyxl")
    raise ValueError("Unsupported file type. Please upload CSV/XLS/XLSX/CSV.GZ or a ZIP containing one of these.")

try:
    if uploaded_file:
        df = _read_uploaded_any(uploaded_file, uploaded_file.name.lower())
    else:
        url = alt_url
        if url.lower().endswith((".csv",".csv.gz")):
            df = pd.read_csv(url)
        else:
            df = pd.read_excel(url, sheet_name=0)
    df, detected = standardize_columns(df)
    id_col = pick_applicant_id_col(df)
    id_series = df[id_col].astype(str) if id_col else pd.Series([f"Row {i+1}" for i in range(len(df))], index=df.index)
except Exception as e:
    overlay.empty()
    st.error(f"Failed to read dataset: {e}")
    st.stop()
finally:
    overlay.empty()  # always clear overlay

# ---------- Preview ----------
st.markdown('<div class="section-title"><h3>Preview — first 5 rows</h3></div>', unsafe_allow_html=True)
st.dataframe(df.head(), use_container_width=True)

# ---------- Model load (silent) ----------
model = None
if os.path.exists(MODELS_DIR) and os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        model = None

# ---------- Feature set ----------
features = [c for c in ["fico_score","dti_computed","loan_to_income","annual_inc","emp_length_years",
                        "has_delinquency","int_rate","installment","grade","purpose"] if c in df.columns]
if len(features) == 0:
    st.error("No modeling parameters detected — add fields like annual_inc, loan_amnt, fico_range_low/high, dti, installment, etc.")
    st.stop()

# Build X (encode + clean)
X = df[features].copy()
for col in ["grade","purpose"]:
    if col in X.columns:
        X[col] = X[col].astype("category").cat.codes
X = clean_feature_matrix(X)

feature_directions = {
    "fico_score": True, "annual_inc": True, "emp_length_years": True,
    "dti_computed": False, "loan_to_income": False, "int_rate": False,
    "installment": False, "has_delinquency": False
}
score_features = [f for f in ["fico_score","dti_computed","loan_to_income","annual_inc","emp_length_years",
                              "has_delinquency","int_rate","installment"] if f in X.columns]
if len(score_features) == 0:
    st.error("Detected features do not include required numeric parameters (e.g., fico_score, dti_computed, loan_to_income, etc.).")
    st.stop()

# Weights from model if available, else rule defaults
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
    rule_w = {"fico_score": 0.25, "dti_computed": 0.20, "loan_to_income": 0.15, "annual_inc": 0.15,
              "emp_length_years": 0.05, "has_delinquency": 0.15, "int_rate": 0.03, "installment": 0.02}
    total = sum(rule_w.get(f,0) for f in score_features)
    if total == 0:
        for f in score_features:
            weights[f] = 1.0/len(score_features) if len(score_features)>0 else 0.0
    else:
        for f in score_features:
            weights[f] = rule_w.get(f,0)/total

# ---------- Row 1: Available params | Selected model | Weights pie ----------
c1, c_mid, c2 = st.columns([2.2, 1.1, 2.2])

with c1:
    st.markdown('<div class="section-title"><h3>Available parameters in the dataset</h3></div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({"Parameters Detected": features}), use_container_width=True, height=220)

with c_mid:
    st.markdown('<div class="section-title"><h3>Selected scoring model</h3></div>', unsafe_allow_html=True)
    mode_title = "Trained ML model (XGBoost)" if model is not None else "Rule-based scoring"
    st.markdown(f'<div class="card"><div class="mode-chip">{mode_title}</div></div>', unsafe_allow_html=True)
    with st.expander("Why this model is being used?"):
        if model is not None:
            st.write(
                "- A trained model is available and compatible with your columns.\n"
                "- Predictions use patterns learned from historical data for more nuanced risk estimates."
            )
        else:
            st.write(
                "- This run uses the **transparent, always-available** rule-based method.\n"
                "- It’s great for quick screening and consistent scoring, even when a trained model isn’t required."
            )

with c2:
    st.markdown('<div class="section-title"><h3>Weightage allotted to each parameter</h3></div>', unsafe_allow_html=True)
    w_table = pd.DataFrame(
        [{"feature": f, "weight_pct": round(weights.get(f, 0) * 100, 2)} for f in score_features]
    ).sort_values("weight_pct", ascending=False)
    if not w_table.empty:
        fig_w = px.pie(w_table, names="feature", values="weight_pct", hole=0.35, title="Feature Weightage")
        fig_w.update_traces(textinfo="label+percent", textfont_size=16)
        st.plotly_chart(apply_fig_style(fig_w), use_container_width=True, key="weights_pie_main")
    else:
        st.info("Weights will appear once key parameters are detected.")

# ---------- Normalization ranges ----------
norm_info = {}
for f in score_features:
    col_vals = pd.to_numeric(df[f], errors="coerce").dropna() if f in df.columns else pd.Series(dtype=float)
    if col_vals.size == 0:
        if f == "fico_score": vmin, vmax = 300.0, 850.0
        elif f in ["dti_computed","loan_to_income"]: vmin, vmax = 0.0, 1.0
        else: vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.nanpercentile(col_vals, 1)) if col_vals.size>1 else float(col_vals.min())
        vmax = float(np.nanpercentile(col_vals, 99)) if col_vals.size>1 else float(col_vals.max())
        if vmin == vmax: vmin -= 1.0; vmax += 1.0
    norm_info[f] = (vmin, vmax)

def norm_value(f, v):
    if pd.isna(v): return 0.5
    vmin, vmax = norm_info.get(f, (0.0,1.0))
    try: val = float(v)
    except: return 0.5
    if vmax == vmin: scaled = 0.5
    else:
        scaled = (val - vmin) / (vmax - vmin)
        scaled = max(0.0, min(1.0, scaled))
    if f in ["dti_computed","loan_to_income","int_rate","installment","has_delinquency"]:
        scaled = 1.0 - scaled  # lower is better
    return scaled

# ---------- Repay score & contributions ----------
scores = []
per_row_contribs = []
for _, row in X.iterrows():
    s = 0.0
    contribs = {}
    for f in score_features:
        raw = row.get(f, np.nan)
        nv  = norm_value(f, raw)
        w   = weights.get(f, 0.0)
        contribs[f] = (nv, w, nv*w)
        s += nv * w
    scores.append(float(round(s * 10.0, 3)))
    per_row_contribs.append(contribs)

# ---------- Repayment score distribution (pie + table side-by-side) ----------
st.markdown('<div class="section-title"><h3>Repayment score distribution</h3></div>', unsafe_allow_html=True)
_s = pd.Series(scores)
_bins = pd.cut(_s, bins=[0,2,4,6,8,10], include_lowest=True, right=True)
_dist = (_bins.value_counts().sort_index().rename_axis("Score band").reset_index(name="Applicants"))

c_left, c_right = st.columns([1.2, 1.0])
with c_left:
    if _dist["Applicants"].sum() > 0:
        _dist["Score band"] = _dist["Score band"].map(lambda iv: f"{iv.left:.0f}–{iv.right:.0f}")
        fig_sd = px.pie(_dist, names="Score band", values="Applicants", hole=0.35,
                        title="Applicants by repayment score band")
        fig_sd.update_traces(textinfo="label+percent", textfont_size=20)
        st.plotly_chart(apply_fig_style(fig_sd), use_container_width=True, key="score_dist_pie")
    else:
        st.info("Not enough data to show the score distribution yet.")

with c_right:
    if _dist["Applicants"].sum() > 0:
        total_appl = int(_dist["Applicants"].sum())
        table_df = _dist.copy()
        table_df["Percent"] = (table_df["Applicants"] / max(total_appl,1) * 100).round(1)
        st.markdown('<div class="section-title"><h3>Score bands — Applicants & Percent</h3></div>', unsafe_allow_html=True)
        st.dataframe(table_df, use_container_width=True, height=280)

with st.expander("What is the Repayment score & how is this distribution computed?"):
    st.markdown(
        """
**Repayment score** is a **0–10** summary that blends key parameters (like FICO, DTI, income, etc.) using fixed weights:
- Higher score ⇒ **safer borrower**; lower score ⇒ **riskier borrower**.
- Example: If FICO is strong (good) and DTI is low (good), the score moves **up**. High DTI or prior delinquencies push it **down**.

**Distribution**: we place scores into 5 bands (**0–2**, **2–4**, **4–6**, **6–8**, **8–10**) and **count applicants** in each band.
The **pie** shows each band’s share. The **table** shows both counts and **percentages** for easy comparison.
"""
    )

# ---------- Prediction using model else rule ----------
probs = None
if (model is not None):
    try:
        expected = None
        try: expected = list(model.feature_names_in_)
        except Exception: expected = None
        X_aligned = X.copy()
        if expected:
            for col in expected:
                if col not in X_aligned.columns:
                    X_aligned[col] = X_aligned.median().get(col, 0)
                    X_aligned[col] = X_aligned[col].fillna(0)
            extra = [c for c in X_aligned.columns if c not in expected]
            if extra: X_aligned = X_aligned.drop(columns=extra)
            X_aligned = X_aligned[expected]
        probs = model.predict_proba(X_aligned)[:,1]  # p(default)
    except Exception:
        probs = None
if probs is None:
    probs = X.apply(rule_based_score, axis=1).values

low_thr = st.sidebar.slider("Low threshold (Will Repay if p < low)", 0.0, 0.5, 0.35, 0.01)
high_thr = st.sidebar.slider("High threshold (Will Not Repay if p > high)", 0.5, 1.0, 0.65, 0.01)
def assign_bucket(p):
    if p < low_thr: return "Will Repay"
    if p > high_thr: return "Will Not Repay"
    return "Maybe"
pred_bucket = [assign_bucket(p) for p in probs]

# ---------- Results ----------
results = df.copy()
results.insert(0, "Applicant", id_series.values)
results["pred_proba"]  = probs
results["pred_bucket"] = pred_bucket
results["repay_score"] = scores

# ---------- Top-3 drivers (SHAP if available else fallback) ----------
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
    except Exception:
        pass

def fallback_top3_text(i):
    contribs = per_row_contribs[i]
    items = sorted(contribs.items(), key=lambda x: -abs(x[1][2]))[:3]
    txts = []
    for feat, (nv, w, c) in items:
        sign = "helping" if nv >= 0.55 else "hurting" if nv <= 0.45 else "neutral"
        label = pretty_label(feat)
        txts.append(f"{label}: {sign}")
    return " | ".join(txts)

top3_col = []
for i in range(len(results)):
    if reason1[i] or reason2[i] or reason3[i]:
        bits = [r for r in [reason1[i], reason2[i], reason3[i]] if r]
        top3_col.append(" | ".join(bits))
    else:
        top3_col.append(fallback_top3_text(i))

# ---------- Prediction buckets — overall split ----------
st.markdown('<div class="section-title"><h3>Prediction buckets — overall split</h3></div>', unsafe_allow_html=True)
_cnt = results["pred_bucket"].value_counts()
_order = ["Will Repay","Maybe","Will Not Repay"]
_cnt = _cnt.reindex(_order, fill_value=0)
_total = int(len(results))
_counts_df = pd.DataFrame({"Bucket": _order,
                           "Count": [int(_cnt.get("Will Repay",0)), int(_cnt.get("Maybe",0)), int(_cnt.get("Will Not Repay",0))]})
_counts_df["All"] = "All"
_counts_df["Label"] = _counts_df.apply(lambda r: f"{r['Bucket']} — {r['Count']/max(_total,1)*100:.1f}% (n={r['Count']})", axis=1)
fig_po = px.bar(_counts_df, x="Count", y="All", color="Bucket", orientation="h", text="Label",
                title="Distribution of all applicants",
                color_discrete_map={"Will Repay":"#2ecc71","Maybe":"#f39c12","Will Not Repay":"#e74c3c"})
fig_po.update_layout(barmode="stack", xaxis=dict(range=[0,max(_total,1)], title=""),
                     yaxis_title="", legend_title="", margin=dict(l=10,r=10,t=60,b=10),
                     font=dict(size=20))
fig_po.update_traces(textposition="inside", insidetextanchor="middle", cliponaxis=False, textfont_size=20)
st.plotly_chart(apply_fig_style(fig_po), use_container_width=True, key="prediction_outcome_100bar")
st.caption(f"Thresholds — Low: p<{low_thr:.2f} ⇒ Will Repay • High: p>{high_thr:.2f} ⇒ Will Not Repay • Between = Maybe")

with st.expander("How is this split computed, and how do the thresholds affect decisions?"):
    safe_conf = (1.0 - float(low_thr)) * 100.0
    risky_conf = float(high_thr) * 100.0
    st.markdown(
        f"""
We convert each applicant’s **default probability** (*pred_proba*) into a bucket using your sliders:

- **If pred_proba < {low_thr:.2f} → Will Repay.**  
  *(means the model is at least **{safe_conf:.0f}%** confident the person is safe)*

- **If {low_thr:.2f} ≤ pred_proba ≤ {high_thr:.2f} → Maybe.**  
  *(means the model isn’t sure — it’s the middle zone)*

- **If pred_proba > {high_thr:.2f} → Will Not Repay.**  
  *(means the model is at least **{risky_conf:.0f}%** confident the person is risky)*

**Changing sliders changes the split**:  
- Lowering **Low** makes *Will Repay* **stricter** (fewer green).  
- Raising **High** makes *Will Not Repay* **stricter** (fewer red).  
- The bar above shows how applicants are distributed across the buckets after applying these rules.
"""
    )

# ---------- Applicant-wise prediction table ----------
st.markdown('<div class="section-title"><h3>Applicant-wise prediction details</h3></div>', unsafe_allow_html=True)
_candidates = [200, 100, 50, 20]
_display_n = next((n for n in _candidates if len(results) >= n), None)
if _display_n is None:
    _display_n = min(len(results), 20)

display_df = pd.DataFrame({
    "Applicant": results["Applicant"].head(_display_n).values,
    "Default probability": results["pred_proba"].head(_display_n).values,
    "Bucket (decision)": results["pred_bucket"].head(_display_n).values,
    "Repay score (0–10, higher = safer)": results["repay_score"].head(_display_n).values,
    "Top 3 drivers (for decision)": top3_col[:_display_n],
})
st.dataframe(display_df, use_container_width=True)

with st.expander("What does this table show? (detailed explanation)"):
    st.markdown(
        """
- **Applicant**: The unique ID or name taken from your file (or “Row N” if no ID column was found).
- **Default probability** (**pred_proba, p**): The estimated chance of default. **Lower is better.**
- **Bucket (decision)**: p is mapped to **Will Repay / Maybe / Will Not Repay** using your sliders (see rules above).
- **Repay score (0–10)**: A weighted 0–10 summary score; **higher means safer**. It’s **not a probability**, but generally moves with p.
- **Top 3 drivers**: The strongest parameters that helped or hurt the decision for that applicant.
"""
    )

# ---------- Detailed Reason Cards (with filters) ----------
st.markdown('<div class="section-title"><h3>Detailed reason cards (all parameters per applicant)</h3></div>', unsafe_allow_html=True)
with st.expander("Open detailed reason cards"):
    with st.expander("What are these cards & how to read them?"):
        st.markdown(
            "- Each card lists **every parameter** with a meter (Good/OK/Poor), its **weight**, a **target** for Will-Repay, and the applicant’s **actual** value.\n"
            "- Parameters toward **Good** push decisions toward *Will Repay*; **Poor** push toward *Will Not Repay*."
        )

    TARGET_NORM = 0.60
    ordered_features = sorted(score_features, key=lambda k: weights.get(k,0), reverse=True)

    def target_text_for(f):
        thr, sign = target_raw_for_feature(f, norm_info, TARGET_NORM)
        if f == "has_delinquency":
            return "Target: No past delinquencies"
        elif f in ["int_rate","dti_computed"]:
            v = thr; v = v*100.0 if v <= 1.0 else v
            return f"Target: {sign} {v:.1f}%"
        elif f == "loan_to_income":
            return f"Target: {sign} {thr:.2f}×"
        elif f in ["annual_inc","installment","loan_amnt"]:
            return f"Target: {sign} {thr:,.0f}"
        elif f == "fico_score":
            return f"Target: {sign} {thr:.0f}"
        elif f == "emp_length_years":
            return f"Target: {sign} {thr:.0f} yrs"
        else:
            return f"Target: {sign} {thr:.2f}"

    # Filters
    filter_choice = st.radio(
        "Show cards for:",
        ["First 200 applicants", "All applicants (may be slow)",
         "Applicants in Maybe bucket", "Applicants in Will Not Repay bucket", "Applicants in Will Repay bucket"],
        horizontal=True, index=0
    )

    if filter_choice.startswith("Applicants in"):
        target_bucket = "Maybe" if "Maybe" in filter_choice else ("Will Not Repay" if "Not Repay" in filter_choice else "Will Repay")
        idx = [i for i, b in enumerate(results["pred_bucket"]) if b == target_bucket]
    else:
        idx = list(range(len(results)))

    max_cards = len(idx) if filter_choice.startswith("All") else min(200, len(idx))

    def bucket_color(bucket: str) -> str:
        return "green" if bucket=="Will Repay" else "red" if bucket=="Will Not Repay" else "orange"

    for k in range(max_cards):
        i = idx[k]
        appl  = results["Applicant"].iloc[i]
        bucket= results["pred_bucket"].iloc[i]
        proba = results["pred_proba"].iloc[i]
        rscore= results["repay_score"].iloc[i]
        row   = df.iloc[i]
        contribs = per_row_contribs[i]

        st.markdown(
            f"""
<div class="rcard">
  <div class="rhead">
    <div class="rtitle">Applicant: {appl}</div>
    <div>
      <span class="rpill {bucket_color(bucket)}">{bucket}</span>
      <span class="mini">p(default): {proba:.2f}</span>
      &nbsp;|&nbsp;<span class="mini">Repay score: {rscore:.2f}</span>
    </div>
  </div>
""", unsafe_allow_html=True)

        for f in ordered_features:
            nv, w, _ = contribs.get(f, (0.5, 0.0, 0.0))
            stat = status_from_nv(nv)
            width = int(round(nv*100))
            label = pretty_label(f)
            actual = format_value(f, row.get(f, np.nan))
            wtxt = f"{weights.get(f,0)*100:.0f}%"
            ttext = target_text_for(f)
            st.markdown(
                f"""
  <div class="frow">
    <div class="fname">{label}</div>
    <div class="meter"><div class="fill {stat}" style="width:{width}%"></div></div>
    <span class="mini">{'Good' if stat=='good' else 'OK' if stat=='ok' else 'Poor'}</span>
    <span class="wchip">weight {wtxt}</span>
    <div class="fsub">{ttext}</div>
    <div class="fsub">Actual: <b>{actual}</b></div>
  </div>
""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ---------- Download CSV ----------
ordered_features = sorted(score_features, key=lambda k: weights.get(k,0), reverse=True)
reasons_json = []
for i in range(len(results)):
    row = df.iloc[i]
    contribs = per_row_contribs[i]
    items = []
    for f in ordered_features:
        nv, w, _ = contribs.get(f, (0.5, 0.0, 0.0))
        stat = status_from_nv(nv)
        thr, sign = target_raw_for_feature(f, norm_info)
        items.append({
            "parameter": pretty_label(f),
            "normalized": round(nv,3),
            "weight": round(w,3),
            "status": stat,
            "actual": format_value(f, row.get(f, np.nan)),
            "target": f"{sign} {format_value(f, thr)}" if f != "has_delinquency" else "No past delinquencies"
        })
    reasons_json.append(json.dumps(items))

out = pd.DataFrame({
    "Applicant": results["Applicant"],
    "default_probability": results["pred_proba"],
    "prediction_bucket": results["pred_bucket"],
    "repay_score": results["repay_score"],
    "top3_drivers": top3_col,
    "detailed_reasons_json": reasons_json
})
csv_bytes = out.to_csv(index=False).encode("utf-8")
st.download_button("Download predictions CSV (with detailed reasons)", data=csv_bytes,
                   file_name="predictions_with_reasons.csv", mime="text/csv")
