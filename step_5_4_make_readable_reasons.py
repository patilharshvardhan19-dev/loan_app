# ~/loan_app/step_5_4_make_readable_reasons.py
import pandas as pd
import os
import math

infn = os.path.expanduser("~/loan_app/outputs/test_top_reasons.csv")
if not os.path.exists(infn):
    print("ERROR: expected file not found:", infn)
    raise SystemExit(1)

df = pd.read_csv(infn)

def humanize(feat_sh):
    # expects format like "dti_computed:12.3456" or "grade:3.0"
    if pd.isna(feat_sh): 
        return ""
    try:
        feat, val = feat_sh.split(":")
        val = float(val)
    except Exception:
        # fallback: just return the string
        return str(feat_sh)
    # Determine direction: positive shap increases default probability
    direction = "increased" if val > 0 else "reduced"
    # Make a simple human label mapping for nicer names
    name_map = {
        "fico_score": "FICO score",
        "dti_computed": "DTI (debt-to-income)",
        "loan_to_income": "Loan-to-Income",
        "annual_inc": "Annual income",
        "emp_length_years": "Employment length (years)",
        "has_delinquency": "Past delinquencies",
        "int_rate": "Interest rate",
        "installment": "Monthly installment",
        "grade": "Loan grade",
        "purpose": "Loan purpose"
    }
    pretty = name_map.get(feat, feat.replace("_", " ").title())
    # Format magnitude
    mag = abs(val)
    # readable magnitude rounding depending on size
    if abs(mag) >= 1:
        mag_s = f"{mag:.2f}"
    else:
        mag_s = f"{mag:.4f}"
    return f"{pretty} ({mag_s}) {direction} default risk."

# Build combined readable reasons
for c in ["reason1","reason2","reason3"]:
    if c not in df.columns:
        df[c] = ""

df["readable_reason1"] = df["reason1"].apply(humanize)
df["readable_reason2"] = df["reason2"].apply(humanize)
df["readable_reason3"] = df["reason3"].apply(humanize)

def combine_row(r):
    parts = [r["readable_reason1"], r["readable_reason2"], r["readable_reason3"]]
    parts = [p for p in parts if p]
    if not parts:
        return ""
    return " ".join(parts)

df["combined_reason"] = df.apply(combine_row, axis=1)

outfn = os.path.expanduser("~/loan_app/outputs/test_reasons_readable.csv")
df.to_csv(outfn, index=False)
print("Wrote human-readable reasons to:", outfn)
print("\nSample (first 6 rows):")
print(df[["test_index","pred_proba","true","readable_reason1","readable_reason2","readable_reason3","combined_reason"]].head(6).to_string(index=False))

