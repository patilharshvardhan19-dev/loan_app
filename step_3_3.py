import pandas as pd
import os
import re

fname = os.path.expanduser("~/loan_app/data/MOTHERFILE.xlsx")
df = pd.read_excel(fname, sheet_name=0, engine="openpyxl")

# Recreate necessary derived columns if missing
if "fico_range_low" in df.columns and "fico_range_high" in df.columns and "fico_score" not in df.columns:
    df["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
if "dti" in df.columns and "dti_computed" not in df.columns:
    df["dti_computed"] = pd.to_numeric(df["dti"], errors="coerce")

# Feature 1: Loan-to-Income ratio
if "loan_amnt" in df.columns and "annual_inc" in df.columns:
    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"]
    print("Created 'loan_to_income' feature.")

# Feature 2: Employment length (convert text to numeric)
if "emp_length" in df.columns:
    def parse_emp_length(x):
        if pd.isna(x): return None
        s = str(x).lower()
        if "10" in s: return 10
        if "<" in s: return 0
        digits = re.findall(r"\d+", s)
        return int(digits[0]) if digits else None
    df["emp_length_years"] = df["emp_length"].apply(parse_emp_length)
    print("Created 'emp_length_years' numeric feature.")

    # Bin employment length
    df["emp_length_bin"] = pd.cut(df["emp_length_years"],
                                  bins=[-1,0,2,5,10,50],
                                  labels=["<1","1-2","3-5","6-10","10+"],
                                  include_lowest=True)
    print("Created 'emp_length_bin' categorical feature.")

# Feature 3: Delinquency flag
if "delinq_2yrs" in df.columns:
    df["has_delinquency"] = (df["delinq_2yrs"] > 0).astype(int)
    print("Created 'has_delinquency' flag.")

# Save a sample with new features
outfn = os.path.expanduser("~/loan_app/outputs/derived_features_sample.csv")
df.head(200).to_csv(outfn, index=False)
print(f"\nSaved sample with derived features to: {outfn}")
print("\nPreview of derived features (first 5 rows):")
keep = [c for c in ["loan_amnt","annual_inc","loan_to_income","emp_length","emp_length_years","emp_length_bin","delinq_2yrs","has_delinquency"] if c in df.columns]
print(df[keep].head())

