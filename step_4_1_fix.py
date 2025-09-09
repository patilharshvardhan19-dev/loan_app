# ~/loan_app/step_4_1_fix.py
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

fname = os.path.expanduser("~/loan_app/data/MOTHERFILE.xlsx")
df = pd.read_excel(fname, sheet_name=0, engine="openpyxl")

# ensure derived columns exist
if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
    df["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
if "dti" in df.columns:
    df["dti_computed"] = pd.to_numeric(df["dti"], errors="coerce")
if "loan_amnt" in df.columns and "annual_inc" in df.columns:
    # compute loan_to_income safely
    df["loan_to_income"] = pd.to_numeric(df["loan_amnt"], errors="coerce") / pd.to_numeric(df["annual_inc"], errors="coerce")
    df["loan_to_income"] = df["loan_to_income"].replace([np.inf, -np.inf], np.nan)

# Print cleaned summary stats
print("=== Cleaned summary stats for key features ===\n")
for col in ["dti_computed","fico_score","loan_to_income"]:
    if col in df.columns:
        print(f"{col}:\n", df[col].describe(), "\n")

# Loan status distribution
if "loan_status" in df.columns:
    print("Loan status distribution:")
    print(df["loan_status"].value_counts(), "\n")

# Save histograms (skip if column missing)
outdir = os.path.expanduser("~/loan_app/outputs")
os.makedirs(outdir, exist_ok=True)

if "dti_computed" in df.columns:
    plt.figure(figsize=(8,5))
    df["dti_computed"].dropna().replace([np.inf, -np.inf], np.nan).dropna().hist(bins=30)
    plt.title("DTI Distribution")
    plt.xlabel("DTI")
    plt.ylabel("Count")
    plt.savefig(os.path.join(outdir, "hist_dti.png"))
    plt.close()

if "fico_score" in df.columns:
    plt.figure(figsize=(8,5))
    df["fico_score"].dropna().hist(bins=30)
    plt.title("FICO Score Distribution")
    plt.xlabel("FICO")
    plt.ylabel("Count")
    plt.savefig(os.path.join(outdir, "hist_fico.png"))
    plt.close()

if "loan_to_income" in df.columns:
    # remove extreme outliers > 10 for plotting clarity
    series = df["loan_to_income"].dropna().replace([np.inf, -np.inf], np.nan).dropna()
    series = series[series < 10]
    plt.figure(figsize=(8,5))
    series.hist(bins=30)
    plt.title("Loan-to-Income Distribution (values < 10)")
    plt.xlabel("LTI")
    plt.ylabel("Count")
    plt.savefig(os.path.join(outdir, "hist_lti.png"))
    plt.close()

print("Saved histograms to:", outdir)

