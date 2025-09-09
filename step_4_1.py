import pandas as pd
import os
import matplotlib.pyplot as plt

fname = os.path.expanduser("~/loan_app/data/MOTHERFILE.xlsx")
df = pd.read_excel(fname, sheet_name=0, engine="openpyxl")

# Derived features
if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
    df["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
if "dti" in df.columns:
    df["dti_computed"] = pd.to_numeric(df["dti"], errors="coerce")
if "loan_amnt" in df.columns and "annual_inc" in df.columns:
    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"]

# --- Quick EDA ---
print("Summary stats for key features:\n")
for col in ["dti_computed","fico_score","loan_to_income"]:
    if col in df.columns:
        print(f"{col}:\n", df[col].describe(), "\n")

# Loan status distribution
if "loan_status" in df.columns:
    print("Loan status distribution:")
    print(df["loan_status"].value_counts())

# Save histograms
plt.figure(figsize=(10,6))
df["dti_computed"].dropna().hist(bins=30)
plt.title("DTI Distribution")
plt.xlabel("DTI %")
plt.ylabel("Count")
plt.savefig(os.path.expanduser("~/loan_app/outputs/hist_dti.png"))
plt.close()

plt.figure(figsize=(10,6))
df["fico_score"].dropna().hist(bins=30)
plt.title("FICO Score Distribution")
plt.xlabel("FICO")
plt.ylabel("Count")
plt.savefig(os.path.expanduser("~/loan_app/outputs/hist_fico.png"))
plt.close()

plt.figure(figsize=(10,6))
df["loan_to_income"].dropna().hist(bins=30)
plt.title("Loan-to-Income Distribution")
plt.xlabel("LTI")
plt.ylabel("Count")
plt.savefig(os.path.expanduser("~/loan_app/outputs/hist_lti.png"))
plt.close()

print("\nSaved histograms to ~/loan_app/outputs/: hist_dti.png, hist_fico.png, hist_lti.png")

