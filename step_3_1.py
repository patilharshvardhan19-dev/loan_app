import pandas as pd
import os

fname = os.path.expanduser("~/loan_app/data/MOTHERFILE.xlsx")
df = pd.read_excel(fname, sheet_name=0, engine="openpyxl")

# Recreate derived columns
if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
    df["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
df["dti_computed"] = df.get("dti", pd.NA)

# --- Cleaning ---
print("Original shape:", df.shape)

# Drop completely empty columns
df = df.dropna(axis=1, how="all")

# Remove impossible FICO values
if "fico_score" in df.columns:
    before = len(df)
    df = df[(df["fico_score"] >= 300) & (df["fico_score"] <= 850)]
    print("Removed", before - len(df), "rows with invalid fico_score.")

# Remove negative or zero annual income
if "annual_inc" in df.columns:
    before = len(df)
    df = df[df["annual_inc"] > 0]
    print("Removed", before - len(df), "rows with nonpositive income.")

# Remove negative loan amounts
if "loan_amnt" in df.columns:
    before = len(df)
    df = df[df["loan_amnt"] > 0]
    print("Removed", before - len(df), "rows with nonpositive loan_amnt.")

print("Cleaned shape:", df.shape)

# Save first 200 cleaned rows for inspection
outfn = os.path.expanduser("~/loan_app/outputs/clean_sample.csv")
df.head(200).to_csv(outfn, index=False)
print("Wrote cleaned sample (200 rows) to:", outfn)

print("\nFirst 5 cleaned rows (showing important columns):")
keep = [c for c in ["loan_amnt", "annual_inc", "dti_computed", "fico_score", "loan_status"] if c in df.columns]
print(df[keep].head())

