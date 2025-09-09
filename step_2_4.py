import pandas as pd
import os

fname = os.path.expanduser("~/loan_app/data/MOTHERFILE.xlsx")
df = pd.read_excel(fname, sheet_name=0, engine="openpyxl")

# If we already created fico_score, recompute just in case
if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
    df["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

if "fico_score" in df.columns:
    # Standardize to 0–1 range assuming FICO range ~ [300, 850]
    df["fico_std"] = (df["fico_score"] - 300) / (850 - 300)
    df["fico_std"] = df["fico_std"].clip(0, 1)
    print("Created normalized 'fico_std' column (0–1 scale).")
    print("\nFirst 6 values of fico_score and fico_std:")
    print(df[["fico_score", "fico_std"]].head(6))
else:
    print("No fico_score column available, skipping.")

