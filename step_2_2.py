import pandas as pd
import os

fname = os.path.expanduser("~/loan_app/data/MOTHERFILE.xlsx")
df = pd.read_excel(fname, sheet_name=0, engine="openpyxl")

if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
    df["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    print("Created new column 'fico_score' from average of fico_range_low and fico_range_high.")
    print("\nFirst 5 values of fico_score:")
    print(df["fico_score"].head())
else:
    print("FICO range columns not found. Skipping.")

