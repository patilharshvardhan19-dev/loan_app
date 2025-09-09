import pandas as pd
import os

fname = os.path.expanduser("~/loan_app/data/MOTHERFILE.xlsx")
df = pd.read_excel(fname, sheet_name=0, engine="openpyxl")

# Count unique values in loan_status
if "loan_status" in df.columns:
    counts = df["loan_status"].value_counts(dropna=False)
    print("Loan status value counts:\n")
    print(counts.to_string())
else:
    print("No 'loan_status' column found.")

