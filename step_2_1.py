import pandas as pd
import os

fname = os.path.expanduser("~/loan_app/data/MOTHERFILE.xlsx")
df = pd.read_excel(fname, sheet_name=0, engine="openpyxl")
cols = [c.lower() for c in df.columns]

# Expected important features (with synonyms)
expected = {
    "credit_score": ["fico", "cibil", "credit_score", "fico_range_high", "fico_range_low"],
    "dti": ["dti", "debt_to_income", "debt_to_income_ratio"],
    "income": ["annual_inc", "gross_income", "income"],
    "loan_amount": ["loan_amnt", "funded_amnt", "loan_amount"],
    "employment_length": ["emp_length", "employment_length"],
    "home_ownership": ["home_ownership"],
    "delinquencies": ["delinq_2yrs", "mths_since_last_delinq", "num_tl_90g_dpd_24m"],
    "loan_status": ["loan_status", "status"],
}

print("Suggested column mapping (based on names found):\n")
for key, synonyms in expected.items():
    found = [c for c in df.columns if c.lower() in synonyms]
    if found:
        print(f"{key:20s} -> {found}")
    else:
        print(f"{key:20s} -> NOT FOUND in this dataset")

