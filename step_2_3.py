# ~/loan_app/step_2_3.py
import pandas as pd
import os
import re

fname = os.path.expanduser("~/loan_app/data/MOTHERFILE.xlsx")
df = pd.read_excel(fname, sheet_name=0, engine="openpyxl")

def find_cols_by_keywords(df, keywords):
    cols = []
    for k in keywords:
        for c in df.columns:
            if k in c.lower():
                cols.append(c)
    return list(dict.fromkeys(cols))  # unique preserve order

# If dti exists, copy into dti_computed
if "dti" in df.columns:
    df["dti_computed"] = pd.to_numeric(df["dti"], errors="coerce")
    print("Found existing 'dti' column. Created 'dti_computed' as copy of 'dti'.")
else:
    # try primary method: installment / (annual_inc/12)
    if ("installment" in df.columns) and ("annual_inc" in df.columns):
        # guard against zero/NaN annual_inc
        monthly_income = pd.to_numeric(df["annual_inc"], errors="coerce") / 12.0
        installment = pd.to_numeric(df["installment"], errors="coerce")
        df["dti_computed"] = (installment / monthly_income).replace([pd.np.inf, -pd.np.inf], pd.NA)
        print("Computed 'dti_computed' using installment / (annual_inc/12).")
    else:
        # look for other monthly-debt-like columns
        possible_debt_cols = find_cols_by_keywords(df, ["monthly", "monthly_debt", "total_debt", "debt_payment", "total_pymnt", "pymnt"])
        possible_income_cols = find_cols_by_keywords(df, ["annual_inc", "income", "gross_income", "annual_income"])
        if possible_debt_cols and possible_income_cols:
            debt_col = possible_debt_cols[0]
            income_col = possible_income_cols[0]
            monthly_income = pd.to_numeric(df[income_col], errors="coerce") / 12.0
            monthly_debt = pd.to_numeric(df[debt_col], errors="coerce")
            df["dti_computed"] = (monthly_debt / monthly_income).replace([pd.np.inf, -pd.np.inf], pd.NA)
            print(f"Computed 'dti_computed' using {debt_col} / ({income_col}/12).")
        else:
            print("Could not find suitable columns to compute DTI. 'dti_computed' not created.")
            df["dti_computed"] = pd.NA

# Show summary
print("\nSample of dti_computed (first 6 rows):")
print(df["dti_computed"].head(6))
print("\nCount of non-null dti_computed:", df["dti_computed"].notnull().sum())
# save a small sample to outputs for inspection
outfn = os.path.expanduser("~/loan_app/outputs/sample_with_dti.csv")
df.head(200).to_csv(outfn, index=False)
print(f"\nWrote sample (200 rows) with dti_computed to: {outfn}")

