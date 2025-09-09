import pandas as pd
import os

fname = os.path.expanduser("~/loan_app/data/MOTHERFILE.xlsx")
if not os.path.exists(fname):
    print("ERROR: file not found at", fname)
    raise SystemExit(1)

print("Loading file (this may take a few seconds)...")
# load first sheet
df = pd.read_excel(fname, sheet_name=0, engine="openpyxl")
print("Done.\n")

print("Rows:", df.shape[0], " Columns:", df.shape[1])
cols = list(df.columns)
print("\nFirst 20 column names:")
for i, c in enumerate(cols[:20], 1):
    print(f"{i:02d}. {c}")

print("\nFirst 3 rows (showing up to first 10 columns):")
display_cols = cols[:10] if len(cols) > 10 else cols
print(df[display_cols].head(3).to_string(index=False))

# Also print top unique values for likely target-like columns (if they exist)
candidates = [c for c in cols if any(k in c.lower() for k in ['status','default','risk','repay','loan_status','loanstatus','target'])]
if candidates:
    print("\nPotential target-like columns detected and their unique values (up to 10 shown):")
    for c in candidates:
        print(f"\n-- {c} :")
        print(df[c].dropna().unique()[:10])
else:
    print("\nNo obvious target-like column names detected (we will ask you later if this dataset is labeled).")

