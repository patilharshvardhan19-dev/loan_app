import pandas as pd
import os

fname = os.path.expanduser("~/loan_app/data/MOTHERFILE.xlsx")
df = pd.read_excel(fname, sheet_name=0, engine="openpyxl")

# Count missing values
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

print("Total columns with missing values:", len(missing))
print("\nTop 15 columns with most missing values:")
print(missing.head(15).to_string())

