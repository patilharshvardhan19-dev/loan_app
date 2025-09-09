# ~/loan_app/step_3_2.py
import pandas as pd
import os
from sklearn.impute import SimpleImputer
import joblib
import numpy as np

fname = os.path.expanduser("~/loan_app/data/MOTHERFILE.xlsx")
df = pd.read_excel(fname, sheet_name=0, engine="openpyxl")

# Recreate earlier derived columns if missing
if "fico_range_low" in df.columns and "fico_range_high" in df.columns and "fico_score" not in df.columns:
    df["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
if "dti" in df.columns and "dti_computed" not in df.columns:
    df["dti_computed"] = pd.to_numeric(df["dti"], errors="coerce")

# Drop columns that are completely empty
df = df.dropna(axis=1, how="all")

# Select numeric and categorical columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print("Numeric columns count:", len(num_cols))
print("Categorical columns count:", len(cat_cols))

# Create imputers
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

# Fit on the dataframe and transform
df[num_cols] = num_imputer.fit_transform(df[num_cols])
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# Save imputers for later use
os.makedirs(os.path.expanduser("~/loan_app/models"), exist_ok=True)
joblib.dump(num_imputer, os.path.expanduser("~/loan_app/models/num_imputer.joblib"))
joblib.dump(cat_imputer, os.path.expanduser("~/loan_app/models/cat_imputer.joblib"))

# Report top columns still NA (should be 0)
missing_after = df.isnull().sum().sort_values(ascending=False)
print("\nTop 10 columns by missing values after imputation:")
print(missing_after.head(10).to_string())

# Save a peek for verification
outfn = os.path.expanduser("~/loan_app/outputs/after_impute_sample.csv")
df.head(200).to_csv(outfn, index=False)
print(f"\nSaved sample after imputation to: {outfn}")
print("Saved imputers to ~/loan_app/models/ (num_imputer.joblib, cat_imputer.joblib)")

