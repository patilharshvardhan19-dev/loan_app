# ~/loan_app/step_6_1_thresholds.py
import pandas as pd
import os

infn = os.path.expanduser("~/loan_app/outputs/test_reasons_readable.csv")
df = pd.read_csv(infn)

# Set thresholds
low_thr = 0.35
high_thr = 0.65

def assign_bucket(p):
    if p < low_thr:
        return "Will Repay"
    elif p > high_thr:
        return "Will Not Repay"
    else:
        return "Maybe"

df["pred_bucket"] = df["pred_proba"].apply(assign_bucket)

print("Counts per bucket (using thresholds", low_thr, high_thr, "):")
print(df["pred_bucket"].value_counts(), "\n")

print("Examples:")
for bucket in ["Will Repay","Maybe","Will Not Repay"]:
    print("\n---", bucket, "---")
    print(df[df["pred_bucket"]==bucket][["pred_proba","true","combined_reason"]].head(3).to_string(index=False))

