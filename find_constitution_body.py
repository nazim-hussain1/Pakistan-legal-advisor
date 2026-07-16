import pandas as pd
import re

df = pd.read_csv("fyp_cleaned_dataset.csv", low_memory=False).fillna("")
col = df["clean_text"].astype(str)

# 1. All occurrences of the exact title "CONSTITUTION OF PAKISTAN"
title_rows = col[col.str.contains(r"CONSTITUTION OF PAKISTAN", case=False, na=False, regex=True)]
print(f"'CONSTITUTION OF PAKISTAN' rows: {title_rows.index.tolist()}")

# 2. Print a wide window around row 328591 to see where TOC ends and
#    real article body (with substantive legal text, not dotted page refs) begins
print("\n--- Window 328590 to 328750 ---")
for i in range(328590, min(len(df), 328750)):
    print(f"[{i}] {col.iloc[i][:110]}")