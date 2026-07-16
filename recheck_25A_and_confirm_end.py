import pandas as pd
import re

df = pd.read_csv("fyp_cleaned_dataset.csv", low_memory=False).fillna("")
col = df["clean_text"].astype(str)

# 1. Article 25 should sit right before 25A (which should sit before 26).
#    Dump a window around where 19A was found (329029) forward to find
#    Article 25/25A/26 directly, whatever the exact format is.
print("--- Window 329029 to 329130 (looking for Art. 25 / 25A / 26) ---")
for i in range(329029, 329130):
    print(f"[{i}] {col.iloc[i][:100]}")

# 2. Confirm the end boundary: compare context right before/after the gap
#    between row 335893 (last dense hit) and row 342232 (next hit, big gap)
print("\n--- Context around row 335893 (suspected end of Constitution doc) ---")
for i in range(335880, 335920):
    print(f"[{i}] {col.iloc[i][:100]}")

print("\n--- Context around row 342232 (checking if it's a DIFFERENT document) ---")
for i in range(342220, 342250):
    print(f"[{i}] {col.iloc[i][:100]}")