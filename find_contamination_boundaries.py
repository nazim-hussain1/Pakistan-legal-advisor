import pandas as pd
import re

df = pd.read_csv("fyp_cleaned_dataset.csv", low_memory=False).fillna("")
col = df["clean_text"].astype(str)

# Flag rows that clearly belong to the intruding document
pattern = re.compile(r"Islamic Conference|Organization of the Islamic|\bOIC\b", re.IGNORECASE)
flagged = col[col.str.contains(pattern)]

print(f"Flagged rows: {len(flagged)}")
print(f"Row index range: {flagged.index.min()} to {flagged.index.max()}")
print(f"Total rows in dataset: {len(df)}")

# Are the flagged rows clustered together, or scattered across the file?
idx = sorted(flagged.index.tolist())
gaps = [idx[i+1] - idx[i] for i in range(len(idx)-1)]
print(f"Largest gap between consecutive flagged rows: {max(gaps) if gaps else 'N/A'}")
print(f"Smallest/median gap: {min(gaps) if gaps else 'N/A'} / {sorted(gaps)[len(gaps)//2] if gaps else 'N/A'}")

# Show a window of context around the FIRST flagged row, to find where
# the foreign document actually begins (its title/heading)
first_idx = idx[0]
print(f"\n--- Context BEFORE first flagged row ({first_idx}) ---")
for i in range(max(0, first_idx - 15), first_idx + 1):
    print(f"[{i}] {col.iloc[i][:120]}")

# Show a window around the LAST flagged row, to find where Constitution
# text resumes afterward
last_idx = idx[-1]
print(f"\n--- Context AFTER last flagged row ({last_idx}) ---")
for i in range(last_idx, min(len(df), last_idx + 15)):
    print(f"[{i}] {col.iloc[i][:120]}")