import pandas as pd
import re

df = pd.read_csv("fyp_cleaned_dataset.csv", low_memory=False).fillna("")
col = df["clean_text"].astype(str)

# The 1973 Constitution's Preamble has very distinctive, unique wording —
# unlikely to appear in any other Pakistani statute.
preamble_pattern = re.compile(r"sovereignty over the entire universe belongs to almighty allah", re.IGNORECASE)
preamble_rows = col[col.str.contains(preamble_pattern)]
print(f"Preamble match rows: {preamble_rows.index.tolist()}")

# Fundamental Rights chapter heading (Part II, Chapter 1 of the Constitution)
fr_pattern = re.compile(r"FUNDAMENTAL RIGHTS", re.IGNORECASE)
fr_rows = col[col.str.contains(fr_pattern)]
print(f"'FUNDAMENTAL RIGHTS' heading rows: {fr_rows.index.tolist()[:10]}")

# Show context around the first Preamble match to confirm the real start
if len(preamble_rows) > 0:
    start = preamble_rows.index[0]
    print(f"\n--- Context around Constitution start (row {start}) ---")
    for i in range(max(0, start - 10), start + 10):
        print(f"[{i}] {col.iloc[i][:100]}")

# Find where Article 1 of the Constitution specifically appears NEAR the
# preamble (not elsewhere in the corpus)
if len(preamble_rows) > 0:
    window_start = preamble_rows.index[0]
    window_end = window_start + 2000  # generous window to scan forward
    nearby = col.iloc[window_start:window_end]
    art1_local = nearby[nearby.str.contains(r"\nArticle 1\b|^Article 1\b", regex=True)]
    print(f"\n'Article 1' near Preamble: rows {art1_local.index.tolist()[:5]}")