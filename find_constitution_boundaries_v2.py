import pandas as pd
import re

df = pd.read_csv("fyp_cleaned_dataset.csv", low_memory=False).fillna("")
col = df["clean_text"].astype(str)

# 1. Loosen the Preamble search — allow for corrupted/spaced-out characters
#    and just match on a few distinctive short fragments separately.
fragments = ["sovereignty", "Almighty Allah", "Objectives Resolution", "trichotomy of powers"]
for frag in fragments:
    pattern = re.compile(re.escape(frag), re.IGNORECASE)
    matches = col[col.str.contains(pattern, regex=True, na=False)]
    print(f"'{frag}': {len(matches)} rows -> {matches.index.tolist()[:5]}")

# 2. For each of the 10 'FUNDAMENTAL RIGHTS' hits, print surrounding context
#    so we can tell which one is the real 1973 Constitution vs. some other Act
fr_rows = [6857, 80827, 80879, 230650, 316966, 317583, 328268, 328270, 328271, 328586]
for r in fr_rows:
    print(f"\n--- Context around row {r} ---")
    for i in range(max(0, r - 5), r + 8):
        print(f"[{i}] {col.iloc[i][:100]}")