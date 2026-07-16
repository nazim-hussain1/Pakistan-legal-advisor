import pandas as pd
import re

df = pd.read_csv("fyp_cleaned_dataset.csv", low_memory=False).fillna("")
col = df["clean_text"].astype(str)

START_ROW = 328669  # confirmed body start (page 1 marker, right after TOC)

# 1. Find "265. Title of Constitution and commencement" and "280. Continuance
#    of Proclamation of Emergency" in the BODY (not the TOC) to bound the end.
#    Search only from START_ROW onward, in a wide window.
window = col.iloc[START_ROW:START_ROW + 15000]
for marker in ["265.", "280.", "FIRST SCHEDULE", "SEVENTH SCHEDULE"]:
    hits = window[window.str.contains(re.escape(marker), na=False)]
    print(f"'{marker}' in body window: rows {hits.index.tolist()[:5]}")

# 2. Re-check 10A / 19A / 25A using the REAL heading format: "10A." at start
#    of a line/cell, restricted to the Constitution body range only.
for art in ["10A", "19A", "25A"]:
    pattern = re.compile(rf"^\s*{art}\.\s")
    hits = window[window.str.contains(pattern, regex=True, na=False)]
    print(f"\nArticle {art} (body-only, correct format): {len(hits)} hits -> {hits.index.tolist()[:5]}")
    for idx in hits.index.tolist()[:3]:
        print(f"   [{idx}] {col.iloc[idx][:100]}")