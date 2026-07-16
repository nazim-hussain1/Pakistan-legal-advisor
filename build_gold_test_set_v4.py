import pandas as pd
import re
import json

df = pd.read_csv("fyp_cleaned_dataset.csv", low_memory=False).fillna("")
col = df["clean_text"].astype(str)

BODY_START = 328676
SCHEDULE_START = 334922  # "FIRST SCHEDULE" — main Articles end before this

# --- Step 1: parse the TOC (328269-328665) as ground-truth (article, title) pairs ---
TOC_START, TOC_END = 328269, 328665
toc_pattern = re.compile(r"^\s*(\d{1,3}[A-Z]{0,2})\.\s+([A-Za-z].+?)\s*\.{5,}")
toc_entries = {}
for r in range(TOC_START, TOC_END + 1):
    line = col.iloc[r]
    m = toc_pattern.match(line)
    if m:
        art, title = m.group(1), m.group(2).strip()
        if art not in toc_entries:
            toc_entries[art] = title

print(f"TOC ground-truth entries parsed: {len(toc_entries)}")
for t in ["8", "9", "10", "10A", "19", "19A", "25", "25A"]:
    print(f"  TOC {t}: {toc_entries.get(t, 'NOT IN TOC')}")

# --- Step 2: find title-line hits, but ONLY within main body (excludes Schedules) ---
title_pattern = re.compile(r"^\s*(?:\d+\s+)?(\d{1,3}[A-Z]{0,2})\.\s+([A-Za-z].{0,80})$")
title_hits = []
for r in range(BODY_START, SCHEDULE_START):
    line = col.iloc[r].strip()
    m = title_pattern.match(line)
    if m:
        art_num, cand_title = m.group(1), m.group(2).strip()
        # Only accept as a REAL title if it matches (or closely matches) the TOC title
        toc_title = toc_entries.get(art_num, "")
        if toc_title and toc_title.split(",")[0][:15].lower() in cand_title.lower():
            title_hits.append((r, art_num, toc_title))  # use TOC's clean title, not the noisy body line

print(f"\nConfirmed title hits (validated against TOC): {len(title_hits)}")

# --- Step 3: extract clause text between confirmed title rows ---
junk_pattern = re.compile(r"^(CONSTITUTION OF PAKISTAN|\d{1,3})$")
articles = {}
for i, (row, art_num, toc_title) in enumerate(title_hits):
    next_row = title_hits[i + 1][0] if i + 1 < len(title_hits) else SCHEDULE_START
    chunk = []
    for r in range(row, next_row):
        line = col.iloc[r].strip()
        if junk_pattern.match(line):
            continue
        chunk.append(line)
    clause_text = " ".join(chunk).strip()
    if art_num not in articles or len(clause_text) > len(articles[art_num]["text"]):
        articles[art_num] = {"title": toc_title, "text": clause_text, "row": row}

print(f"Distinct validated articles: {len(articles)}\n")
for t in ["8", "9", "10", "10A", "19", "19A", "25", "25A", "184"]:
    a = articles.get(t)
    if a:
        print(f"Article {t}: title='{a['title']}'")
        print(f"   text: {a['text'][:150]}")
    else:
        print(f"Article {t}: NOT FOUND")
    print()

# --- Step 4: which TOC articles never got matched in the body? ---
missing_from_body = [a for a in toc_entries if a not in articles]
print(f"TOC articles with no validated body match: {len(missing_from_body)}")
print(sorted(missing_from_body, key=lambda x: (len(x), x))[:40])

gold_set = []
for art_num, data in sorted(articles.items(), key=lambda x: (len(x[0]), x[0])):
    gold_set.append({
        "article": art_num,
        "title": data["title"],
        "text": data["text"][:600],
        "row": data["row"],
        "status": "verified"
    })

with open("gold_test_set_v4.json", "w", encoding="utf-8") as f:
    json.dump(gold_set, f, indent=2, ensure_ascii=False)

print(f"\nWrote gold_test_set_v4.json: {len(gold_set)} verified entries")