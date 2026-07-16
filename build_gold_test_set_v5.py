import pandas as pd
import re
import json

df = pd.read_csv("fyp_cleaned_dataset.csv", low_memory=False).fillna("")
col = df["clean_text"].astype(str)

BODY_START = 328676
SCHEDULE_START = 334922
TOC_START, TOC_END = 328269, 328665

# --- Step 1: merge wrapped TOC lines into single logical lines ---
# A line that STARTS a new entry begins with "N." or "NA."; any line that
# does NOT start that way is a continuation of the previous entry's title
# or dots, and should be appended to it.
entry_start = re.compile(r"^\s*(\d{1,3}[A-Z]{0,2})\.\s+(.+)$")
merged_lines = []
current = None
for r in range(TOC_START, TOC_END + 1):
    line = col.iloc[r]
    m = entry_start.match(line)
    if m:
        if current:
            merged_lines.append(current)
        current = {"row": r, "art": m.group(1), "text": m.group(2)}
    else:
        if current:
            current["text"] += " " + line
if current:
    merged_lines.append(current)

# --- Step 2: extract clean title (strip dots/page number) from merged text ---
toc_entries = {}
for entry in merged_lines:
    # Title is everything before the first run of 5+ dots
    parts = re.split(r"\.{5,}", entry["text"], maxsplit=1)
    title = parts[0].strip().rstrip(".")
    if entry["art"] not in toc_entries:
        toc_entries[entry["art"]] = title

print(f"TOC ground-truth entries parsed: {len(toc_entries)}")
for t in ["8", "9", "10", "10A", "19", "19A", "24", "25", "25A"]:
    print(f"  TOC {t}: {toc_entries.get(t, 'NOT IN TOC')}")

# --- Step 3: body title-hit validation against corrected TOC (same as v4) ---
title_pattern = re.compile(r"^\s*(?:\d+\s+)?(\d{1,3}[A-Z]{0,2})\.\s+([A-Za-z].{0,80})$")
title_hits = []
for r in range(BODY_START, SCHEDULE_START):
    line = col.iloc[r].strip()
    m = title_pattern.match(line)
    if m:
        art_num, cand_title = m.group(1), m.group(2).strip()
        toc_title = toc_entries.get(art_num, "")
        if toc_title and toc_title.split(",")[0][:15].lower() in cand_title.lower():
            title_hits.append((r, art_num, toc_title))

print(f"\nConfirmed title hits: {len(title_hits)}")

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
for t in ["8", "9", "10", "10A", "19", "19A", "24", "25", "25A", "184"]:
    a = articles.get(t)
    print(f"Article {t}: {a['title'] if a else 'NOT FOUND'}")
    if a:
        print(f"   text: {a['text'][:150]}")

missing_from_body = [a for a in toc_entries if a not in articles]
print(f"\nTOC articles with no validated body match: {len(missing_from_body)}")
print(sorted(missing_from_body, key=lambda x: (len(x), x)))

gold_set = []
for art_num, data in sorted(articles.items(), key=lambda x: (len(x[0]), x[0])):
    gold_set.append({
        "article": art_num, "title": data["title"],
        "text": data["text"][:600], "row": data["row"], "status": "verified"
    })
with open("gold_test_set_v5.json", "w", encoding="utf-8") as f:
    json.dump(gold_set, f, indent=2, ensure_ascii=False)
print(f"\nWrote gold_test_set_v5.json: {len(gold_set)} verified entries")