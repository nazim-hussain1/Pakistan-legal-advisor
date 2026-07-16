import pandas as pd
import re
import json

df = pd.read_csv("fyp_cleaned_dataset.csv", low_memory=False).fillna("")
col = df["clean_text"].astype(str)

FILE_START, FILE_END = 328245, 335908
BODY_START = 328676

body_rows = list(range(BODY_START, FILE_END + 1))

title_pattern = re.compile(r"^\s*(?:\d+\s+)?(\d{1,3}[A-Z]{0,2})\.\s+([A-Za-z].{0,80})$")

# Pass 1: collect every title row in document order (don't dedupe yet —
# we need the actual sequence to bound clause windows correctly)
title_hits = []
for r in body_rows:
    line = col.iloc[r].strip()
    m = title_pattern.match(line)
    if m:
        title_hits.append((r, m.group(1), m.group(2).strip()))

print(f"Total title-line hits (raw, before dedup): {len(title_hits)}")

# Junk lines to strip out of collected body text
junk_pattern = re.compile(r"^(CONSTITUTION OF PAKISTAN|\d{1,3})$")

# Pass 2: for each title hit, body = rows between this title and the NEXT
# title hit (any article number), filtered of page-break junk.
articles = {}
for i, (row, art_num, title_text) in enumerate(title_hits):
    next_row = title_hits[i + 1][0] if i + 1 < len(title_hits) else FILE_END
    chunk = []
    for r in range(row + 1, next_row):
        line = col.iloc[r].strip()
        if junk_pattern.match(line):
            continue
        chunk.append(line)
    clause_text = " ".join(chunk).strip()
    # Keep the occurrence with the LONGEST clause text if the article number
    # repeats (title-only duplicate rows will lose to the real one)
    if art_num not in articles or len(clause_text) > len(articles[art_num]["text"]):
        articles[art_num] = {"title": title_text, "text": clause_text, "row": row}

print(f"Distinct article numbers after dedup: {len(articles)}\n")

for target in ["8", "9", "10", "10A", "19", "19A", "25", "25A", "184"]:
    a = articles.get(target)
    if a:
        print(f"Article {target}: title='{a['title']}'")
        print(f"   text: {a['text'][:150]}")
    else:
        print(f"Article {target}: NOT FOUND")
    print()

empty = [k for k, v in articles.items() if len(v["text"]) < 20]
print(f"Articles with suspiciously short/empty text (<20 chars): {len(empty)}")
print(empty[:30])

gold_set = []
for art_num, data in sorted(articles.items(), key=lambda x: (len(x[0]), x[0])):
    gold_set.append({
        "article": art_num,
        "title": data["title"],
        "text": data["text"][:600],
        "row": data["row"],
        "status": "verified" if len(data["text"]) >= 20 else "needs_review"
    })

with open("gold_test_set_v3.json", "w", encoding="utf-8") as f:
    json.dump(gold_set, f, indent=2, ensure_ascii=False)

print(f"\nWrote gold_test_set_v3.json: {len(gold_set)} total, "
      f"{sum(1 for g in gold_set if g['status']=='verified')} verified, "
      f"{sum(1 for g in gold_set if g['status']=='needs_review')} need review")