import pandas as pd
import re
import json

df = pd.read_csv("fyp_cleaned_dataset.csv", low_memory=False).fillna("")
col = df["clean_text"].astype(str)

FILE_START, FILE_END = 328245, 335908
BODY_START = 328676

body = col.iloc[BODY_START:FILE_END + 1]
rows = list(range(BODY_START, FILE_END + 1))

# Title lines: "N. Title words" (short, no digits like "(1)" following)
# Clause lines: "N. (1) actual legal text..." OR start with "[N. clause text"
title_pattern = re.compile(r"^\s*(?:\d+\s+)?(\d{1,3}[A-Z]{0,2})\.\s+([A-Za-z].{0,80})$")
clause_pattern = re.compile(r"^\s*(?:\d+[\[\(]?\s*)?(\d{1,3}[A-Z]{0,2})\.\s*(\(1\)|\[|[A-Z].{20,})")

articles = {}  # art_num -> {"title": ..., "title_row": ..., "clause": ..., "clause_row": ...}

for i, r in enumerate(rows):
    line = col.iloc[r].strip()

    tm = title_pattern.match(line)
    if tm:
        art = tm.group(1)
        if art not in articles:
            articles[art] = {"title": tm.group(2).strip(), "title_row": r,
                              "clause": None, "clause_row": None}
        continue  # a line is either a title OR a clause start, not both

    cm = clause_pattern.match(line)
    if cm:
        art = cm.group(1)
        if art not in articles:
            articles[art] = {"title": None, "title_row": None, "clause": None, "clause_row": None}
        if articles[art]["clause"] is None:
            # gather this line + next few lines until next heading/clause/page marker
            chunk = [line]
            for r2 in rows[i+1:i+10]:
                nxt = col.iloc[r2]
                if title_pattern.match(nxt) or clause_pattern.match(nxt) or "CONSTITUTION OF PAKISTAN" in nxt:
                    break
                chunk.append(nxt)
            articles[art]["clause"] = " ".join(chunk)[:600]
            articles[art]["clause_row"] = r

# Sanity check on key articles
print(f"Total distinct article numbers: {len(articles)}\n")
for target in ["8", "9", "10", "10A", "19", "19A", "25", "25A", "184"]:
    a = articles.get(target)
    if a:
        print(f"Article {target}:")
        print(f"  title:  {a['title']}")
        print(f"  clause: {(a['clause'] or 'MISSING')[:150]}")
    else:
        print(f"Article {target}: NOT FOUND")
    print()

# How many articles have a title but NO clause text (still incomplete)?
incomplete = [k for k, v in articles.items() if v["clause"] is None]
print(f"Articles with title but no clause text found: {len(incomplete)}")
print(incomplete[:30])

# Write gold_test_set.json — only include entries that have real clause text
gold_set = []
for art_num, data in sorted(articles.items(), key=lambda x: (len(x[0]), x[0])):
    gold_set.append({
        "article": art_num,
        "title": data["title"],
        "text": data["clause"],
        "row": data["clause_row"] or data["title_row"],
        "status": "verified" if data["clause"] else "title_only_needs_review"
    })

with open("gold_test_set_v2.json", "w", encoding="utf-8") as f:
    json.dump(gold_set, f, indent=2, ensure_ascii=False)

print(f"\nWrote gold_test_set_v2.json with {len(gold_set)} entries "
      f"({sum(1 for g in gold_set if g['status']=='verified')} verified, "
      f"{sum(1 for g in gold_set if g['status']!='verified')} need review)")