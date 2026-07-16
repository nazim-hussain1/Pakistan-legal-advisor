import json

with open("gold_test_set_v5.json", encoding="utf-8") as f:
    gold = json.load(f)

by_art = {g["article"]: g for g in gold}

# --- Fix 1: swap the full 25 / 25A records (content, not just titles) ---
a25, a25a = by_art.get("25"), by_art.get("25A")
if a25 and a25a:
    a25["text"], a25a["text"] = a25a["text"], a25["text"]
    a25["title"], a25a["title"] = "Equality of citizens", "Right to education"
    a25["row"], a25a["row"] = a25a["row"], a25["row"]
    print("Swapped 25 <-> 25A records (content confirmed via manual check)")

# --- Fix 2: recover the 28 articles that exist in body but failed the ---
#     over-strict TOC-title validation. Re-extract them directly by row,
#     using the row numbers you already confirmed above.
recovered_rows = {
    "47": 329382, "91": 330463, "96": 330566, "103": 330650, "104": 330660,
    "105": 330667, "106": 330699, "133": 331156, "134": 331159, "135": 331162,
    "167": 331749, "171": 331819, "197": 332281, "214": 333040, "216": 333084,
    "223": 333206, "229": 333396, "247": 333905, "256": 334052, "260": 334081,
    "262": 334233, "270": 334392, "272": 334653, "152A": 331393,
    "212A": 332974, "212B": 332980, "224A": 333300, "203CC": 332570,
}
omitted_articles = {"96", "134", "135", "152A", "212A", "212B", "203CC", "247"}

import pandas as pd
df = pd.read_csv("fyp_cleaned_dataset.csv", low_memory=False).fillna("")
col = df["clean_text"].astype(str)

for art, row in recovered_rows.items():
    if art in omitted_articles:
        title_line = col.iloc[row]
        title = title_line.split(".", 1)[1].strip() if "." in title_line else title_line
        by_art[art] = {
            "article": art, "title": title.replace("[Omitted ]", "").strip() or "Omitted",
            "text": col.iloc[row:row+3].str.cat(sep=" "),
            "row": row, "status": "omitted_by_amendment"
        }
    else:
        chunk = " ".join(col.iloc[row:row+8].tolist())
        title_line = col.iloc[row]
        title = title_line.split(".", 1)[1].strip() if "." in title_line else title_line
        by_art[art] = {
            "article": art, "title": title,
            "text": chunk[:600], "row": row, "status": "verified"
        }

print(f"Recovered {len(recovered_rows)} previously-missing articles")

final_gold = sorted(by_art.values(), key=lambda x: (len(x["article"]), x["article"]))
with open("gold_test_set.json", "w", encoding="utf-8") as f:
    json.dump(final_gold, f, indent=2, ensure_ascii=False)

verified = sum(1 for g in final_gold if g["status"] == "verified")
omitted = sum(1 for g in final_gold if g["status"] == "omitted_by_amendment")
print(f"\nFinal gold_test_set.json: {len(final_gold)} total articles")
print(f"  verified: {verified}")
print(f"  omitted_by_amendment: {omitted}")