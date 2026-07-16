import json

with open("gold_test_set.json", encoding="utf-8") as f:
    gold = json.load(f)

by_art = {g["article"]: g for g in gold}

# Fix 247: reclassify as omitted, replace bled-through text with just its
# own row content (no forward window into the next chapter/article)
a247 = by_art.get("247")
if a247:
    a247["title"] = "Omitted"
    a247["status"] = "omitted_by_amendment"
    a247["text"] = "247. [* * * * * * ] (Article omitted; provision repealed, redacted in source text)"
    print("Fixed 247: retagged omitted_by_amendment, removed bled-through text")

# Fix 203CC: trim title to stop before "Omitted by..."
a203cc = by_art.get("203CC")
if a203cc:
    a203cc["title"] = "[Penal of Ulema and Ulema members.]"
    print("Fixed 203CC: trimmed title to remove omission explanation")

final_gold = sorted(by_art.values(), key=lambda x: (len(x["article"]), x["article"]))
with open("gold_test_set.json", "w", encoding="utf-8") as f:
    json.dump(final_gold, f, indent=2, ensure_ascii=False)

verified = sum(1 for g in final_gold if g["status"] == "verified")
omitted = sum(1 for g in final_gold if g["status"] == "omitted_by_amendment")
print(f"\nFinal gold_test_set.json: {len(final_gold)} total")
print(f"  verified: {verified}")
print(f"  omitted_by_amendment: {omitted}")