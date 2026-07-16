import json

with open("gold_test_set.json", encoding="utf-8") as f:
    gold = json.load(f)

by_art = {g["article"]: g for g in gold}

# 1. Confirm the 25/25A swap actually landed correctly
for art in ["25", "25A"]:
    a = by_art[art]
    print(f"Article {art}: title='{a['title']}'")
    print(f"   text: {a['text'][:120]}\n")

# 2. Confirm omitted articles are tagged and readable
print("--- Omitted articles ---")
for g in gold:
    if g["status"] == "omitted_by_amendment":
        print(f"  {g['article']}: title='{g['title']}' | text: {g['text'][:80]}")

# 3. Spot-check a few of the recovered 28 for clean boundaries (no bleed
#    into next article's title)
print("\n--- Spot-check recovered articles ---")
for art in ["91", "197", "260", "47"]:
    a = by_art[art]
    print(f"  {art}: title='{a['title']}'")
    print(f"     text: {a['text'][:150]}\n")

# 4. Any duplicate/empty titles left over as a sanity net
empties = [g["article"] for g in gold if not g["title"] or len(g["title"]) < 3]
print(f"Articles with empty/near-empty titles: {empties}")