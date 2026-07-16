import json

with open("gold_test_set_v5.json", encoding="utf-8") as f:
    gold = json.load(f)

by_art = {g["article"]: g for g in gold}

a25 = by_art.get("25")
a25a = by_art.get("25A")

if a25 and a25a:
    # Confirm which is which by content, then swap titles only (text stays put —
    # text extraction was correct, only the title label got crossed)
    if "equal before law" in a25["text"].lower() and "free and compulsory education" in a25a["text"].lower():
        # Already correctly matched to content — just fix the mislabeled titles
        a25["title"], a25a["title"] = "Equality of citizens", "Right to education"
        print("Fixed: swapped titles for 25 and 25A to match their actual clause content")
    else:
        print("Unexpected content pattern — manual check needed:")
        print("25:", a25["text"][:100])
        print("25A:", a25a["text"][:100])

with open("gold_test_set_v6.json", "w", encoding="utf-8") as f:
    json.dump(gold, f, indent=2, ensure_ascii=False)

print(f"\nWrote gold_test_set_v6.json with corrected 25/25A")