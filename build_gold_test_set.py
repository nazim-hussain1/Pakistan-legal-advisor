import pandas as pd
import re
import json

df = pd.read_csv("fyp_cleaned_dataset.csv", low_memory=False).fillna("")
col = df["clean_text"].astype(str)

# 1. Find ALL file boundary markers in the corpus
start_markers = col[col.str.contains(r"--- START OF FILE:", na=False)]
end_markers = col[col.str.contains(r"--- END OF FILE:", na=False)]
print(f"Total START markers: {len(start_markers)}")
print(f"Total END markers: {len(end_markers)}")

# 2. Locate the Constitution's own file segment specifically, using a
#    near-unique phrase from its Preamble/date line as an anchor
anchor = col[col.str.contains(r"12TH APRIL,\s*1973", na=False, regex=True)]
print(f"\n'12TH APRIL, 1973' anchor rows: {anchor.index.tolist()}")

if len(anchor) > 0:
    anchor_row = anchor.index[0]
    # Walk backward to find this file's START marker
    prior_starts = start_markers[start_markers.index < anchor_row]
    file_start = prior_starts.index.max() if len(prior_starts) else None
    # Walk forward to find this file's END marker
    later_ends = end_markers[end_markers.index > anchor_row]
    file_end = later_ends.index.min() if len(later_ends) else None
    print(f"\nConstitution file boundaries: START={file_start}, END={file_end}")

    if file_start is not None and file_end is not None:
        print(f"START line: {col.iloc[file_start][:150]}")
        print(f"Total rows in this file: {file_end - file_start}")

        # 3. Extract the body range: skip past the TOC by finding the
        #    Preamble start (the actual "Whereas sovereignty..." line)
        body = col.iloc[file_start:file_end + 1]
        preamble_hits = body[body.str.contains(r"Whereas sovereignty", na=False, regex=True)]
        body_start = preamble_hits.index.min() if len(preamble_hits) else file_start

        print(f"\nBody (post-TOC) starts at row {body_start}")

        # 4. Extract every article heading + following body text within
        #    [body_start, file_end]. Pattern: line starts with "N." or "NA."
        #    optionally preceded by stray page-number digits (like "15 25A.")
        heading_pattern = re.compile(r"^\s*(?:\d+\s+)?(\d+[A-Z]{0,2})\.\s+(.+)$")
        articles = {}
        rows = list(range(body_start, file_end + 1))
        for i, r in enumerate(rows):
            line = col.iloc[r]
            m = heading_pattern.match(line)
            if m:
                art_num, rest = m.group(1), m.group(2)
                # Collect the next few lines as body text until next heading
                body_lines = [rest]
                for r2 in rows[i+1:i+8]:
                    nxt = col.iloc[r2]
                    if heading_pattern.match(nxt) or "CONSTITUTION OF PAKISTAN" in nxt:
                        break
                    body_lines.append(nxt)
                if art_num not in articles:  # keep first occurrence only
                    articles[art_num] = {
                        "row": r,
                        "text": " ".join(body_lines)[:500]
                    }

        print(f"\nTotal distinct article numbers found: {len(articles)}")
        for target in ["8", "9", "10", "10A", "19", "19A", "25", "25A", "184"]:
            if target in articles:
                print(f"  Article {target}: row {articles[target]['row']} -> {articles[target]['text'][:100]}")
            else:
                print(f"  Article {target}: NOT FOUND")

        # 5. Write gold_test_set.json
        gold_set = []
        for art_num, data in sorted(articles.items(), key=lambda x: (len(x[0]), x[0])):
            gold_set.append({
                "article": art_num,
                "row": data["row"],
                "text_snippet": data["text"],
                "status": "verified"
            })

        with open("gold_test_set_rebuilt.json", "w", encoding="utf-8") as f:
            json.dump(gold_set, f, indent=2, ensure_ascii=False)
        print(f"\nWrote gold_test_set_rebuilt.json with {len(gold_set)} entries")