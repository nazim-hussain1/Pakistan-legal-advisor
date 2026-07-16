"""
verify_gold_test_set.py
══════════════════════════════════════════════════════════════════
Verifies (or corrects) gold_test_set.json against your ACTUAL
fyp_cleaned_dataset.csv — instead of trusting hand-typed article
numbers, which risk drifting from the real numbering after the
27th Amendment restructuring.

WHY THIS IS SEPARATE FROM evaluate_retrieval.py
------------------------------------------------
This script does NOT import retrieval.py, so it does NOT load the
embedder, reranker, FAISS, or BM25 — it only needs pandas + re.
That makes it fast (seconds, not minutes) and runnable before you've
even built the index, purely to sanity-check the gold set itself.

BOUNDARY DEFINITION (important)
--------------------------------
retrieval.py's RecursiveCharacterTextSplitter treats "\\nArticle "
(a literal newline immediately followed by "Article ") as a primary
chunk boundary. This script uses the SAME boundary rule to find real
article headings — not just any mention of "Article N" anywhere in
the text (which would false-positive on cross-references like
"...as provided in Article 25..." appearing inside a different
article's body).

USAGE
-----
    python verify_gold_test_set.py

    python verify_gold_test_set.py --dataset fyp_cleaned_dataset.csv \\
                                    --gold gold_test_set.json \\
                                    --out verification_report.json

    python verify_gold_test_set.py --suggest-corrections

OUTPUT
------
1. Console report: for every gold entry, one of:
     VERIFIED   — article heading found, and its content contains at
                  least one of the entry's keywords (good sign the
                  number AND the topic still line up)
     MISMATCH   — article heading found, but none of the keywords
                  appear in its content (number exists, but may now
                  cover a different topic — classic renumbering risk)
     NOT_FOUND  — no heading for that article number exists at all
                  in the dataset (numbering has shifted or article
                  was removed/renumbered)
     REVIEW     — match_mode="keyword" entries (no expected number
                  to check); shows ranked keyword-match candidates
                  for manual inspection

2. verification_report.json — full machine-readable report.

3. (only with --suggest-corrections) gold_test_set.suggested.json —
   a COPY of the gold set with expected_article corrected ONLY where
   exactly one unambiguous high-confidence candidate was found. This
   never overwrites your original file, and anything ambiguous is
   left untouched for you to decide by hand.
"""

import argparse
import json
import os
import re
import sys

import pandas as pd

# Same boundary + cleaning logic as retrieval.py, duplicated here
# deliberately so this script has zero heavy dependencies.
ARTICLE_HEADING = re.compile(r"\nArticle\s+(\d+[A-Za-z]*)\b", re.IGNORECASE)


def read_csv_dataset(path: str) -> str:
    print(f"[LOAD] Reading dataset from: {path}")
    if not os.path.exists(path):
        print(f"[FATAL] File not found: {path}")
        sys.exit(1)
    try:
        df = pd.read_csv(path, low_memory=False).fillna("")
    except UnicodeDecodeError:
        df = pd.read_csv(path, low_memory=False, encoding="latin-1").fillna("")
    combined = df.astype(str).agg(" ".join, axis=1)
    text = "\n".join(combined.tolist())
    print(f"[OK] Loaded {len(text):,} characters, {len(df):,} rows")
    return text


def preprocess_text(text: str) -> str:
    # Identical to retrieval.preprocess_text — must match exactly so
    # boundaries line up with what actually gets chunked/indexed.
    text = re.sub(r"\bPage \d+\b", "", text)
    text = re.sub(r"_{3,}", " ", text)
    text = re.sub(r"-{3,}", " ", text)
    text = re.sub(r"\s{3,}", "  ", text)
    return text.strip()


def build_article_index(text: str, max_block_chars: int = 4000) -> dict:
    """Returns {article_number: block_text} using the same '\\nArticle '
    boundary retrieval.py's chunker relies on. First occurrence of a
    given number wins (duplicates logged as a warning)."""
    matches = list(ARTICLE_HEADING.finditer(text))
    print(f"[INDEX] Found {len(matches)} article headings using '\\nArticle ' boundary")

    index = {}
    duplicates = []
    for i, m in enumerate(matches):
        num = m.group(1).upper()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:min(end, start + max_block_chars)]
        if num in index:
            duplicates.append(num)
        else:
            index[num] = block

    if duplicates:
        uniq = sorted(set(duplicates))
        print(f"[WARN] {len(uniq)} article number(s) appear as headings more than once "
              f"(keeping first occurrence): {', '.join(uniq[:15])}"
              f"{' ...' if len(uniq) > 15 else ''}")
    return index


def keyword_hits(block: str, keywords: list) -> list:
    block_lower = block.lower()
    return [kw for kw in keywords if kw.lower() in block_lower]


def rank_candidates(article_index: dict, keywords: list, top_n: int = 5) -> list:
    scored = []
    for num, block in article_index.items():
        hits = keyword_hits(block, keywords)
        if hits:
            scored.append({"article": num, "hits": hits, "hit_count": len(hits)})
    scored.sort(key=lambda x: x["hit_count"], reverse=True)
    return scored[:top_n]


def verify_entry(entry: dict, article_index: dict) -> dict:
    keywords = entry.get("keywords", [])

    if entry["match_mode"] == "keyword" or not entry.get("expected_article"):
        candidates = rank_candidates(article_index, keywords)
        return {
            "id": entry["id"], "topic": entry["topic"], "language": entry["language"],
            "status": "REVIEW",
            "expected_article": entry.get("expected_article"),
            "candidates": candidates,
            "note": "No expected article to verify — ranked candidates shown for manual review.",
        }

    art = entry["expected_article"].upper()
    if art not in article_index:
        candidates = rank_candidates(article_index, keywords)
        return {
            "id": entry["id"], "topic": entry["topic"], "language": entry["language"],
            "status": "NOT_FOUND",
            "expected_article": art,
            "candidates": candidates,
            "note": f"Article {art} has no heading in the dataset. "
                    f"{'Top candidate(s) by keyword match shown below.' if candidates else 'No keyword candidates found either — topic may not be covered by this dataset at all.'}",
        }

    block = article_index[art]
    hits = keyword_hits(block, keywords)
    if hits:
        return {
            "id": entry["id"], "topic": entry["topic"], "language": entry["language"],
            "status": "VERIFIED",
            "expected_article": art,
            "matched_keywords": hits,
            "preview": block[:220].replace("\n", " ") + "...",
        }
    else:
        candidates = rank_candidates(article_index, keywords)
        return {
            "id": entry["id"], "topic": entry["topic"], "language": entry["language"],
            "status": "MISMATCH",
            "expected_article": art,
            "preview": block[:220].replace("\n", " ") + "...",
            "candidates": candidates,
            "note": f"Article {art} exists but none of the topic keywords appear in its "
                    f"content — it may now cover a different subject after renumbering.",
        }


def print_report(results: list):
    print("\n" + "=" * 78)
    print(f"{'ID':<8}{'Status':<12}{'Article':<10}{'Topic'}")
    print("=" * 78)
    counts = {"VERIFIED": 0, "MISMATCH": 0, "NOT_FOUND": 0, "REVIEW": 0}
    for r in results:
        counts[r["status"]] += 1
        art = r["expected_article"] or "—"
        print(f"{r['id']:<8}{r['status']:<12}{art:<10}{r['topic']}")
    print("=" * 78)
    print(f"VERIFIED: {counts['VERIFIED']}   MISMATCH: {counts['MISMATCH']}   "
          f"NOT_FOUND: {counts['NOT_FOUND']}   REVIEW: {counts['REVIEW']}")

    problems = [r for r in results if r["status"] in ("MISMATCH", "NOT_FOUND", "REVIEW")]
    if problems:
        print("\n" + "-" * 78)
        print("DETAIL — entries needing attention")
        print("-" * 78)
        for r in problems:
            print(f"\n[{r['id']}] {r['topic']} ({r['language']}) — {r['status']}")
            print(f"  {r.get('note', '')}")
            if r["status"] == "MISMATCH":
                print(f"  Current content preview: {r['preview']}")
            for c in r.get("candidates", []):
                print(f"    candidate: Article {c['article']}  "
                      f"(matched: {', '.join(c['hits'])})")


def write_suggested_corrections(gold_data: dict, results: list, out_path: str):
    """Applies a correction ONLY when there's exactly one unambiguous
    candidate — never overwrites the original gold_test_set.json."""
    by_id = {r["id"]: r for r in results}
    applied, skipped = [], []

    corrected = json.loads(json.dumps(gold_data))  # deep copy
    for entry in corrected["queries"]:
        r = by_id.get(entry["id"])
        if not r or r["status"] not in ("NOT_FOUND", "MISMATCH", "REVIEW"):
            continue
        candidates = r.get("candidates", [])
        if len(candidates) == 1 and candidates[0]["hit_count"] >= 2:
            old = entry["expected_article"]
            entry["expected_article"] = candidates[0]["article"]
            entry["match_mode"] = "article"
            applied.append((entry["id"], old, candidates[0]["article"]))
        else:
            skipped.append(entry["id"])

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(corrected, f, indent=2, ensure_ascii=False)

    print(f"\n[SUGGESTED] Wrote {out_path}")
    if applied:
        print("  Auto-corrected (single unambiguous candidate, >=2 keyword hits):")
        for qid, old, new in applied:
            print(f"    {qid}: {old!r} -> Article {new}")
    if skipped:
        print(f"  Left unchanged (ambiguous or no strong candidate) — review by hand: "
              f"{', '.join(skipped)}")
    print("  This file does NOT overwrite gold_test_set.json. Review the diff, then "
          "manually copy over what you accept.")


def main():
    parser = argparse.ArgumentParser(description="Verify gold_test_set.json against the real dataset.")
    parser.add_argument("--dataset", default="fyp_cleaned_dataset.csv")
    parser.add_argument("--gold", default="gold_test_set.json")
    parser.add_argument("--out", default="verification_report.json")
    parser.add_argument("--suggest-corrections", action="store_true",
                         help="Also write gold_test_set.suggested.json with unambiguous fixes applied")
    args = parser.parse_args()

    if not os.path.exists(args.gold):
        print(f"[FATAL] Gold test set not found: {args.gold}")
        sys.exit(1)
    with open(args.gold, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    text = preprocess_text(read_csv_dataset(args.dataset))
    article_index = build_article_index(text)

    results = [verify_entry(entry, article_index) for entry in gold_data["queries"]]
    print_report(results)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {args.out}")

    if args.suggest_corrections:
        write_suggested_corrections(gold_data, results, "gold_test_set.suggested.json")


if __name__ == "__main__":
    main()