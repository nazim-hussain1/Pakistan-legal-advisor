# Evaluation Pipeline — Fix & Run Steps

## What was wrong (summary)

1. **`retrieval.py`'s chunker** used `"\nArticle "` as its primary split
   boundary. The Constitution body in `fyp_cleaned_dataset.csv` never uses
   that string — it uses bare numeric headings (`"\n10. "`, `"\n10A. "`,
   `"\n25A. "`). `"\nArticle N"` headings DO exist elsewhere in the 566k-row
   corpus (the OIC Charter document), so the old chunker was cutting on
   article-boundaries for the *wrong* document, and never on the
   Constitution. This is exactly why `verification_report.json` showed
   T3/T5/T7/T8 resolving to OIC Charter text instead of Constitution
   provisions.
2. **`evaluate_retrieval.py` expects `{"queries": [...]}`** — a 30-item
   query file with `query`/`expected_article`/`keywords` per entry. The
   finalized `gold_test_set.json` (296 entries) is the flat *article
   corpus*, not that shape. Pointing `evaluate_retrieval.py --gold
   gold_test_set.json` at it crashes on `gold_data["queries"]`.
3. Good news: the article numbers you'd already assigned in the original
   30-query set (`gold_test_set.suggested.json`) were **correct** — 10,
   10A, 15, 16, 19, 19A, 24, 25, 25A all exist as verified entries in your
   finalized `gold_test_set.json`. Only the chunker/verifier boundary logic
   was broken, not your article-number research.

## Files provided

- **`retrieval.py`** — drop-in replacement. Adds a marker pre-pass so the
  splitter correctly treats numeric statutory headings as a primary
  boundary, alongside the existing `"\nArticle "` separator (kept for any
  other document in the corpus that genuinely uses that format).
- **`verify_gold_test_set.py`** — drop-in replacement. `ARTICLE_HEADING`
  now matches the same numeric-heading pattern as `retrieval.py`, so
  verification and real chunking agree.
- **`gold_test_set_queries.json`** — the 30-query evaluation file in the
  shape `evaluate_retrieval.py` actually consumes. This is what you pass
  as `--gold` to both `verify_gold_test_set.py` and `evaluate_retrieval.py`
  from now on. Keep `gold_test_set.json` (296 articles) as-is — it's your
  frozen reference corpus, referenced conceptually but not loaded directly
  by these two scripts.

## Steps to run locally

```bash
# 1. Back up and replace the two Python files
cp retrieval.py retrieval.py.bak
cp verify_gold_test_set.py verify_gold_test_set.py.bak
# then copy the two replacement files from this output into your project root

# 2. Force a full reindex — the OLD faiss_index.bin/chunks.npy were built
#    with the broken boundaries and are now stale
rm faiss_index.bin chunks.npy

# 3. Rebuild the index (this re-runs retrieval.py's module-level build,
#    same as starting the backend normally — takes ~3 min on first build)
python Backend.py
# Ctrl+C once you see "[READY] Flask is starting on port 7860..." —
# the index files are already written to disk by then.

# 4. Re-verify the query set against the NOW-correct boundary logic
python verify_gold_test_set.py --gold gold_test_set_queries.json \
                                --dataset fyp_cleaned_dataset.csv \
                                --out verification_report.json
# Expect T1, T2, T3, T4, T5, T6, T7, T8, T9 (all languages) to show
# VERIFIED now instead of MISMATCH/NOT_FOUND. T10 will likely stay REVIEW —
# that's expected and documented (see gold_test_set_queries.json _meta).

# 5. Run the actual retrieval evaluation for Thesis Table 4.1
python evaluate_retrieval.py --gold gold_test_set_queries.json --k 5 --candidates 20
```

`evaluate_retrieval.py` will print the Table 4.1 hit-rate matrix
(FAISS-only / BM25-only / Hybrid RRF / Hybrid+Reranker × English/Roman
Urdu/Urdu Script) plus full Precision@K/Recall@K/F1/Hit-Rate, and save
`results/retrieval_eval_raw.json`, `results/retrieval_eval_summary.json`,
and `results/retrieval_eval_summary.csv`.

## After this run

- **Freeze `gold_test_set.json`** (296 articles) as-is — don't touch it
  further, it's your verified reference corpus.
- **Freeze `gold_test_set_queries.json`** once step 4's verification comes
  back clean — this becomes your immutable 30-query evaluation set for the
  thesis, replacing every `gold_test_set_v*.json` / `_suggested.json`
  iteration.
- Paste me the terminal output from steps 4 and 5 and I'll help you drop
  the real numbers into Thesis Table 4.1 and write the accompanying
  Results/Discussion prose — right now Chapter 4 has the numbers you
  originally *hoped* for (94%/84%/89%), not numbers from an actual run
  against the fixed pipeline, so those need to be replaced once you have
  real output.
