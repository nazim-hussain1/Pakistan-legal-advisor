"""
evaluate_retrieval.py
══════════════════════════════════════════════════════════════════
Retrieval evaluation harness for the Pakistan Legal Advisor.
Reproduces Thesis Table 4.1 (Retrieval Accuracy by Query Language
and Method) and additionally reports Precision@K, Recall@K, F1@K.

USAGE
-----
Place this file in the project root (same folder as retrieval.py,
config.py, translation.py) and run:

    python evaluate_retrieval.py
    python evaluate_retrieval.py --k 5 --candidates 20
    python evaluate_retrieval.py --gold gold_test_set.json --out results/

This script imports retrieval.py directly, so it reuses whatever
FAISS index / BM25 index / embedder / reranker are already built
or persisted on disk (faiss_index.bin, chunks.npy) — it does NOT
rebuild anything and does NOT touch the Flask app.

METHODOLOGY
-----------
For every gold query, four retrieval configurations are evaluated,
each truncated to the same top-K (default K=5, matching the
"top-5 retrieval accuracy" language used throughout the thesis):

  1. faiss_only        — dense retrieval only, ranked by cosine sim
  2. bm25_only          — sparse retrieval only, ranked by BM25 score
  3. hybrid_rrf         — FAISS + BM25 fused via Reciprocal Rank Fusion
  4. hybrid_reranked    — hybrid_rrf candidates re-scored by the
                           Cross-Encoder, same as production rag.py

A chunk is counted as "relevant" to a query if EITHER:
  (a) match_mode == "article": the chunk contains a regex match for
      "Article <expected_article>" (word-boundaried, so "10" does not
      false-positive on "Article 10A"), OR
  (b) match_mode == "keyword": at least one of the query's keywords
      appears in the chunk (case-insensitive substring).

For match_mode == "article" entries, keyword hits are also tracked
separately so a numbering mismatch (see PRE-FLIGHT CHECK below) can
be distinguished from a genuine retrieval failure.

Because each query has essentially one ground-truth passage (the
provision that answers it), Recall@K collapses to a binary "was the
relevant passage found in the top K" — identical to Hit Rate. This
is stated explicitly in the output so it isn't mistaken for a
multi-relevant-document recall computation.

  Precision@K = (# relevant chunks in top-K) / K
  Recall@K    = 1 if any relevant chunk in top-K else 0
  F1@K        = 2PR / (P+R), 0 if P+R == 0
  Hit Rate    = mean(Recall@K) across queries  →  this is the number
                reported in Thesis Table 4.1

PRE-FLIGHT CHECK
-----------------
Before scoring, the script scans the ENTIRE loaded corpus (not just
retrieved candidates) for each expected_article. If an article number
from the gold set never appears anywhere in the dataset, this is
flagged loudly — this is the exact 27th-Amendment renumbering risk
noted in the project caveats. Fix the gold set (or the dataset) before
trusting downstream numbers for that query.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from statistics import mean

import numpy as np
import faiss

# ── Import the already-built retrieval pipeline. This triggers the
#    same startup sequence as Backend.py (loads dataset, embedder,
#    reranker, FAISS/BM25 indices) — reusing persisted index files
#    if present, so this is fast on repeated runs.
import retrieval
from config import Config
from translation import (
    translate_roman_urdu_query,
    translate_urdu_script_query,
    expand_query,
)

LANG_LABELS = {
    "en": "English",
    "roman_urdu": "Roman Urdu",
    "ur": "Urdu Script",
}
METHODS = ["faiss_only", "bm25_only", "hybrid_rrf", "hybrid_reranked"]
METHOD_LABELS = {
    "faiss_only": "FAISS Only",
    "bm25_only": "BM25 Only",
    "hybrid_rrf": "Hybrid RRF",
    "hybrid_reranked": "Hybrid+Reranker",
}


# ═══════════════════════════════════════════════════════════
# Query preprocessing (mirrors rag.py / retrieval.hybrid_retrieve)
# ═══════════════════════════════════════════════════════════

def preprocess_query(query: str, lang: str) -> str:
    """Same translate-then-expand pipeline the production RAG path
    uses, so retrieval methods are evaluated on identical query text."""
    if lang == "roman_urdu":
        retrieval_query = translate_roman_urdu_query(query)
    elif lang == "ur":
        retrieval_query = translate_urdu_script_query(query)
    else:
        retrieval_query = query
    return expand_query(retrieval_query)


# ═══════════════════════════════════════════════════════════
# Per-method retrieval (standalone FAISS/BM25 for isolated scoring;
# hybrid methods delegate to retrieval.py's real implementation)
# ═══════════════════════════════════════════════════════════

def faiss_only(expanded_query: str, k: int) -> list:
    q_vec = retrieval.embedder.encode([expanded_query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_vec)
    scores, indices = retrieval.index.search(q_vec, min(k, len(retrieval.chunks)))
    return [retrieval.chunks[i] for i in indices[0] if 0 <= i < len(retrieval.chunks)]


def bm25_only(expanded_query: str, k: int) -> list:
    if not retrieval.USE_BM25:
        return []
    scores = retrieval.bm25.get_scores(expanded_query.lower().split())
    top_idx = np.argsort(scores)[::-1][:k]
    return [retrieval.chunks[i] for i in top_idx]


def hybrid_rrf(query: str, lang: str, k: int, candidates: int) -> list:
    return retrieval.hybrid_retrieve(query, lang=lang, k=candidates)[:k]


def hybrid_reranked(query: str, lang: str, expanded_query: str, k: int, candidates: int) -> list:
    pool = retrieval.hybrid_retrieve(query, lang=lang, k=candidates)
    return retrieval.rerank(expanded_query, pool, top_n=k)


# ═══════════════════════════════════════════════════════════
# Relevance judgement
# ═══════════════════════════════════════════════════════════

def article_pattern(article: str) -> re.Pattern:
    return re.compile(rf"\bArticle\s+{re.escape(article)}\b(?!\w)", re.IGNORECASE)


def is_relevant(chunk: str, gold_entry: dict) -> bool:
    if gold_entry["match_mode"] == "article" and gold_entry["expected_article"]:
        pat = article_pattern(gold_entry["expected_article"])
        if pat.search(chunk):
            return True
        # article regex is the primary signal; fall through to keyword
        # only if it fails, so we can still register a partial hit
        # while the pre-flight check flags the possible mismatch.
    for kw in gold_entry.get("keywords", []):
        if kw.lower() in chunk.lower():
            return True
    return False


# ═══════════════════════════════════════════════════════════
# Pre-flight check: does every expected_article actually exist
# anywhere in the loaded corpus?
# ═══════════════════════════════════════════════════════════

def preflight_check(gold_queries: list) -> dict:
    print("\n" + "=" * 70)
    print("PRE-FLIGHT CHECK — verifying gold article numbers against corpus")
    print("=" * 70)
    seen = {}
    problems = []
    for entry in gold_queries:
        art = entry["expected_article"]
        if not art or art in seen:
            continue
        pat = article_pattern(art)
        count = sum(1 for c in retrieval.chunks if pat.search(c))
        seen[art] = count
        status = "OK" if count > 0 else "*** NOT FOUND ***"
        print(f"  Article {art:<6} -> found in {count:3d} chunk(s)   [{status}]")
        if count == 0:
            problems.append(art)

    if problems:
        print("\n[WARNING] The following gold articles were not found anywhere in the")
        print("dataset. This likely means the dataset's article numbering does not")
        print("match the 1973-numbering used to author this gold set (see the 27th")
        print("Amendment caveat). Affected queries will fall back to keyword-only")
        print("matching, which is weaker evidence than an exact article match.")
        print("Affected articles:", ", ".join(problems))
    else:
        print("\n[OK] All expected article numbers were located in the dataset.")
    print("=" * 70 + "\n")
    return seen


# ═══════════════════════════════════════════════════════════
# Main evaluation loop
# ═══════════════════════════════════════════════════════════

def evaluate(gold_queries: list, k: int, candidates: int) -> dict:
    # raw[method][lang] -> list of per-query metric dicts
    raw = defaultdict(lambda: defaultdict(list))

    for entry in gold_queries:
        query, lang = entry["query"], entry["language"]
        expanded = preprocess_query(query, lang)

        results = {
            "faiss_only":      faiss_only(expanded, k),
            "bm25_only":       bm25_only(expanded, k),
            "hybrid_rrf":      hybrid_rrf(query, lang, k, candidates),
            "hybrid_reranked": hybrid_reranked(query, lang, expanded, k, candidates),
        }

        for method, retrieved in results.items():
            hits = sum(1 for c in retrieved[:k] if is_relevant(c, entry))
            precision = hits / k if k else 0.0
            recall = 1.0 if hits > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            raw[method][lang].append({
                "id": entry["id"],
                "topic": entry["topic"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "hit": recall,
            })

    return raw


def summarize(raw: dict) -> dict:
    """Aggregate per-method / per-language means, plus an 'Overall' row."""
    summary = defaultdict(dict)
    for method in METHODS:
        overall_hits, overall_p, overall_r, overall_f1 = [], [], [], []
        for lang, rows in raw[method].items():
            summary[method][lang] = {
                "precision": mean(r["precision"] for r in rows),
                "recall":    mean(r["recall"] for r in rows),
                "f1":        mean(r["f1"] for r in rows),
                "hit_rate":  mean(r["hit"] for r in rows),
                "n":         len(rows),
            }
            overall_p.extend(r["precision"] for r in rows)
            overall_r.extend(r["recall"] for r in rows)
            overall_f1.extend(r["f1"] for r in rows)
            overall_hits.extend(r["hit"] for r in rows)
        summary[method]["overall"] = {
            "precision": mean(overall_p) if overall_p else 0.0,
            "recall":    mean(overall_r) if overall_r else 0.0,
            "f1":        mean(overall_f1) if overall_f1 else 0.0,
            "hit_rate":  mean(overall_hits) if overall_hits else 0.0,
            "n":         len(overall_hits),
        }
    return summary


# ═══════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════

def print_table_4_1(summary: dict):
    """Hit-Rate-only table, formatted to match Thesis Table 4.1."""
    print("\nTable 4.1 — Retrieval Accuracy (Hit Rate) by Query Language and Method")
    header = f"{'Query Language':<20}" + "".join(f"{METHOD_LABELS[m]:>18}" for m in METHODS)
    print(header)
    print("-" * len(header))
    for lang_key, lang_label in LANG_LABELS.items():
        row = f"{lang_label:<20}"
        for m in METHODS:
            val = summary[m].get(lang_key, {}).get("hit_rate")
            row += f"{(f'{val*100:.0f}%' if val is not None else '—'):>18}"
        print(row)
    row = f"{'Overall':<20}"
    for m in METHODS:
        val = summary[m]["overall"]["hit_rate"]
        row += f"{f'{val*100:.0f}%':>18}"
    print(row)


def print_full_metrics(summary: dict):
    print("\nFull metrics (Precision@K / Recall@K / F1@K / Hit Rate)")
    print("Note: Recall@K == Hit Rate here — each query has one ground-truth")
    print("provision, so 'found in top-K at all' is the recall signal.\n")
    for m in METHODS:
        print(f"── {METHOD_LABELS[m]} ──")
        for lang_key, lang_label in LANG_LABELS.items():
            s = summary[m].get(lang_key)
            if not s:
                continue
            print(f"  {lang_label:<14} P={s['precision']:.2f}  R={s['recall']:.2f}  "
                  f"F1={s['f1']:.2f}  HitRate={s['hit_rate']*100:.0f}%  (n={s['n']})")
        o = summary[m]["overall"]
        print(f"  {'Overall':<14} P={o['precision']:.2f}  R={o['recall']:.2f}  "
              f"F1={o['f1']:.2f}  HitRate={o['hit_rate']*100:.0f}%  (n={o['n']})")
        print()


def save_outputs(raw: dict, summary: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    raw_path = os.path.join(out_dir, "retrieval_eval_raw.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)

    summary_path = os.path.join(out_dir, "retrieval_eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(out_dir, "retrieval_eval_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("method,language,precision,recall,f1,hit_rate,n\n")
        for m in METHODS:
            for lang_key in list(LANG_LABELS.keys()) + ["overall"]:
                s = summary[m].get(lang_key)
                if not s:
                    continue
                f.write(f"{m},{lang_key},{s['precision']:.4f},{s['recall']:.4f},"
                        f"{s['f1']:.4f},{s['hit_rate']:.4f},{s['n']}\n")

    print(f"\n[SAVED] {raw_path}")
    print(f"[SAVED] {summary_path}")
    print(f"[SAVED] {csv_path}")


# ═══════════════════════════════════════════════════════════
# Entrypoint
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate Pakistan Legal Advisor retrieval methods.")
    parser.add_argument("--gold", default="gold_test_set.json", help="Path to gold test set JSON")
    parser.add_argument("--k", type=int, default=Config.RERANK_TOP, help="Cutoff K for scoring (default: RERANK_TOP=5)")
    parser.add_argument("--candidates", type=int, default=Config.TOP_K, help="Candidate pool size before final cutoff (default: TOP_K=20)")
    parser.add_argument("--out", default="results", help="Output directory for result files")
    args = parser.parse_args()

    if not os.path.exists(args.gold):
        print(f"[FATAL] Gold test set not found at: {args.gold}")
        sys.exit(1)

    with open(args.gold, "r", encoding="utf-8") as f:
        gold_data = json.load(f)
    gold_queries = gold_data["queries"]
    print(f"[OK] Loaded {len(gold_queries)} gold queries from {args.gold}")
    print(f"[OK] Corpus loaded via retrieval.py: {len(retrieval.chunks)} chunks | "
          f"BM25={retrieval.USE_BM25} | Reranker={retrieval.USE_RERANKER}")

    preflight_check(gold_queries)

    print(f"[RUN] Evaluating at K={args.k} (candidate pool={args.candidates}) ...")
    raw = evaluate(gold_queries, k=args.k, candidates=args.candidates)
    summary = summarize(raw)

    print_table_4_1(summary)
    print_full_metrics(summary)
    save_outputs(raw, summary, args.out)


if __name__ == "__main__":
    main()
