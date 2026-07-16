"""
RAG retrieval core: dataset loading, legal-aware chunking, embedding &
reranker model loading, FAISS + BM25 hybrid retrieval with Reciprocal
Rank Fusion, and Cross-Encoder reranking.

NOTE: This module does real work at import time (loads the dataset,
builds/loads the FAISS index, loads the embedding + reranker models).
That is intentional — Backend.py imports this module once at process
startup so all heavy initialization happens before Flask starts
accepting requests, exactly like the original monolithic script.
"""
import os
import re

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Config
from translation import (
    translate_roman_urdu_query,
    translate_urdu_script_query,
    expand_query,
)

try:
    from rank_bm25 import BM25Okapi
    print("[OK] rank_bm25 loaded")
except ImportError:
    raise ImportError("Run: pip install rank-bm25")

print("[OK] faiss loaded")
print("[OK] sentence_transformers loaded")


# ═══════════════════════════════════════════════════════════
# DATASET LOADING & LEGAL-AWARE CHUNKING
# ═══════════════════════════════════════════════════════════

def read_csv_dataset(path):
    print(f"Loading dataset from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path).fillna("")
    combined = df.astype(str).agg(" ".join, axis=1)
    result = "\n".join(combined.tolist())
    print(f"[OK] Dataset loaded: {len(result):,} characters")
    return result


def preprocess_text(text):
    text = re.sub(r"\bPage \d+\b", "", text)
    text = re.sub(r"_{3,}", " ", text)
    text = re.sub(r"-{3,}", " ", text)
    text = re.sub(r"\s{3,}", "  ", text)
    return text.strip()


def create_legal_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1400, chunk_overlap=300,
        separators=[
            "\nArticle ",
            "\nPart ", "\nChapter ",
            "\n\n", "\n   (", "\n  (", "\n (",
            "\n", " ", ""
        ]
    )
    chunks = splitter.split_text(text)
    print(f"[OK] Created {len(chunks)} chunks")
    return chunks


try:
    dataset_text = read_csv_dataset(Config.DATASET_PATH)
    cleaned_text = preprocess_text(dataset_text)
    chunks_list  = create_legal_chunks(cleaned_text)
except Exception as e:
    print(f"[FATAL] Dataset loading failed: {e}")
    raise


# ═══════════════════════════════════════════════════════════
# EMBEDDING & RERANKER MODELS
# ═══════════════════════════════════════════════════════════

print("Loading embedding model...")
try:
    embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
    print("[OK] Embedding model loaded")
except Exception as e:
    print(f"[WARN] Primary model failed: {e}. Falling back...")
    embedder = SentenceTransformer(Config.EMBEDDING_MODEL_FALLBACK)
    print("[OK] Fallback embedding model loaded")

print("Loading reranker model...")
try:
    reranker = CrossEncoder(Config.RERANKER_MODEL)
    USE_RERANKER = True
    print("[OK] Reranker loaded")
except Exception as e:
    print(f"[WARN] Reranker failed: {e}")
    USE_RERANKER = False


# ═══════════════════════════════════════════════════════════
# FAISS VECTOR INDEX
# ═══════════════════════════════════════════════════════════

def build_faiss(chunks):
    print(f"Encoding {len(chunks)} chunks...")
    embeddings = embedder.encode(
        chunks, batch_size=16, convert_to_numpy=True, show_progress_bar=True
    ).astype("float32")
    faiss.normalize_L2(embeddings)
    idx = faiss.IndexFlatIP(embeddings.shape[1])
    idx.add(embeddings)
    faiss.write_index(idx, Config.INDEX_FILE)
    np.save(Config.CHUNKS_FILE, np.array(chunks, dtype=object))
    print(f"[OK] FAISS index built: {len(chunks)} vectors")
    return idx


try:
    if os.path.exists(Config.INDEX_FILE) and os.path.exists(Config.CHUNKS_FILE):
        print("Loading existing FAISS index...")
        index  = faiss.read_index(Config.INDEX_FILE)
        chunks = np.load(Config.CHUNKS_FILE, allow_pickle=True).tolist()
        print(f"[OK] FAISS index loaded: {len(chunks)} chunks")
    else:
        print("No existing index. Building...")
        index  = build_faiss(chunks_list)
        chunks = chunks_list
except Exception as e:
    print(f"[WARN] Index load failed ({e}). Rebuilding...")
    for f in [Config.INDEX_FILE, Config.CHUNKS_FILE]:
        if os.path.exists(f):
            os.remove(f)
    index  = build_faiss(chunks_list)
    chunks = chunks_list


# ═══════════════════════════════════════════════════════════
# BM25 SPARSE INDEX
# ═══════════════════════════════════════════════════════════

print("Building BM25 index...")
try:
    tokenized_chunks = [c.lower().split() for c in chunks]
    bm25     = BM25Okapi(tokenized_chunks)
    USE_BM25 = True
    print("[OK] BM25 index ready")
except Exception as e:
    print(f"[WARN] BM25 failed: {e}")
    USE_BM25 = False


# ═══════════════════════════════════════════════════════════
# HYBRID RETRIEVAL (FAISS + BM25 via Reciprocal Rank Fusion)
# ═══════════════════════════════════════════════════════════

def hybrid_retrieve(query: str, lang: str, k: int = Config.TOP_K) -> list:
    if lang == "roman_urdu":
        retrieval_query = translate_roman_urdu_query(query)
    elif lang == "ur":
        retrieval_query = translate_urdu_script_query(query)
    else:
        retrieval_query = query
    expanded = expand_query(retrieval_query)
    print(f"[EXPANDED] {expanded[:120]}...")

    q_vec = embedder.encode([expanded], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_vec)
    vec_scores, vec_indices = index.search(q_vec, min(k * 2, len(chunks)))

    rrf_scores = {}
    valid_vec_indices = []
    for rank, (score, idx) in enumerate(zip(vec_scores[0], vec_indices[0])):
        if idx < len(chunks) and score >= Config.MIN_SCORE:
            valid_vec_indices.append(int(idx))
            rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0) + 1 / (Config.RRF_K + rank + 1)

    if USE_BM25:
        try:
            bm25_scores = bm25.get_scores(expanded.lower().split())
            bm25_top_k  = np.argsort(bm25_scores)[::-1][:k * 2]
            for rank, idx in enumerate(bm25_top_k):
                rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0) + 1 / (Config.RRF_K + rank + 1)
        except Exception as e:
            print(f"[WARN] BM25 retrieval error: {e}")

    if not rrf_scores:
        return [chunks[i] for i in valid_vec_indices[:k] if i < len(chunks)]

    top_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
    return [chunks[i] for i in top_indices if i < len(chunks)]


# ═══════════════════════════════════════════════════════════
# CROSS-ENCODER RERANKING
# ═══════════════════════════════════════════════════════════

def rerank(query: str, candidates: list, top_n: int = Config.RERANK_TOP) -> list:
    if not candidates:
        return []
    if not USE_RERANKER or len(candidates) <= top_n:
        return candidates[:top_n]
    try:
        pairs  = [(query, c) for c in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [text for _, text in ranked[:top_n]]
    except Exception as e:
        print(f"[WARN] Reranker error: {e}")
        return candidates[:top_n]


def assemble_context(top_chunks: list) -> str:
    return "\n\n---\n\n".join(
        f"[Provision {i+1}]\n{chunk}" for i, chunk in enumerate(top_chunks)
    )
