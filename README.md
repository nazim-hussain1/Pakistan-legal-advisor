---
title: Pakistan Legal Advisor
emoji: ⚖️
colorFrom: green
colorTo: green
sdk: docker
pinned: false
---

# Pakistan Legal Advisor Chatbot

A multilingual RAG-powered legal chatbot for Pakistani law.
Built as a Final Year Project — BS Artificial Intelligence, 2026.

## Tech Stack
- Flask Backend
- FAISS Vector Search + BM25
- Sentence Transformers (multi-qa-mpnet)
- Google Gemini API (gemini-flash-lite-latest, fallback: gemini-2.0-flash)
- English + Urdu + Roman Urdu support

## Setup
1. Clone the repo
2. Install dependencies: `pip install -r Requirements.txt`
3. Add your API key to Secrets: `GEMINI_API_KEY`
4. Run: `python Backend.py`