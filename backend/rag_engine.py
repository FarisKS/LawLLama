"""
RAG Engine — FAISS-based retrieval for LawBot India
=====================================================
1. INDEXING  — IPC, BNS, contract docs chunked → embedded → FAISS IndexFlatIP
2. RETRIEVAL — query embedded → cosine similarity search → top-k chunks
3. CONTEXT   — chunks formatted and injected into Groq prompt

JSON shapes (actual):
  ipc_dataset.json      → {"metadata": {...}, "sections": [{section_number, section_title, description, keywords, ...}]}
  bns_dataset.json      → {"metadata": {...}, "sections": [{section_number, section_title, description, keywords, ...}]}
  mappings.json         → {"metadata": {...}, "mappings": [{ipc_section, bns_section, ...}]}
  contract_dataset.json → [{category, act, section, description, key_clauses, red_flags, advice, keywords}]
"""
import json, os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K      = 5
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")


def _chunk(text: str, size: int = 250) -> list:
    words = text.split()
    chunks, cur, length = [], [], 0
    for w in words:
        cur.append(w)
        length += len(w) + 1
        if length >= size:
            chunks.append(" ".join(cur))
            cur = cur[-8:]
            length = sum(len(x)+1 for x in cur)
    if cur:
        chunks.append(" ".join(cur))
    return chunks or [text]


class RAGEngine:
    def __init__(self):
        print("[RAG] Loading embedding model...")
        self.embedder = SentenceTransformer(MODEL_NAME)
        self.chunks   = []
        self.index    = None
        self._build()

    def _build(self):
        print("[RAG] Building FAISS index...")

        with open(os.path.join(DATA_DIR, "ipc_dataset.json"))      as f: _ipc = json.load(f)
        with open(os.path.join(DATA_DIR, "bns_dataset.json"))      as f: _bns = json.load(f)
        with open(os.path.join(DATA_DIR, "mappings.json"))         as f: _map = json.load(f)
        with open(os.path.join(DATA_DIR, "contract_dataset.json")) as f: _con = json.load(f)

        # Unwrap wrapper dicts:
        #   criminal datasets → {"metadata":…, "sections":[…]}
        #   mappings          → {"metadata":…, "mappings":[…]}
        #   contracts         → already a plain list
        ipc       = _ipc["sections"] if isinstance(_ipc, dict) else _ipc
        bns       = _bns["sections"] if isinstance(_bns, dict) else _bns
        mappings  = _map["mappings"] if isinstance(_map, dict) else _map
        contracts = _con["sections"] if isinstance(_con, dict) else _con

        # Criminal datasets use "section_number" / "section_title" (not "section"/"title")
        ipc_to_bns = {m["ipc_section"]: m["bns_section"] for m in mappings}
        bns_lookup = {b["section_number"]: b for b in bns}

        # ── IPC chunks ──────────────────────────────────────────────────────
        for r in ipc:
            sec   = r["section_number"]
            title = r["section_title"]
            bns_ref = ""
            if sec in ipc_to_bns:
                bns_sec = ipc_to_bns[sec]
                b = bns_lookup.get(bns_sec)
                if b:
                    bns_ref = (f" BNS equivalent: §{b['section_number']} "
                               f"{b['section_title']}.")
            full = (f"IPC Section {sec} {title}. "
                    f"{r.get('description', '')} "
                    f"Keywords: {', '.join(r.get('keywords', [])[:8])}."
                    f"{bns_ref}")
            for chunk in _chunk(full):
                self.chunks.append({
                    "text":     chunk,
                    "source":   "IPC",
                    "section":  sec,
                    "title":    title,
                    "type":     "criminal",
                    "category": r.get("category_name", r.get("category", "")),
                })

        # ── BNS chunks ──────────────────────────────────────────────────────
        for r in bns:
            sec   = r["section_number"]
            title = r["section_title"]
            full = (f"BNS Section {sec} {title}. "
                    f"{r.get('description', '')} "
                    f"Keywords: {', '.join(r.get('keywords', [])[:8])}.")
            for chunk in _chunk(full):
                self.chunks.append({
                    "text":     chunk,
                    "source":   "BNS",
                    "section":  sec,
                    "title":    title,
                    "type":     "criminal",
                    "category": r.get("category_name", r.get("category", "")),
                })

        # ── Contract chunks ─────────────────────────────────────────────────
        # contract_dataset is a flat list using "section", "category", "act" directly
        for r in contracts:
            full = (f"{r['category']} under {r['act']} Section {r['section']}. "
                    f"{r.get('description', '')} "
                    f"Key clauses: {', '.join(r.get('key_clauses', []))}. "
                    f"Red flags: {', '.join(r.get('red_flags', []))}. "
                    f"Advice: {r.get('advice', '')}. "
                    f"Keywords: {', '.join(r.get('keywords', [])[:8])}.")
            for chunk in _chunk(full):
                self.chunks.append({
                    "text":      chunk,
                    "source":    "CONTRACT",
                    "section":   r["section"],
                    "title":     r["category"],
                    "type":      "contract",
                    "act":       r.get("act", ""),
                    "red_flags": r.get("red_flags", []),
                })

        # ── Embed and build FAISS IndexFlatIP (cosine via L2-norm) ──────────
        texts = [c["text"] for c in self.chunks]
        embs  = self.embedder.encode(texts, show_progress_bar=False,
                                     batch_size=64).astype("float32")
        faiss.normalize_L2(embs)
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)
        print(f"[RAG] Index ready — {len(self.chunks)} chunks, dim={embs.shape[1]}")

    def retrieve(self, query: str, top_k: int = TOP_K, doc_type: str = None) -> list:
        q = self.embedder.encode([query], show_progress_bar=False).astype("float32")
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, top_k * 4)
        results, seen = [], set()
        for score, idx in zip(scores[0], idxs[0]):
            c   = self.chunks[idx]
            if doc_type and c["type"] != doc_type:
                continue
            key = f"{c['source']}_{c['section']}"
            if key in seen:
                continue
            seen.add(key)
            results.append({**c, "score": float(score)})
            if len(results) >= top_k:
                break
        return results

    def format_context(self, chunks: list) -> str:
        lines = []
        for i, c in enumerate(chunks, 1):
            lines.append(f"[{i}] [{c['source']} §{c['section']}] {c['title']}")
            lines.append(f"    {c['text']}")
        return "\n".join(lines)
