"""
RAG Engine — FAISS-based retrieval for LawBot India
=====================================================
1. INDEXING  — IPC, BNS, contract docs chunked → embedded → FAISS IndexFlatIP
2. RETRIEVAL — query embedded → cosine similarity search → top-k chunks
3. CONTEXT   — chunks formatted and injected into Groq prompt
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

        with open(os.path.join(DATA_DIR, "ipc_dataset.json"))      as f: ipc      = json.load(f)
        with open(os.path.join(DATA_DIR, "bns_dataset.json"))      as f: bns      = json.load(f)
        with open(os.path.join(DATA_DIR, "mappings.json"))         as f: mappings = json.load(f)
        with open(os.path.join(DATA_DIR, "contract_dataset.json")) as f: contracts= json.load(f)

        ipc_to_bns = {m["ipc_section"]: m["bns_section"] for m in mappings}
        bns_lookup = {b["section"]: b for b in bns}

        # IPC chunks — enrich with BNS info if mapping exists
        for r in ipc:
            bns_ref = ""
            if r["section"] in ipc_to_bns:
                bns_sec = ipc_to_bns[r["section"]]
                b = bns_lookup.get(bns_sec)
                if b:
                    bns_ref = f" BNS equivalent: §{b['section']} {b['title']}. Punishment: {b.get('punishment','—')}."
            full = (f"IPC Section {r['section']} {r['title']}. "
                    f"{r['description']} "
                    f"Keywords: {', '.join(r.get('keywords',[])[:8])}. "
                    f"Punishment: {r.get('punishment','—')}."
                    f"{bns_ref}")
            for chunk in _chunk(full):
                self.chunks.append({"text": chunk, "source": "IPC", "section": r["section"],
                                    "title": r["title"], "type": "criminal",
                                    "category": r.get("category","")})

        # BNS chunks
        for r in bns:
            full = (f"BNS Section {r['section']} {r['title']}. "
                    f"{r['description']} "
                    f"Punishment: {r.get('punishment','—')}. "
                    f"Keywords: {', '.join(r.get('keywords',[])[:8])}.")
            for chunk in _chunk(full):
                self.chunks.append({"text": chunk, "source": "BNS", "section": r["section"],
                                    "title": r["title"], "type": "criminal",
                                    "category": r.get("category","")})

        # Contract chunks
        for r in contracts:
            full = (f"{r['category']} under {r['act']} Section {r['section']}. "
                    f"{r['description']} "
                    f"Key clauses: {', '.join(r.get('key_clauses',[]))}. "
                    f"Red flags: {', '.join(r.get('red_flags',[]))}. "
                    f"Advice: {r.get('advice','')}. "
                    f"Keywords: {', '.join(r.get('keywords',[])[:8])}.")
            for chunk in _chunk(full):
                self.chunks.append({"text": chunk, "source": "CONTRACT",
                                    "section": r["section"], "title": r["category"],
                                    "type": "contract", "act": r.get("act",""),
                                    "red_flags": r.get("red_flags",[])})

        # Embed and build FAISS IndexFlatIP (cosine via L2-norm)
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
