# ⚖️ LawBot India — AI Legal Assistant

BTech 3rd Year Project — Flask + BART + FAISS RAG + Groq/LLaMA 3.3

## Architecture

```
Input (Text / PDF)
      ↓
[Privacy]     — mask Aadhaar, PAN, phone, email, bank accounts
      ↓
[Translate]   — Hindi / Malayalam → English (auto-detect)
      ↓
[BART-base]   — optional local summarization (toggle in UI, RTX 3050 GPU)
      ↓
[FAISS RAG]   — IndexFlatIP cosine search → top-5 relevant legal chunks
      ↓
[Groq/LLaMA]  — legal analysis grounded in BART summary + RAG context
      ↓
[Chat Session] — interactive follow-up Q&A with document context
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Groq API key (free at console.groq.com)
export GROQ_API_KEY=your_key_here

# 3. Run
python app.py
# → Open http://localhost:5000
```

## Project Structure

```
lawbot_flask/
├── app.py                      # Flask backend — all routes
├── requirements.txt
├── templates/
│   └── index.html              # Frontend (your design, unchanged + chat + BART toggle)
├── backend/
│   ├── rag_engine.py           # FAISS IndexFlatIP — build & retrieve
│   ├── predictor.py            # IPC/BNS/contract matching via RAG
│   ├── summarizer.py           # BART-base — optional GPU summarization
│   ├── analyzer.py             # Groq/LLaMA — analysis + chat
│   ├── translator.py           # Hindi / Malayalam ↔ English
│   ├── privacy.py              # PII masking (Aadhaar, PAN, phone...)
│   └── pdf_reader.py           # PDF text extraction
└── data/
    ├── ipc_dataset.json        # 290 IPC sections
    ├── bns_dataset.json        # 39 BNS sections (with punishment + enhanced keywords)
    ├── mappings.json           # 165 auto-generated IPC→BNS mappings
    └── contract_dataset.json   # 10 contract types
```

## Data

| Dataset | Records | Source |
|---|---|---|
| IPC | 290 sections | IPC_Complete_Dataset.json |
| BNS | 39 sections | bns_contract_intelligence_dataset |
| IPC→BNS Mappings | 165 | Auto-generated via category + keyword matching |
| Contract types | 10 | Custom (land sale, rent, loan, POA, employment, partnership, NDA, RERA, will, MOU) |

## Features

- 📄 Text & PDF input
- 🔍 Auto document type detection (criminal vs contract)
- ⚖️ Top 3 IPC sections via FAISS semantic search
- 🔗 Safe IPC→BNS mapping (only confirmed mappings shown)
- 📑 Contract clause analysis
- ⚡ BART-base optional toggle (GPU — RTX 3050 recommended)
- 🔍 FAISS RAG context visible in UI (expandable)
- 💬 Interactive chat follow-up session
- 🌐 English · Hindi · Malayalam
- 🔒 PII masking before AI processing

## Future Work

- Case precedent search using 26,000 Supreme Court PDFs (1960-2009) from Kaggle
- Overnight preprocessing → FAISS index saved to disk → instant load
- Outcome prediction based on similar past cases

## Getting a Free Groq API Key

1. Go to https://console.groq.com
2. Sign up — no credit card needed
3. Create API key → paste in export command above
