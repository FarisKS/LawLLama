"""
LawBot India — Flask Backend
BART (optional) + FAISS RAG + Groq/LLaMA + Interactive Chat
"""
import re, os
from flask import Flask, render_template, request, jsonify, session
from backend.translator import translate_to_english, translate_text, detect_language
from backend.pdf_reader import extract_pdf_text
from backend.rag_engine import RAGEngine
from backend.predictor import LawPredictor
from backend.summarizer import summarize_if_enabled
from backend.analyzer import analyze_criminal_case, analyze_contract, chat_followup
from backend.privacy import mask_pii, get_privacy_report

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "lawbot-india-secret-2024")

# ── Load models at startup ────────────────────────────────────────────────────
print("[STARTUP] Loading RAG engine and predictor...")
rag       = RAGEngine()
predictor = LawPredictor(rag_engine=rag)
print("[STARTUP] Models ready.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    # ── Get inputs ──
    text_input  = request.form.get("text", "").strip()
    output_lang = request.form.get("lang", "English")
    mask_priv   = request.form.get("pii", "true") == "true"
    use_bart    = request.form.get("bart", "false") == "true"

    # Handle PDF upload
    if not text_input and "pdf" in request.files:
        pdf_file = request.files["pdf"]
        if pdf_file and pdf_file.filename.endswith(".pdf"):
            text_input = extract_pdf_text(pdf_file)

    if not text_input:
        return jsonify({"error": "No text provided."}), 400

    result = {}

    # ── Step 1: Privacy masking ──
    display_text = text_input
    if mask_priv:
        display_text, found_items = mask_pii(text_input)
        privacy_note = get_privacy_report(found_items)
        if privacy_note:
            result["privacy_note"] = privacy_note

    # ── Step 2: Language detection + translate ──
    src_code, src_name     = detect_language(text_input)
    english_text, _        = translate_to_english(display_text)
    result["src_lang"]     = src_name

    # ── Step 3: BART summarization  ──
    bart_summary           = summarize_if_enabled(english_text, use_bart)
    result["bart_used"]    = use_bart
    result["bart_summary"] = bart_summary

    # ── Step 4: FAISS RAG retrieval ──
    doc_type               = predictor.detect_document_type(english_text)
    result["doc_type"]     = doc_type
    rag_chunks             = rag.retrieve(english_text, top_k=5, doc_type=doc_type)
    rag_context            = rag.format_context(rag_chunks)
    result["rag_chunks"]   = [
        {"source": c["source"], "section": c["section"],
         "title": c["title"], "score": round(c["score"], 3)}
        for c in rag_chunks
    ]

    # ── Step 5: Criminal or Contract analysis ──
    if doc_type == "criminal":
        matched            = predictor.find_top_ipc_sections(english_text, top_k=3)
        result["sections"] = [
            {
                "ipc_section": s["ipc"]["section"],
                "ipc_title":   s["ipc"]["title"],
                "ipc_desc":    s["ipc"]["description"],
                "has_bns":     s["has_bns"],
                "bns_section": s["bns"]["section"]    if s["has_bns"] else None,
                "bns_title":   s["bns"]["title"]      if s["has_bns"] else None,
                "bns_punish":  s["bns"].get("punishment","—") if s["has_bns"] else None,
                "confidence":  round(s["confidence"] * 100),
            }
            for s in matched
        ]
        analysis_en = analyze_criminal_case(english_text, matched, bart_summary, rag_context)

    else:
        contract_match, conf = predictor.find_contract_type(english_text)
        result["contract"]   = {
            "category":   contract_match["category"],
            "act":        contract_match["act"],
            "section":    contract_match["section"],
            "confidence": round(conf * 100),
            "red_flags":  contract_match.get("red_flags", []),
        }
        analysis_en = analyze_contract(english_text, contract_match, bart_summary, rag_context)

    # ── Step 6: Translate output ──
    analyses = {}
    if output_lang in ("English", "All Three"):
        analyses["en"] = _fmt(analysis_en)
    if output_lang in ("Hindi (हिन्दी)", "All Three"):
        analyses["hi"] = _fmt(translate_text(analysis_en, "hi"))
    if output_lang in ("Malayalam (മലയാളം)", "All Three"):
        analyses["ml"] = _fmt(translate_text(analysis_en, "ml"))

    result["analyses"]     = analyses
    result["output_lang"]  = output_lang

    # ── Save session context for chat ──
    session["doc_text"]    = english_text[:3000]
    session["bart_summary"]= bart_summary
    session["rag_context"] = rag_context
    session["doc_type"]    = doc_type
    session["chat_history"]= [{"role": "assistant", "content": analysis_en}]
    session["output_lang"] = output_lang

    return jsonify(result)


@app.route("/chat", methods=["POST"])
def chat():
    """Interactive follow-up chat endpoint."""
    data    = request.get_json()
    user_msg = data.get("message", "").strip()

    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    # Retrieve session context
    doc_text     = session.get("doc_text", "")
    bart_summary = session.get("bart_summary", "")
    rag_context  = session.get("rag_context", "")
    doc_type     = session.get("doc_type", "criminal")
    chat_history = session.get("chat_history", [])
    output_lang  = session.get("output_lang", "English")

    if not doc_text:
        return jsonify({"error": "No document analyzed yet. Please analyze a document first."}), 400

    # Get reply from Groq
    reply_en = chat_followup(
        user_message  = user_msg,
        chat_history  = chat_history,
        document_text = doc_text,
        bart_summary  = bart_summary,
        rag_context   = rag_context,
        doc_type      = doc_type
    )

    # Translate if needed
    lang_map = {
        "Hindi (हिन्दी)":     "hi",
        "Malayalam (മലയാളം)": "ml",
    }
    lang_code = lang_map.get(output_lang)
    reply_display = _fmt(translate_text(reply_en, lang_code)) if lang_code else _fmt(reply_en)

    # Update history
    chat_history.append({"role": "user",      "content": user_msg})
    chat_history.append({"role": "assistant",  "content": reply_en})
    session["chat_history"] = chat_history[-20:]  # keep last 20 turns

    return jsonify({"reply": reply_display})


@app.route("/chat/clear", methods=["POST"])
def clear_chat():
    session.pop("chat_history", None)
    return jsonify({"ok": True})


def _fmt(text: str) -> str:
    """Convert **bold** markdown → <b> tags, newlines → <br>."""
    return re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text).replace('\n', '<br>')


if __name__ == "__main__":
    app.run(debug=True, port=5000)
