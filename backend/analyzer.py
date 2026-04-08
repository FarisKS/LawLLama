"""
Analyzer — BART summary → FAISS RAG context → Groq/LLaMA legal analysis + chat
"""
from groq import Groq
import os

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
GROQ_MODEL  = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are LawBot, an expert Indian legal assistant helping common citizens understand
legal documents and cases in plain, simple language. You are grounded in:
- Indian Penal Code (IPC) and Bharatiya Nyaya Sanhita (BNS 2023)
- Indian Contract Act, Transfer of Property Act, RERA
- Common legal documents: sale deeds, rent agreements, loan agreements, employment contracts

Rules:
- Always explain in simple language — no jargon
- Be empathetic and practical  
- Always recommend consulting a qualified advocate for serious matters
- Use ONLY the provided RAG context and document as your knowledge base
- If something is not in the context, say so honestly"""


def _chat(messages: list, max_tokens: int = 1200) -> str:
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL, max_tokens=max_tokens, messages=messages
    )
    return response.choices[0].message.content


def analyze_criminal_case(text: str, matched_sections: list,
                           bart_summary: str, rag_context: str) -> str:
    sections_info = ""
    for i, s in enumerate(matched_sections, 1):
        ipc = s["ipc"]
        sections_info += f"\n{i}. IPC §{ipc['section']} — {ipc['title']}: {ipc['description']}"
        if s["has_bns"]:
            bns = s["bns"]
            sections_info += f"\n   ↳ BNS §{bns['section']} — {bns['title']} | Punishment: {bns.get('punishment','—')}"
        else:
            sections_info += "\n   ↳ No direct BNS equivalent (removed/restructured in BNS 2023)"

    bart_block = f"\n=== BART LOCAL SUMMARY ===\n{bart_summary}\n" if bart_summary else ""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""Analyze this legal case using the provided context.
{bart_block}
=== RAG RETRIEVED LEGAL CONTEXT ===
{rag_context}

=== MATCHED IPC/BNS SECTIONS ===
{sections_info}

=== ORIGINAL CASE TEXT ===
{text[:2000]}

Provide a structured response with exactly these sections:

**PLAIN LANGUAGE SUMMARY**
(3-4 simple sentences — what happened, who is involved)

**LEGAL SECTIONS INVOLVED**
(For each matched section — what it means for this specific case in plain terms)

**PUNISHMENT DETAILS**
(Maximum punishment under BNS/IPC for each applicable section)

**IMMEDIATE STEPS TO TAKE**
(5 numbered concrete actions the victim/complainant should take RIGHT NOW)

**YOUR RIGHTS**
(4 numbered key legal rights in this situation)

**IMPORTANT WARNING**
(One critical caution the person must know)

Keep language simple. Be direct and empathetic."""}
    ]
    return _chat(messages)


def analyze_contract(text: str, contract_info: dict,
                     bart_summary: str, rag_context: str) -> str:
    category  = contract_info.get("category", "Legal Agreement") if contract_info else "Legal Agreement"
    act       = contract_info.get("act", "")                      if contract_info else ""
    section   = contract_info.get("section", "")                  if contract_info else ""
    red_flags = contract_info.get("red_flags", [])                if contract_info else []

    bart_block = f"\n=== BART LOCAL SUMMARY ===\n{bart_summary}\n" if bart_summary else ""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""Analyze this legal contract using the provided context.
{bart_block}
=== RAG RETRIEVED LEGAL CONTEXT ===
{rag_context}

=== DOCUMENT TYPE ===
{category} under {act}, Section {section}
Known red flags: {', '.join(red_flags)}

=== ORIGINAL DOCUMENT TEXT ===
{text[:2000]}

Provide a structured response with exactly these sections:

**WHAT IS THIS DOCUMENT?**
(2-3 simple sentences explaining what this contract is)

**KEY CLAUSES EXPLAINED**
(List and explain each important clause in plain language)

**YOUR OBLIGATIONS**
(Numbered list — what YOU must do under this contract)

**OTHER PARTY'S OBLIGATIONS**
(Numbered list — what the OTHER PARTY must do)

**RED FLAGS / CONCERNS**
(Numbered list — risky, unfair, or missing clauses found)

**BEFORE YOU SIGN — CHECKLIST**
(5 numbered things to verify before signing)

**LEGAL PROTECTIONS YOU HAVE**
(Laws that protect you in this agreement)

Keep language very simple. Be direct and practical."""}
    ]
    return _chat(messages)


def chat_followup(user_message: str, chat_history: list,
                  document_text: str, bart_summary: str,
                  rag_context: str, doc_type: str) -> str:
    """Handle follow-up questions in the interactive chat session."""
    bart_block = f"BART Summary: {bart_summary}\n" if bart_summary else ""
    system_ctx = f"""{SYSTEM_PROMPT}

=== DOCUMENT CONTEXT ===
{bart_block}RAG Legal Context:
{rag_context}

Original Document (excerpt):
{document_text[:1500]}

Document Type: {doc_type}
===
The user has received an initial analysis and is asking follow-up questions.
Answer based only on the document and context above. Be concise and helpful."""

    messages = [{"role": "system", "content": system_ctx}]
    messages += chat_history[-10:]  # keep last 10 turns for context window
    messages.append({"role": "user", "content": user_message})
    return _chat(messages, max_tokens=600)
