import requests
import json
import time

BASE_URL = "http://localhost:5000"

# ── 25 Criminal Case Test Queries ─────────────────────────────────────────
criminal_queries = [
    "A person was stabbed with a knife during a robbery",
    "Someone broke into my house at night and stole jewellery",
    "My employer has not paid my salary for 3 months",
    "A man threatened to kill me if I report to police",
    "A group of people attacked me on the street",
    "My neighbour damaged my car intentionally",
    "A person forged my signature on a legal document",
    "Someone is blackmailing me with my private photos",
    "A drunk driver hit my vehicle and injured me",
    "My husband physically abused me and took my money",
    "A shopkeeper sold me expired medicine",
    "Someone hacked my bank account and stole money",
    "A person kidnapped a child for ransom",
    "My business partner cheated me out of shared funds",
    "A government official demanded bribe for approving my file",
    "Someone spread false rumours about me online",
    "A person tried to rape a woman in a public place",
    "My landlord forcefully evicted me without notice",
    "A man was found with illegal weapons",
    "Someone set fire to my property",
    "A person was murdered during a land dispute",
    "My employee stole cash from the office",
    "A man was found selling drugs near a school",
    "Someone impersonated a police officer to extort money",
    "A person was arrested for sedition after a social media post",
]

# ── 25 Contract Test Queries ───────────────────────────────────────────────
contract_queries = [
    "Review this rental agreement for a 2BHK flat in Kochi",
    "Check if this employment contract has a valid termination clause",
    "Analyse this NDA for confidentiality obligations",
    "Is this freelance service agreement legally binding?",
    "What are the penalty clauses in this vendor contract?",
    "Review the intellectual property clause in this software contract",
    "Does this loan agreement mention prepayment penalties?",
    "What are the liabilities mentioned in this construction contract?",
    "Check the notice period clause in this employment agreement",
    "Is the arbitration clause in this partnership deed valid?",
    "Review the payment terms in this consulting agreement",
    "Does this lease deed mention maintenance responsibilities?",
    "Analyse the indemnity clause in this service contract",
    "What are the termination conditions in this SaaS agreement?",
    "Review the governing law clause in this contract",
    "Is the non-compete clause in this agreement enforceable?",
    "What are the delivery obligations in this supply agreement?",
    "Check the force majeure clause in this business contract",
    "Review the warranty terms in this product purchase agreement",
    "Does this shareholders agreement have a buyout clause?",
    "What are the dispute resolution terms in this MOU?",
    "Check the confidentiality period in this NDA",
    "Review the liability cap in this IT services contract",
    "Does this franchise agreement mention territory restrictions?",
    "Analyse the renewal clause in this annual maintenance contract",
]

# ── PII Test Inputs ────────────────────────────────────────────────────────
pii_inputs = [
    ("My Aadhaar is 1234 5678 9012 and phone is 9876543210", ["1234 5678 9012", "9876543210"]),
    ("PAN card ABCDE1234F was used for fraud", ["ABCDE1234F"]),
    ("Email me at testuser@gmail.com about my case", ["testuser@gmail.com"]),
    ("Bank account 9876543210123456 was hacked", ["9876543210123456"]),
    ("Contact 9123456789 or email user@yahoo.com", ["9123456789", "user@yahoo.com"]),
]

# ── Hindi/Malayalam Test Inputs ────────────────────────────────────────────
multilingual_inputs = [
    ("मेरे पड़ोसी ने मुझे धमकी दी", "hi"),   # Hindi
    ("എന്റെ ഭർത്താവ് എന്നെ മർദ്ദിച്ചു", "ml"),  # Malayalam
]

results = {
    "criminal": [], "contract": [],
    "pii": [], "multilingual": [],
    "response_times_bart_on": [],
    "response_times_bart_off": [],
}

# ═══════════════════════════════════════════════════════════════════════════
# TEST 1 — Criminal Queries (BART off for speed)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[1/4] Testing criminal queries...")
for i, query in enumerate(criminal_queries):
    try:
        start = time.time()
        r = requests.post(f"{BASE_URL}/analyze", data={
            "input_text": query, "language": "en",
            "use_pii": "on", "use_bart": ""
        }, timeout=30)
        elapsed = time.time() - start
        data = r.json()

        got_sections = bool(data.get("sections"))
        got_analysis = bool(data.get("analysis"))
        results["criminal"].append({
            "query": query, "ok": got_sections and got_analysis,
            "sections_count": len(data.get("sections", [])),
            "time": round(elapsed, 2)
        })
        results["response_times_bart_off"].append(elapsed)
        print(f"  [{i+1}/25] {'✓' if got_sections and got_analysis else '✗'}  {round(elapsed,2)}s")
    except Exception as e:
        print(f"  [{i+1}/25] ERROR: {e}")
        results["criminal"].append({"query": query, "ok": False, "time": 0})

# ═══════════════════════════════════════════════════════════════════════════
# TEST 2 — Contract Queries (BART off)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[2/4] Testing contract queries...")
for i, query in enumerate(contract_queries):
    try:
        start = time.time()
        r = requests.post(f"{BASE_URL}/analyze", data={
            "input_text": query, "language": "en",
            "use_pii": "on", "use_bart": ""
        }, timeout=30)
        elapsed = time.time() - start
        data = r.json()

        got_analysis = bool(data.get("analysis"))
        results["contract"].append({
            "query": query, "ok": got_analysis,
            "time": round(elapsed, 2)
        })
        results["response_times_bart_off"].append(elapsed)
        print(f"  [{i+1}/25] {'✓' if got_analysis else '✗'}  {round(elapsed,2)}s")
    except Exception as e:
        print(f"  [{i+1}/25] ERROR: {e}")
        results["contract"].append({"query": query, "ok": False, "time": 0})

# ═══════════════════════════════════════════════════════════════════════════
# TEST 3 — PII Detection
# ═══════════════════════════════════════════════════════════════════════════
print("\n[3/4] Testing PII masking...")
pii_correct = 0
for text, expected_pii in pii_inputs:
    try:
        r = requests.post(f"{BASE_URL}/analyze", data={
            "input_text": text, "language": "en",
            "use_pii": "on", "use_bart": ""
        }, timeout=30)
        data = r.json()
        masked_text = data.get("masked_text", "") or data.get("analysis", "")

        # Check if each expected PII token is masked/removed
        detected = all(p not in masked_text for p in expected_pii)
        pii_correct += int(detected)
        results["pii"].append({"input": text, "detected": detected})
        print(f"  {'✓' if detected else '✗'}  {text[:50]}...")
    except Exception as e:
        print(f"  ERROR: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# TEST 4 — Multilingual
# ═══════════════════════════════════════════════════════════════════════════
print("\n[4/4] Testing multilingual inputs...")
ml_correct = 0
for text, lang in multilingual_inputs:
    try:
        r = requests.post(f"{BASE_URL}/analyze", data={
            "input_text": text, "language": lang,
            "use_pii": "on", "use_bart": ""
        }, timeout=30)
        data = r.json()
        ok = bool(data.get("analysis")) and "error" not in data.get("analysis", "").lower()
        ml_correct += int(ok)
        results["multilingual"].append({"text": text, "lang": lang, "ok": ok})
        print(f"  {'✓' if ok else '✗'}  [{lang}] {text}")
    except Exception as e:
        print(f"  ERROR: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# METRICS SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
criminal_ok  = sum(1 for r in results["criminal"]  if r["ok"])
contract_ok  = sum(1 for r in results["contract"]  if r["ok"])
avg_time_off = sum(results["response_times_bart_off"]) / len(results["response_times_bart_off"])

print("\n" + "="*55)
print("  LAWLAMMA EVALUATION METRICS")
print("="*55)
print(f"  Criminal Query Success Rate : {criminal_ok}/25  = {criminal_ok/25*100:.1f}%")
print(f"  Contract Query Success Rate : {contract_ok}/25  = {contract_ok/25*100:.1f}%")
print(f"  PII Detection Accuracy      : {pii_correct}/{len(pii_inputs)}   = {pii_correct/len(pii_inputs)*100:.1f}%")
print(f"  Multilingual Support        : {ml_correct}/{len(multilingual_inputs)}    = {ml_correct/len(multilingual_inputs)*100:.1f}%")
print(f"  Avg Response Time (w/o BART): {avg_time_off:.2f}s")
print("="*55)

# Save to file
with open("test_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("\n  Full results saved to test_results.json")