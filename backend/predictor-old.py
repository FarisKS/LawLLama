import json, os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


class LawPredictor:
    def __init__(self, rag_engine):
        with open(os.path.join(DATA_DIR, "ipc_dataset.json"))      as f: self.ipc       = json.load(f)
        with open(os.path.join(DATA_DIR, "bns_dataset.json"))      as f: self.bns       = json.load(f)
        with open(os.path.join(DATA_DIR, "mappings.json"))         as f: self.mapping   = json.load(f)
        with open(os.path.join(DATA_DIR, "contract_dataset.json")) as f: self.contracts = json.load(f)

        self.ipc_to_bns = {m["ipc_section"]: m["bns_section"] for m in self.mapping}
        self.bns_lookup  = {b["section"]: b for b in self.bns}
        self.ipc_lookup  = {s["section"]: s for s in self.ipc}
        self.rag         = rag_engine
        self.embedder    = rag_engine.embedder  # reuse — don't load twice

    def detect_document_type(self, text: str) -> str:
        criminal_kw = [
            "murder","theft","rape","assault","robbery","fraud","FIR","police",
            "accused","complaint","arrested","victim","harassed","beaten","abused",
            "attacked","stabbed","killed","dowry","extortion","kidnap","dacoity",
            "हत्या","चोरी","बलात्कार","मारपीट","शिकायत",
            "കൊലപാതകം","മോഷണം","പരാതി","ആക്രമണം"
        ]
        contract_kw = [
            "agreement","contract","party","clause","hereby","whereas","sale deed",
            "lease","rent","loan","MOU","partnership","seller","buyer","landlord",
            "tenant","lender","borrower","mortgage","deed","execute","notarize",
            "अनुबंध","समझौता","पट्टा","किराया",
            "കരാർ","വാടക","വിൽപ്പന"
        ]
        t  = text.lower()
        cs = sum(1 for k in criminal_kw  if k.lower() in t)
        co = sum(1 for k in contract_kw  if k.lower() in t)
        return "contract" if co > cs else "criminal"

    def find_top_ipc_sections(self, text: str, top_k: int = 3) -> list:
        chunks = self.rag.retrieve(text, top_k=top_k * 4, doc_type="criminal")
        results, seen_ipc, seen_bns = [], set(), set()
        for chunk in chunks:
            if chunk["source"] != "IPC":
                continue
            sec = chunk["section"]
            if sec in seen_ipc:
                continue
            seen_ipc.add(sec)
            ipc_entry = self.ipc_lookup.get(sec)
            if not ipc_entry:
                continue
            bns_entry = None
            if sec in self.ipc_to_bns:
                bns_sec = self.ipc_to_bns[sec]
                if bns_sec in self.bns_lookup and bns_sec not in seen_bns:
                    bns_entry = self.bns_lookup[bns_sec]
                    seen_bns.add(bns_sec)
            results.append({
                "ipc": ipc_entry, "bns": bns_entry,
                "has_bns": bns_entry is not None,
                "confidence": round(chunk["score"], 3)
            })
            if len(results) >= top_k:
                break
        return results

    def find_contract_type(self, text: str) -> tuple:
        chunks = self.rag.retrieve(text, top_k=3, doc_type="contract")
        if chunks:
            title    = chunks[0]["title"]
            contract = next((c for c in self.contracts if c["category"] == title), None)
            if contract:
                return contract, chunks[0]["score"]
        return self.contracts[0], 0.5
