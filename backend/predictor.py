import json, os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


class LawPredictor:
    def __init__(self, rag_engine):
        with open(os.path.join(DATA_DIR, "ipc_dataset.json"))      as f: _ipc = json.load(f)
        with open(os.path.join(DATA_DIR, "bns_dataset.json"))      as f: _bns = json.load(f)
        with open(os.path.join(DATA_DIR, "mappings.json"))         as f: _map = json.load(f)
        with open(os.path.join(DATA_DIR, "contract_dataset.json")) as f: _con = json.load(f)

        # Unwrap wrapper dicts — same structure as rag_engine.py
        #   ipc/bns → {"metadata":…, "sections":[…]}
        #   mappings → {"metadata":…, "mappings":[…]}
        #   contracts → already a plain list
        ipc_list       = _ipc["sections"] if isinstance(_ipc, dict) else _ipc
        bns_list       = _bns["sections"] if isinstance(_bns, dict) else _bns
        mappings_list  = _map["mappings"] if isinstance(_map, dict) else _map
        self.contracts = _con["sections"] if isinstance(_con, dict) else _con

        # ipc/bns use "section_number" and "section_title" — not "section"/"title"
        self.ipc_to_bns = {m["ipc_section"]: m["bns_section"] for m in mappings_list}
        self.bns_lookup  = {b["section_number"]: b for b in bns_list}
        self.ipc_lookup  = {s["section_number"]: s for s in ipc_list}

        self.rag      = rag_engine
        self.embedder = rag_engine.embedder   # reuse — don't load twice

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
        cs = sum(1 for k in criminal_kw if k.lower() in t)
        co = sum(1 for k in contract_kw if k.lower() in t)
        return "contract" if co > cs else "criminal"

    def find_top_ipc_sections(self, text: str, top_k: int = 3) -> list:
        chunks = self.rag.retrieve(text, top_k=top_k * 4, doc_type="criminal")
        results, seen_ipc, seen_bns = [], set(), set()

        for chunk in chunks:
            if chunk["source"] != "IPC":
                continue
            sec = chunk["section"]          # rag_engine stores normalised "section" key
            if sec in seen_ipc:
                continue
            seen_ipc.add(sec)

            ipc_entry = self.ipc_lookup.get(sec)
            if not ipc_entry:
                continue

            # Build a normalised ipc dict for the analyzer (uses "section", "title", "description")
            ipc_norm = {
                "section":     ipc_entry["section_number"],
                "title":       ipc_entry["section_title"],
                "description": ipc_entry.get("description", ""),
            }

            bns_norm = None
            if sec in self.ipc_to_bns:
                bns_sec   = self.ipc_to_bns[sec]
                bns_entry = self.bns_lookup.get(bns_sec)
                if bns_entry and bns_sec not in seen_bns:
                    seen_bns.add(bns_sec)
                    bns_norm = {
                        "section":     bns_entry["section_number"],
                        "title":       bns_entry["section_title"],
                        "description": bns_entry.get("description", ""),
                        "punishment":  bns_entry.get("punishment", "—"),
                    }

            results.append({
                "ipc":        ipc_norm,
                "bns":        bns_norm,
                "has_bns":    bns_norm is not None,
                "confidence": round(chunk["score"], 3),
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
