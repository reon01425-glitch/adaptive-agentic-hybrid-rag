"""
Query Understanding Agent (QUA) - Algorithm 1 in Paper.
Performs:
1. Institutional acronym expansion via D_acronym.
2. Canonical process label grounding against L(V).
3. Synthesizes normalized query q' with domain tag [domain=ICICoS].
"""

import re
from typing import Dict, List, Set, Tuple, Any

ACRONYM_DICTIONARY: Dict[str, str] = {
    "loa": "Letter of Acceptance (LoA)",
    "spj": "Surat Pertanggungjawaban (SPJ / Payment Voucher)",
    "va": "Bank Virtual Account (VA)",
    "kta": "Kartu Tanda Anggota (KTA IEEE Member Card)",
    "ktm": "Kartu Tanda Mahasiswa (KTM Student ID)",
    "tpc": "Technical Program Committee (TPC)",
    "ecf": "electronic IEEE Copyright Form (eCF)",
    "pdfexpress": "IEEE PDF eXpress",
    "pdf express": "IEEE PDF eXpress",
    "sk": "Surat Keputusan (Decree)",
    "npwp": "Nomor Pokok Wajib Pajak (NPWP / Tax ID)",
    "nik": "Nomor Induk Kependudukan (NIK)",
    "edas": "EDAS Conference Management System",
    "icicos": "ICICoS Conference"
}

class QueryUnderstandingAgent:
    """Algorithm 1: Query Understanding & Canonical Normalization"""

    def __init__(self, canonical_labels: List[str]):
        self.canonical_labels = canonical_labels
        self.grounding_threshold = 0.40

    def expand_acronyms(self, query: str) -> str:
        """Expands enterprise and conference acronyms."""
        tokens = re.findall(r"\w+|[^\w\s]", query)
        expanded_tokens = []
        for tok in tokens:
            low = tok.lower()
            if low in ACRONYM_DICTIONARY:
                expanded_tokens.append(f"{tok} ({ACRONYM_DICTIONARY[low]})")
            else:
                expanded_tokens.append(tok)
        
        # Detokenize nicely
        text = " ".join(expanded_tokens)
        text = re.sub(r"\s+([,.\?!;:])", r"\1", text)
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)
        return text

    def ground_canonical_labels(self, query_expanded: str) -> List[str]:
        """Matches canonical BPMN activity labels L(v) based on token overlap."""
        q_words = set(re.findall(r"\w+", query_expanded.lower()))
        matched_labels = []

        for label in self.canonical_labels:
            l_words = set(re.findall(r"\w+", label.lower()))
            overlap = q_words.intersection(l_words)
            if not l_words:
                continue
            sim = len(overlap) / len(l_words)
            if sim >= self.grounding_threshold or len(overlap) >= 2:
                matched_labels.append(label)

        return matched_labels[:3]

    def normalize_query(self, query: str) -> Dict[str, Any]:
        """
        Executes Algorithm 1:
        Returns:
            q_norm: str
            matched_labels: List[str]
            domain: str
        """
        q_exp = self.expand_acronyms(query)
        matched_labels = self.ground_canonical_labels(q_exp)
        
        canonical_injection = ""
        if matched_labels:
            canonical_injection = f" [Canonical Activities: {'; '.join(matched_labels)}]"

        q_norm = f"[domain=ICICoS] {q_exp}{canonical_injection}"
        
        return {
            "original_query": query,
            "expanded_query": q_exp,
            "matched_labels": matched_labels,
            "normalized_query": q_norm,
            "domain": "ICICoS"
        }
