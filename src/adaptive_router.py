"""
Adaptive Router Agent and Dynamic Quota Engine.
Evaluates query feature vector phi(q') to compute Softmax intent distribution:
P(r | q') over T = {SOP, BPMN, Hybrid}
Allocates dynamic retrieval quotas k(r) = <k_sop, k_bpmn>:
• SOP: <6, 2>
• BPMN: <3, 4>
• Hybrid: <5, 3>
"""

import re
import numpy as np
from typing import Dict, Any, Tuple

from src.config import QUOTAS

class AdaptiveRouterAgent:
    """Softmax Intent Router with Dynamic Quota Allocation."""

    def __init__(self):
        # Feature keyword patterns
        self.sop_keywords = {
            "biaya", "tarif", "fee", "rate", "harga", "dokumen", "lampiran", "syarat",
            "checklist", "form", "formulir", "attachment", "ktm", "kta", "npwp", "nik",
            "pajak", "pph", "halaman", "page", "limit", "idr", "usd", "rupiah", "persyaratan"
        }
        
        self.bpmn_keywords = {
            "urutan", "langkah", "tahapan", "tahap", "alur", "proses", "setelah", "sebelum",
            "siapa", "aktor", "role", "swimlane", "bertugas", "wewenang", "sequence", "order",
            "chronological", "next", "prior", "handoff", "diteruskan", "milestone", "kegiatan"
        }
        
        self.gateway_keywords = {
            "jika", "apakah", "bila", "ditolak", "disetujui", "revisi", "diskrepansi", "kurang bayar",
            "salah", "tidak valid", "invalid", "reject", "rejection", "grace period", "skor",
            "threshold", "ambang", "cabang", "branch", "terpenuhi", "gagal", "remediasi"
        }

    def extract_features(self, query: str) -> np.ndarray:
        """
        Extracts 6-dimensional feature vector phi(q'):
        [f_sop, f_bpmn, f_gateway, len_norm, has_dual_intent, has_conditional]
        """
        tokens = set(re.findall(r"\w+", query.lower()))
        
        sop_count = sum(1 for w in tokens if w in self.sop_keywords)
        bpmn_count = sum(1 for w in tokens if w in self.bpmn_keywords)
        gw_count = sum(1 for w in tokens if w in self.gateway_keywords)

        has_dual = 1.0 if (sop_count > 0 and (bpmn_count > 0 or gw_count > 0)) else 0.0
        has_conditional = 1.0 if gw_count > 0 else 0.0
        len_norm = min(len(tokens) / 20.0, 1.0)

        return np.array([sop_count, bpmn_count, gw_count, has_dual, has_conditional, len_norm], dtype=np.float32)

    def route(self, query: str) -> Dict[str, Any]:
        """
        Computes Softmax intent distribution and dynamic quotas.
        Returns:
            intent: "SOP" | "BPMN" | "Hybrid"
            probabilities: Dict[str, float]
            quotas: Dict[str, int]
        """
        phi = self.extract_features(query)
        sop_c, bpmn_c, gw_c, has_dual, has_cond, _ = phi

        # Calibrated logit weights
        # Classes: 0: SOP, 1: BPMN, 2: Hybrid
        # Logit formulas:
        w_sop = 2.2 * sop_c - 1.2 * bpmn_c - 0.8 * gw_c - 2.0 * has_dual + 0.5
        w_bpmn = -1.2 * sop_c + 2.0 * bpmn_c + 1.8 * gw_c - 1.5 * has_dual + 0.3
        w_hyb = 1.5 * sop_c + 1.4 * (bpmn_c + gw_c) + 4.0 * has_dual - 0.5

        logits = np.array([w_sop, w_bpmn, w_hyb], dtype=np.float32)
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        labels = ["SOP", "BPMN", "Hybrid"]
        best_idx = int(np.argmax(probs))
        predicted_intent = labels[best_idx]

        quota = QUOTAS[predicted_intent]

        return {
            "intent": predicted_intent,
            "probabilities": {
                "SOP": round(float(probs[0]), 4),
                "BPMN": round(float(probs[1]), 4),
                "Hybrid": round(float(probs[2]), 4)
            },
            "k_sop": quota["k_sop"],
            "k_bpmn": quota["k_bpmn"],
            "feature_vector": phi.tolist()
        }
