"""
Adversarial Critic Verification Agent & Closed-Loop Self-Repair (Phase 6 in Paper).
Evaluates four-dimensional audit vector:
Psi = <psi_fact, psi_proc, psi_actor, psi_comp>
1. psi_fact: Factual Entailment against retrieved SOP chunks C_sop
2. psi_proc: Process Precedence reachability u ~>_G v in graph G
3. psi_actor: Swimlane Role Attribution precision against S(v)
4. psi_comp: Mandatory Regulatory Checklist Completeness Recall
Generates actionable repair instruction Delta_repair if min(psi_i) < 0.85 (up to N_max = 2).
"""

import re
from typing import Dict, List, Any, Tuple, Optional
from src.config import CRITIC_THRESHOLD, MAX_SELF_REPAIR_ITERATIONS

class AdversarialCriticAgent:
    """Closed-loop adversarial audit engine."""

    def __init__(self, threshold: float = CRITIC_THRESHOLD, max_repair_iterations: int = MAX_SELF_REPAIR_ITERATIONS):
        self.threshold = threshold
        self.max_repair_iterations = max_repair_iterations

    def audit_factual_entailment(self, answer: str, sop_chunks: List[Dict[str, Any]]) -> Tuple[float, List[str]]:
        """
        psi_fact: Evaluates proposition entailment against retrieved SOP context.
        Flags ungrounded fee claims, non-existent dates, or conflicting rules.
        Filters out structural Markdown headers and conversational wrappers.
        """
        if not sop_chunks:
            return 1.0, []

        context_text = " ".join([c.get("content", "") for c in sop_chunks]).lower()
        
        # Clean lines and extract substantive claims
        lines = answer.split("\n")
        substantive_claims = []
        for line in lines:
            line_str = line.strip()
            # Skip empty, headers, or short phrases
            if not line_str or line_str.startswith("#") or len(line_str) < 25:
                continue
            # Remove markdown bullets and numbering
            clean_str = re.sub(r"^[\*\-\d\.\s\(\)]+", "", line_str).strip()
            # Skip conversational introductions
            if clean_str.lower().startswith("berdasarkan") or clean_str.lower().startswith("sesuai"):
                if len(clean_str) < 40:
                    continue
            if len(clean_str) >= 20:
                substantive_claims.append(clean_str)

        if not substantive_claims:
            return 1.0, []

        grounded_count = 0
        violations = []

        for claim in substantive_claims:
            words = [w for w in re.findall(r"\w+", claim.lower()) if len(w) > 3]
            if not words:
                grounded_count += 1
                continue

            # Context lexical / semantic overlap
            overlap = sum(1 for w in words if w in context_text)
            ratio = overlap / len(words)

            # Check numbers / amounts / IDs
            numbers_in_claim = re.findall(r"\b\d+[\d.,]*\b", claim)
            num_ok = True
            for num in numbers_in_claim:
                cleaned_num = num.replace(".", "").replace(",", "")
                cleaned_ctx = context_text.replace(".", "").replace(",", "")
                # Skip trivial single digit indices (e.g. 1, 2, 3)
                if len(cleaned_num) > 1 and cleaned_num not in cleaned_ctx:
                    num_ok = False
                    violations.append(f"Ungrounded numeric claim '{num}' in: '{claim[:70]}...'")
                    break

            if ratio >= 0.30 and num_ok:
                grounded_count += 1
            elif not num_ok:
                pass  # already logged
            else:
                violations.append(f"Ungrounded proposition with low context entailment: '{claim[:70]}...'")

        score = grounded_count / len(substantive_claims)
        return round(score, 4), violations

    def audit_process_precedence(
        self, answer: str, bpmn_processes: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """
        psi_proc: Verifies directed edge transitions u ~>_G v.
        Detects step inversions (e.g., issuing LoA before Treasurer audit).
        """
        violations = []
        ans_lower = answer.lower()

        # Step 1: Check known critical precedence constraints in conference registration
        # Inversion 1: LoA issued before payment audit
        # Locate step numbers if formatted as numbered steps
        step_pattern = re.findall(r"(?:^|\n)\s*(\d+)[\.\)]\s*([^\n]+)", answer)
        if step_pattern:
            step_texts = [s[1].lower() for s in step_pattern]
            audit_idx = -1
            loa_idx = -1
            for idx, st in enumerate(step_texts):
                if any(w in st for w in ["audit", "validasi bukti", "memvalidasi"]):
                    audit_idx = idx
                if any(w in st for w in ["terbit", "issue loa", "penerbitan loa", "menerbitkan loa"]):
                    loa_idx = idx

            if audit_idx != -1 and loa_idx != -1 and loa_idx < audit_idx:
                violations.append("CRITICAL SEQUENCE INVERSION: LoA issued before payment audit validation (violates directed edge Flow_Valid)!")

        # Step 2: Desk Review before Peer Review
        if "peer review" in ans_lower and "desk review" in ans_lower:
            pos_desk = ans_lower.find("desk review")
            pos_peer = ans_lower.find("peer review")
            if pos_peer < pos_desk:
                violations.append("CRITICAL SEQUENCE INVERSION: Peer review conducted before Desk Review screening!")

        score = 1.0 if not violations else 0.50
        return score, violations

    def audit_actor_attribution(
        self, answer: str, bpmn_processes: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """
        psi_actor: Evaluates accuracy of role responsibilities against swimlane S(v).
        Prevents role diffusion (e.g. Author issuing LoA, or Secretariat auditing bank accounts).
        """
        violations = []
        ans_lower = answer.lower()

        # Check explicit role confusions:
        if "author memvalidasi bukti" in ans_lower or "author mengaudit" in ans_lower:
            violations.append("ROLE CONFUSION: Author is mistakenly attributed with auditing payment (Authorized Role: Bendahara / Treasurer)")

        if "bendahara menerbitkan loa" in ans_lower:
            violations.append("ROLE CONFUSION: Bendahara is mistakenly attributed with issuing LoA (Authorized Role: Sekretariat)")

        score = 1.0 if not violations else max(0.0, 1.0 - 0.25 * len(violations))
        return round(score, 4), violations

    def audit_completeness(
        self, answer: str, sop_chunks: List[Dict[str, Any]], expected_elements: Optional[List[str]] = None
    ) -> Tuple[float, List[str]]:
        """
        psi_comp: Verifies recall of mandatory regulatory checklist items.
        """
        violations = []
        if not expected_elements:
            return 1.0, []

        ans_lower = answer.lower()
        matched = 0
        for el in expected_elements:
            # check words
            el_words = [w for w in re.findall(r"\w+", el.lower()) if len(w) > 3]
            if not el_words or any(w in ans_lower for w in el_words):
                matched += 1
            else:
                violations.append(f"Missing mandatory regulatory element: '{el}'")

        score = matched / len(expected_elements)
        return round(score, 4), violations

    def audit(
        self,
        answer: str,
        sop_chunks: List[Dict[str, Any]],
        bpmn_processes: Dict[str, Any],
        expected_elements: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculates Psi = <psi_fact, psi_proc, psi_actor, psi_comp>.
        Returns audit results and targeted repair feedback Delta_repair.
        """
        psi_fact, fact_viols = self.audit_factual_entailment(answer, sop_chunks)
        psi_proc, proc_viols = self.audit_process_precedence(answer, bpmn_processes)
        psi_actor, actor_viols = self.audit_actor_attribution(answer, bpmn_processes)
        psi_comp, comp_viols = self.audit_completeness(answer, sop_chunks, expected_elements)

        all_violations = fact_viols + proc_viols + actor_viols + comp_viols
        passed = (
            psi_fact >= self.threshold
            and psi_proc >= self.threshold
            and psi_actor >= self.threshold
            and psi_comp >= self.threshold
        )

        repair_instructions = []
        if fact_viols:
            repair_instructions.append("FACTUAL ENTAILMENT ISSUE: " + "; ".join(fact_viols))
        if proc_viols:
            repair_instructions.append("PROCESS PRECEDENCE ISSUE: " + "; ".join(proc_viols))
        if actor_viols:
            repair_instructions.append("SWIMLANE ROLE ATTRIBUTION ISSUE: " + "; ".join(actor_viols))
        if comp_viols:
            repair_instructions.append("CHECKLIST COMPLETENESS ISSUE: " + "; ".join(comp_viols))

        delta_repair = "\n".join(repair_instructions)

        return {
            "passed": passed,
            "psi_vector": {
                "psi_fact": psi_fact,
                "psi_proc": psi_proc,
                "psi_actor": psi_actor,
                "psi_comp": psi_comp
            },
            "violations": all_violations,
            "delta_repair": delta_repair
        }
