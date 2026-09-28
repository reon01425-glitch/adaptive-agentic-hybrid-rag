"""
Adaptive Agentic Hybrid RAG with BPMN-Grounded Graph Reasoning.
Unified End-to-End Pipeline Coordinating all 6 Phases:
- Phase 1: Query Understanding Agent (QUA)
- Phase 2: Adaptive Intent Router Agent
- Phase 3: Dual Retrieval Execution (SOP Vector + BPMN DiGraph)
- Phase 4: Cross-Modal Context Fusion Layer
- Phase 5: Grounded LLM Generator (Draft)
- Phase 6: Adversarial Critic Verification & Closed-Loop Self-Repair
"""

import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from src.config import BPMN_DIR, SOP_DIR, CRITIC_THRESHOLD, MAX_SELF_REPAIR_ITERATIONS
from src.bpmn_parser import MultiBPMNRepository
from src.sop_indexer import SOPIndexer
from src.query_understanding import QueryUnderstandingAgent
from src.adaptive_router import AdaptiveRouterAgent
from src.context_fusion import CrossModalContextFusion
from src.generator import GroundedGenerator
from src.critic_verification import AdversarialCriticAgent

class AdaptiveAgenticHybridRAGPipeline:
    """Full End-to-End Proposed Architecture."""

    def __init__(
        self,
        bpmn_dir: Path = BPMN_DIR,
        sop_dir: Path = SOP_DIR,
        critic_threshold: float = CRITIC_THRESHOLD,
        max_repair_iterations: int = MAX_SELF_REPAIR_ITERATIONS
    ):
        print("🚀 Initializing Adaptive Agentic Hybrid RAG Pipeline...")
        # 1. Ingestion & Dual Representation (Layer 1)
        self.bpmn_repo = MultiBPMNRepository(bpmn_dir)
        self.sop_indexer = SOPIndexer(sop_dir)

        # Collect canonical BPMN labels
        all_nodes = self.bpmn_repo.get_all_nodes()
        canonical_labels = [n.get("label", "") for n in all_nodes if n.get("label")]

        # 2. Multi-Agent Orchestration & Retrieval (Layer 2)
        self.qua = QueryUnderstandingAgent(canonical_labels)
        self.router = AdaptiveRouterAgent()

        # 3. Context Fusion, Generation & Verification (Layer 3)
        self.fusion = CrossModalContextFusion(target_domain="ICICoS")
        self.generator = GroundedGenerator()
        self.critic = AdversarialCriticAgent(
            threshold=critic_threshold, max_repair_iterations=max_repair_iterations
        )
        print("✅ Pipeline Initialized Successfully.")

    def run(self, query: str, expected_elements: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Executes complete multi-agent operational sequence across all 6 phases.
        """
        total_start = time.time()
        phase_latencies = {}

        # --- PHASE 1: Query Understanding ---
        t0 = time.time()
        qua_result = self.qua.normalize_query(query)
        q_norm = qua_result["normalized_query"]
        phase_latencies["phase1_qua"] = round(time.time() - t0, 3)

        # --- PHASE 2: Adaptive Intent Routing ---
        t0 = time.time()
        route_result = self.router.route(q_norm)
        k_sop = route_result["k_sop"]
        k_bpmn = route_result["k_bpmn"]
        phase_latencies["phase2_router"] = round(time.time() - t0, 3)

        # --- PHASE 3: Dual Retrieval Execution ---
        t0 = time.time()
        # 3a. Retrieve SOP text chunks
        sop_chunks = self.sop_indexer.retrieve(query=q_norm, top_k=k_sop)
        
        # 3b. Retrieve BPMN graph sub-topology
        matched_nodes = []
        for n in self.bpmn_repo.get_all_nodes():
            n_label = n.get("label", "").lower()
            if any(w in query.lower() for w in n_label.split() if len(w) > 3):
                matched_nodes.append(n["id"])

        if not matched_nodes and route_result["intent"] == "SOP":
            bpmn_graph_text = ""
        else:
            if not matched_nodes:
                q_l = query.lower()
                if "review" in q_l or "turnitin" in q_l or "similarity" in q_l or "rubric" in q_l:
                    matched_nodes = ["Task_SubmitManuscript", "Task_DeskReview", "Gateway_ReviewDecision"]
                elif "camera" in q_l or "xpress" in q_l or "copyright" in q_l or "publikasi" in q_l:
                    matched_nodes = ["Task_SubmitCameraReady", "Gateway_FormatAudit", "Task_IEEETransfer"]
                else:
                    matched_nodes = ["Task_SubmitRequest", "Task_AuditPayment", "Gateway_PaymentCheck"]
            bpmn_graph_text = self.bpmn_repo.query_subgraph(matched_nodes[:k_bpmn], hops=2)

        phase_latencies["phase3_retrieval"] = round(time.time() - t0, 3)

        # --- PHASE 4: Cross-Modal Context Fusion ---
        t0 = time.time()
        all_bpmn_nodes = self.bpmn_repo.get_all_nodes()
        fused_context = self.fusion.fuse(
            sop_chunks=sop_chunks,
            bpmn_graph_text=bpmn_graph_text,
            bpmn_nodes=all_bpmn_nodes
        )
        phase_latencies["phase4_fusion"] = round(time.time() - t0, 3)

        # --- PHASE 5: Grounded LLM Generation (Draft) ---
        t0 = time.time()
        gen_res = self.generator.generate_draft(query=q_norm, context=fused_context)
        current_answer = gen_res["answer"]
        phase_latencies["phase5_draft_gen"] = round(time.time() - t0, 3)

        # --- PHASE 6: Adversarial Critic Verification & Closed-Loop Self-Repair ---
        t0 = time.time()
        repair_iterations = 0
        repair_history = []
        
        audit_res = self.critic.audit(
            answer=current_answer,
            sop_chunks=sop_chunks,
            bpmn_processes=self.bpmn_repo.processes,
            expected_elements=expected_elements
        )
        repair_history.append({
            "iteration": 0,
            "passed": audit_res["passed"],
            "psi_vector": audit_res["psi_vector"],
            "violations": audit_res["violations"]
        })

        # Closed-loop self-repair loop
        while not audit_res["passed"] and repair_iterations < self.critic.max_repair_iterations:
            repair_iterations += 1
            delta_repair = audit_res["delta_repair"]
            
            repaired_res = self.generator.generate_repaired(
                draft_answer=current_answer,
                repair_feedback=delta_repair,
                context=fused_context,
                query=q_norm
            )
            current_answer = repaired_res["answer"]

            # Re-audit repaired answer
            audit_res = self.critic.audit(
                answer=current_answer,
                sop_chunks=sop_chunks,
                bpmn_processes=self.bpmn_repo.processes,
                expected_elements=expected_elements
            )
            repair_history.append({
                "iteration": repair_iterations,
                "passed": audit_res["passed"],
                "psi_vector": audit_res["psi_vector"],
                "violations": audit_res["violations"]
            })

        phase_latencies["phase6_critic_audit"] = round(time.time() - t0, 3)
        total_latency = round(time.time() - total_start, 3)

        return {
            "query": query,
            "normalized_query": q_norm,
            "router_intent": route_result["intent"],
            "router_probabilities": route_result["probabilities"],
            "k_sop": k_sop,
            "k_bpmn": k_bpmn,
            "retrieved_sop_chunks": len(sop_chunks),
            "retrieved_sop_titles": [c["title"] for c in sop_chunks],
            "sop_chunks": sop_chunks,
            "fused_context_length": len(fused_context),
            "fused_context": fused_context,
            "draft_answer": gen_res["answer"],
            "final_answer": current_answer,
            "audit_passed": audit_res["passed"],
            "final_psi": audit_res["psi_vector"],
            "repair_iterations": repair_iterations,
            "repair_history": repair_history,
            "latencies": phase_latencies,
            "total_latency_seconds": total_latency
        }
