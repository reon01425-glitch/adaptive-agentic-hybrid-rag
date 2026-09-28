"""
Baseline Architectures for ICICoS Benchmark Comparison (Section IV-C in Paper):
1. LLM Only (Zero-shot) (k_sop=0, k_bpmn=0)
2. Traditional Vector RAG (k_sop=6, k_bpmn=0)
3. Graph RAG (BPMN Subgraph Only) (k_sop=0, k_bpmn=4)
4. Static Hybrid RAG (Direct Concatenation) (k_sop=5, k_bpmn=3, no agents, no critic)
"""

import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from src.config import BPMN_DIR, SOP_DIR, QUOTAS
from src.bpmn_parser import MultiBPMNRepository
from src.sop_indexer import SOPIndexer
from src.generator import GroundedGenerator

class BaselineModels:
    """Implements all 4 comparative baseline architectures."""

    def __init__(self, bpmn_dir: Path = BPMN_DIR, sop_dir: Path = SOP_DIR):
        self.bpmn_repo = MultiBPMNRepository(bpmn_dir)
        self.sop_indexer = SOPIndexer(sop_dir)
        self.generator = GroundedGenerator()

    def run_zero_shot(self, query: str) -> Dict[str, Any]:
        """
        Baseline 1: LLM Only (Zero-shot)
        Parametric generation without external retrieval (k_sop=0, k_bpmn=0).
        """
        t0 = time.time()
        context = "No external documents provided. Answer relying solely on parametric knowledge."
        gen_res = self.generator.generate_draft(query=query, context=context)
        latency = round(time.time() - t0, 3)

        return {
            "architecture": "LLM Only (Zero-shot)",
            "query": query,
            "context": context,
            "retrieved_sop_chunks": 0,
            "retrieved_bpmn_nodes": 0,
            "answer": gen_res["answer"],
            "latency_seconds": latency
        }

    def run_vector_rag(self, query: str, k_sop: int = 6) -> Dict[str, Any]:
        """
        Baseline 2: Traditional Vector RAG
        Standard dense semantic retrieval over SOP text chunks (k_sop=6, k_bpmn=0).
        Suffers from metric distance symmetry / topological scrambling.
        """
        t0 = time.time()
        # Flat text retrieval
        chunks = self.sop_indexer.retrieve(query=query, top_k=k_sop)
        
        context_parts = ["=== RETRIEVED SOP TEXT PASSAGES (Vector Search) ==="]
        for idx, c in enumerate(chunks, 1):
            context_parts.append(f"Passage {idx} ({c['title']}):\n{c['content']}")
        context = "\n\n".join(context_parts)

        gen_res = self.generator.generate_draft(query=query, context=context)
        latency = round(time.time() - t0, 3)

        return {
            "architecture": "Traditional Vector RAG",
            "query": query,
            "context": context,
            "retrieved_sop_chunks": len(chunks),
            "retrieved_bpmn_nodes": 0,
            "answer": gen_res["answer"],
            "latency_seconds": latency,
            "retrieved_chunks": chunks
        }

    def run_graph_rag(self, query: str, k_bpmn: int = 4) -> Dict[str, Any]:
        """
        Baseline 3: Graph RAG (BPMN Subgraph Only)
        Traverses NetworkX subgraphs without textual SOP enrichment (k_sop=0, k_bpmn=4).
        """
        t0 = time.time()
        matched_nodes = []
        for n in self.bpmn_repo.get_all_nodes():
            n_label = n.get("label", "").lower()
            if any(w in query.lower() for w in n_label.split() if len(w) > 3):
                matched_nodes.append(n["id"])
        if not matched_nodes:
            q_l = query.lower()
            if "review" in q_l or "turnitin" in q_l or "similarity" in q_l or "rubric" in q_l:
                matched_nodes = ["Task_SubmitManuscript", "Task_DeskReview", "Gateway_ReviewDecision"]
            elif "camera" in q_l or "xpress" in q_l or "copyright" in q_l or "publikasi" in q_l:
                matched_nodes = ["Task_SubmitCameraReady", "Gateway_FormatAudit", "Task_IEEETransfer"]
            else:
                matched_nodes = ["Task_SubmitRequest", "Task_AuditPayment", "Gateway_PaymentCheck"]

        graph_text = self.bpmn_repo.query_subgraph(matched_nodes[:k_bpmn], hops=2)
        context = f"=== BPMN GRAPH TOPOLOGY (GraphRAG) ===\n{graph_text}"

        gen_res = self.generator.generate_draft(query=query, context=context)
        latency = round(time.time() - t0, 3)

        return {
            "architecture": "Graph RAG (BPMN Subgraph)",
            "query": query,
            "context": context,
            "retrieved_sop_chunks": 0,
            "retrieved_bpmn_nodes": len(matched_nodes[:k_bpmn]),
            "answer": gen_res["answer"],
            "latency_seconds": latency,
            "graph_text": graph_text
        }

    def run_static_hybrid_rag(self, query: str, k_sop: int = 5, k_bpmn: int = 3) -> Dict[str, Any]:
        """
        Baseline 4: Static Hybrid RAG (Direct Concatenation)
        Uniform retrieval (k_sop=5, k_bpmn=3) without QUA, without router, without critic.
        """
        t0 = time.time()
        # 1. Fetch text chunks
        chunks = self.sop_indexer.retrieve(query=query, top_k=k_sop)
        
        # 2. Fetch graph sub-topology
        matched_nodes = []
        for n in self.bpmn_repo.get_all_nodes():
            n_label = n.get("label", "").lower()
            if any(w in query.lower() for w in n_label.split() if len(w) > 3):
                matched_nodes.append(n["id"])
        if not matched_nodes:
            q_l = query.lower()
            if "review" in q_l or "turnitin" in q_l or "similarity" in q_l or "rubric" in q_l:
                matched_nodes = ["Task_SubmitManuscript", "Task_DeskReview", "Gateway_ReviewDecision"]
            elif "camera" in q_l or "xpress" in q_l or "copyright" in q_l or "publikasi" in q_l:
                matched_nodes = ["Task_SubmitCameraReady", "Gateway_FormatAudit", "Task_IEEETransfer"]
            else:
                matched_nodes = ["Task_SubmitRequest", "Task_AuditPayment", "Gateway_PaymentCheck"]
        graph_text = self.bpmn_repo.query_subgraph(matched_nodes[:k_bpmn], hops=2)

        # 3. Direct concatenation without domain isolation or node-chunk projection
        context = (
            f"=== DIRECT CONCATENATION: SOP TEXT + PROCESS GRAPH ===\n\n"
            f"[BPMN GRAPH]\n{graph_text}\n\n"
            f"[SOP TEXT CHUNKS]\n" +
            "\n\n".join([f"Chunk {i+1} ({c['title']}):\n{c['content']}" for i, c in enumerate(chunks)])
        )

        # 4. Single-pass generation
        gen_res = self.generator.generate_draft(query=query, context=context)
        latency = round(time.time() - t0, 3)

        return {
            "architecture": "Static Hybrid RAG",
            "query": query,
            "context": context,
            "retrieved_sop_chunks": len(chunks),
            "retrieved_bpmn_nodes": len(matched_nodes[:k_bpmn]),
            "answer": gen_res["answer"],
            "latency_seconds": latency,
            "retrieved_chunks": chunks
        }
