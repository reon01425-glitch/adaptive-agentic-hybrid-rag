"""
Interactive CLI Tester for Adaptive Agentic Hybrid RAG with BPMN-Grounded Reasoning.
Allows users to input arbitrary queries and inspect all 6 runtime phases step-by-step:
1. Query Understanding & Acronym Expansion
2. Adaptive Intent Router & Dynamic Quotas
3. Dual Retrieval Execution
4. Cross-Modal Context Fusion
5. Grounded LLM Generation
6. Adversarial Critic Audit & Closed-Loop Self-Repair
"""

import sys
import json
from src.pipeline import AdaptiveAgenticHybridRAGPipeline

def run_interactive_cli():
    print("=" * 80)
    print("🤖 ICICOS ADAPTIVE AGENTIC HYBRID RAG INTERACTIVE EXPLORER")
    print("=" * 80)
    
    pipeline = AdaptiveAgenticHybridRAGPipeline()
    print("\nSystem ready! Type your question below (or 'exit' to quit).")
    print("Example Queries:")
    print("  1. 'Berapa biaya registrasi IEEE Member dan dokumen apa yang wajib dilampirkan?'")
    print("  2. 'Bagaimana urutan tahapan yang benar dari bayar hingga LoA terbit?'")
    print("  3. 'Apa yang terjadi jika pembayaran ditolak karena ada diskrepansi nominal transfer?'")
    print("  4. 'Berapa biaya registrasi mahasiswa dan siapa saja aktor yang berwenang memvalidasi?'")
    print("-" * 80)

    # Check if a query was passed as command line argument
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        queries = [query]
    else:
        # Default showcase queries
        queries = [
            "Berapa biaya registrasi untuk IEEE Member Author dan dokumen apa yang wajib dilampirkan?",
            "Bagaimana urutan tahapan yang benar dari pengunggahan bukti bayar hingga penulis mendapatkan LoA?",
            "What sequence of actions occurs after a payment invoice or receipt is rejected due to discrepancy by finance?",
            "Biaya registrasi mahasiswa berapa, dokumen apa saja yang harus diunggah, dan siapa yang memvalidasi hingga LoA terbit?"
        ]

    for q in queries:
        print(f"\n👉 USER QUERY: {q}")
        print("=" * 80)
        res = pipeline.run(q)

        print("\n[PHASE 1: QUERY UNDERSTANDING AGENT (QUA)]")
        print(f"  • Normalized Query (q'): {res['normalized_query']}")
        print(f"  • Execution Time: {res['latencies']['phase1_qua']}s")

        print("\n[PHASE 2: ADAPTIVE INTENT ROUTER AGENT]")
        print(f"  • Predicted Intent R(q'): {res['router_intent']}")
        print(f"  • Softmax Probabilities P(r|q'): {res['router_probabilities']}")
        print(f"  • Dynamic Quota Allocation k(r): <k_sop={res['k_sop']}, k_bpmn={res['k_bpmn']}>")
        print(f"  • Execution Time: {res['latencies']['phase2_router']}s")

        print("\n[PHASE 3: TARGETED DUAL RETRIEVAL]")
        print(f"  • Retrieved SOP Chunks ({res['retrieved_sop_chunks']}): {res['retrieved_sop_titles']}")
        print(f"  • Execution Time: {res['latencies']['phase3_retrieval']}s")

        print("\n[PHASE 4: CROSS-MODAL CONTEXT FUSION]")
        print(f"  • Fused Context Length: {res['fused_context_length']} characters")
        print(f"  • Execution Time: {res['latencies']['phase4_fusion']}s")

        print("\n[PHASE 5: GROUNDED GENERATION (DRAFT)]")
        print(f"  • Draft Answer (A_draft):\n{res['draft_answer']}")
        print(f"  • Execution Time: {res['latencies']['phase5_draft_gen']}s")

        print("\n[PHASE 6: ADVERSARIAL CRITIC VERIFICATION & SELF-REPAIR]")
        print(f"  • Audit Vector Psi: {res['final_psi']}")
        print(f"  • Audit Compliance: {'PASSED (Audit-Ready)' if res['audit_passed'] else 'REJECTED'}")
        print(f"  • Repair Iterations: {res['repair_iterations']}")
        if res['repair_iterations'] > 0:
            for rep in res['repair_history']:
                print(f"    - Iteration {rep['iteration']}: Violations = {rep['violations']}")
        print(f"  • Execution Time: {res['latencies']['phase6_critic_audit']}s")

        print("\n[FINAL AUDIT-COMPLIANT OUTPUT (A_final)]")
        print("=" * 80)
        print(res["final_answer"])
        print("=" * 80)
        print(f"⏱️ Total End-to-End Latency: {res['total_latency_seconds']} seconds\n")

if __name__ == "__main__":
    run_interactive_cli()
