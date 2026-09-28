"""
Streamlit Web Application: ICICoS ProcessQA Intelligence Studio
Interactive interface for 'Adaptive Agentic Hybrid RAG with BPMN-Grounded Graph Reasoning'
Provides:
1. Live Interactive Process QA Studio with Full 6-Phase Trace & Comparative Evaluation
2. Empirical Benchmark Results Explorer (Tables III, IV, V, VI, VII, IX & Figures 1-4)
3. Dual-Modal Enterprise Knowledge Space Explorer (SOPs, BPMN DiGraphs, Dataset)
"""

import os
import json
import time
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

from src.config import BASE_DIR, RESULTS_DIR, DATASET_PATH, SOP_DIR, BPMN_DIR
from src.pipeline import AdaptiveAgenticHybridRAGPipeline
from src.baselines import BaselineModels
from src.evaluator import BenchmarkEvaluator

# Page Configuration
st.set_page_config(
    page_title="ICICoS ProcessQA • Adaptive Agentic Hybrid RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(120deg, #1e3a8a, #3b82f6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #4b5563;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-green { background-color: #d1fae5; color: #065f46; }
    .badge-blue { background-color: #dbeafe; color: #1e40af; }
    .badge-purple { background-color: #f3e8ff; color: #6b21a8; }
    .badge-amber { background-color: #fef3c7; color: #92400e; }
    .phase-card {
        border-left: 4px solid #3b82f6;
        padding: 10px 16px;
        background-color: #f8fafc;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline_and_baselines():
    """Initializes and caches the multi-agent pipeline and baselines."""
    pipeline = AdaptiveAgenticHybridRAGPipeline()
    baselines = BaselineModels()
    evaluator = BenchmarkEvaluator()
    return pipeline, baselines, evaluator

@st.cache_data
def load_dataset():
    if Path(DATASET_PATH).exists():
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# Sidebar Navigation & Information
st.sidebar.markdown("## 🧭 Navigation")
app_mode = st.sidebar.radio(
    "Choose View:",
    [
        "🎯 Live Process QA Studio",
        "📊 Benchmark & Statistical Results",
        "🔬 Component Ablation Study",
        "📚 Knowledge Space & BPMN Explorer"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Pipeline Specifications")
st.sidebar.markdown("""
* **Dual-Modal Knowledge Space**: $\\mathcal{K} = (\\mathcal{K}_{sop}, \\mathcal{K}_{bpmn})$
* **Runtime Orchestration**: 6 Execution Phases
* **Verification Threshold**: $\\tau = 0.85$
* **Max Self-Repair Iterations**: $N_{max} = 2$
* **Corpus Size**: $N = 338$ ProcessQA Queries
""")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Paper Mimicked**: *Adaptive Agentic Hybrid RAG with BPMN-Grounded Graph Reasoning for Explainable Enterprise Process Question Answering* (Lintang et al.)")

pipeline, baselines, evaluator = load_pipeline_and_baselines()
dataset = load_dataset()

# =========================================================================
# VIEW 1: LIVE INTERACTIVE PROCESS QA STUDIO
# =========================================================================
if app_mode == "🎯 Live Process QA Studio":
    st.markdown('<div class="main-header">⚡ ICICoS ProcessQA Intelligence Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">BPMN-Grounded Agentic Reasoning for Enterprise Operational Compliance</div>', unsafe_allow_html=True)

    # Preset query catalog
    preset_dict = {
        "Document: IEEE Member Fee & Attachments": "Berapa biaya registrasi untuk IEEE Member Author dan dokumen apa yang wajib dilampirkan untuk mendapatkan potongan?",
        "Document: Keynote Speaker Honorarium (SPJ)": "What mandatory attachments must accompany a keynote speaker honorarium payment voucher (SPJ)?",
        "Document: Page Limit & Overlength Surcharge": "Berapa batas halaman standar paper ICICoS dan berapa biaya kelebihan per halaman jika naskah melebihi batas?",
        "Document: 5 Mandatory Registration Attachments": "What five mandatory attachments must be compiled and uploaded during the conference registration phase?",
        "Document: Turnitin Similarity Threshold": "Berapa batas maksimal similarity Turnitin pada tahap desk review ICICoS dan berapa threshold acceptance score peer review?",
        "Workflow: Registration Payment to LoA Sequence": "Bagaimana urutan tahapan yang benar dari pengunggahan bukti bayar hingga penerbitan LoA?",
        "Workflow: Peer Review Chronological Flow": "What is the chronological sequence of peer review from initial manuscript submission to the final camera-ready invitation?",
        "Workflow: Swimlane Actor Responsibilities": "Siapa saja aktor yang bertugas dalam proses registrasi dan apa tanggung jawab masing-masing?",
        "Workflow: Camera-Ready IEEE Publication Steps": "What sequence of steps must an author take to publish a camera-ready paper to IEEE Xplore after receiving acceptance?",
        "Gateway: Payment Discrepancy & Rejection Handling": "What sequence of actions occurs after a payment invoice or receipt is rejected due to discrepancy?",
        "Gateway: Review Score 3.10 Branch Decision": "Apa keputusan dan jalur alur yang terjadi jika sebuah naskah memperoleh rata-rata review score 3.10?",
        "Gateway: Camera-Ready Format Audit Defects": "What happens at the Format Audit Gateway if a camera-ready submission has missing IEEE copyright notices or margin defects?",
        "Gateway: LoA Issuance Precedence Barrier": "Apakah author bisa mendapatkan LoA jika bukti transfer belum diverifikasi oleh Bendahara?",
        "Gateway: Turnitin 23% Plagiarism Desk Rejection": "Bagaimana penanganan naskah jika Turnitin menunjukkan similarity 23% pada tahap desk review?",
        "Hybrid: Student Fee + Checklist + Verification Flow": "Biaya registrasi mahasiswa berapa, dokumen apa saja yang harus diunggah, dan siapa yang memverifikasi pembayarannya?",
        "Hybrid: Second Paper Discount & Audit Procedure": "Berapa potongan biaya registrasi untuk paper kedua penulis yang sama, dan bagaimana alur verifikasinya?",
        "Hybrid: 7-Page Overlength Non-Member Calculation": "Jika paper 7 halaman diterima, berapa total biaya registrasi non-member dan bagaimana tahapan publikasi camera ready hingga IEEE Xplore?",
        "Hybrid: Underpayment Remediation & Grace Period": "Bagaimana prosedur penanganan jika penulis melakukan salah transfer kurang bayar dan berapa lama tenggat waktu perbaikannya?"
    }

    col_select, col_mode = st.columns([3, 1.2])
    with col_select:
        selected_preset = st.selectbox(
            "Select an authentic enterprise benchmark query (or type custom below):",
            ["Custom Query"] + list(preset_dict.keys())
        )
    with col_mode:
        eval_architecture = st.selectbox(
            "Architecture to Execute:",
            [
                "🚀 5. Proposed: Adaptive Agentic Hybrid RAG",
                "⚡ 4. Static Hybrid RAG (Uniform Concat)",
                "📄 2. Traditional Vector RAG",
                "🕸️ 3. Graph RAG (BPMN Subgraph Only)",
                "🤖 1. LLM Only (Zero-shot Baseline)",
                "⚖️ Compare: Proposed vs. Static Hybrid Side-by-Side"
            ]
        )

    default_query = preset_dict.get(selected_preset, "") if selected_preset != "Custom Query" else ""
    query_input = st.text_area("User Process Query (Indonesian or English):", value=default_query, height=85)

    run_btn = st.button("🚀 Execute Enterprise RAG Reasoning", type="primary", use_container_width=True)

    if run_btn and query_input.strip():
        st.markdown("---")
        
        # -------------------------------------------------------------
        # COMPARATIVE SIDE-BY-SIDE MODE
        # -------------------------------------------------------------
        if "Compare:" in eval_architecture:
            col_prop, col_stat = st.columns(2)
            
            with col_prop:
                st.subheader("🚀 Proposed: Adaptive Agentic Hybrid RAG")
                with st.spinner("Executing 6-Phase Adaptive Agentic Pipeline..."):
                    t0 = time.time()
                    res_prop = pipeline.run(query_input)
                    lat_prop = time.time() - t0
                    f_prop = evaluator.compute_faithfulness(res_prop["final_answer"], res_prop["fused_context"])

                st.markdown(f"**Latency:** `{lat_prop:.2f}s` | **Faithfulness:** `{f_prop:.3f}` | **Router:** `{res_prop['router_intent']}`")
                st.markdown(f"**Critic Verification:** {'✅ PASSED' if res_prop['audit_passed'] else '⚠️ REPAIRED'}")
                st.info(res_prop["final_answer"])

            with col_stat:
                st.subheader("⚡ Baseline: Static Hybrid RAG")
                with st.spinner("Executing Static Concatenation RAG..."):
                    t0 = time.time()
                    res_stat = baselines.run_static_hybrid_rag(query_input)
                    lat_stat = time.time() - t0
                    f_stat = evaluator.compute_faithfulness(res_stat["answer"], res_stat["context"])

                st.markdown(f"**Latency:** `{lat_stat:.2f}s` | **Faithfulness:** `{f_stat:.3f}` | **Router:** `None (Static 5+3)`")
                st.markdown("**Critic Verification:** `None (Single-Pass)`")
                st.warning(res_stat["answer"])

        # -------------------------------------------------------------
        # SINGLE ARCHITECTURE EXECUTION WITH DETAILED 6-PHASE TRACE
        # -------------------------------------------------------------
        else:
            if "Proposed" in eval_architecture:
                with st.spinner("Executing Full 6-Phase Multi-Agent Execution Sequence..."):
                    t_start = time.time()
                    res = pipeline.run(query_input)
                    total_dur = time.time() - t_start
                    faith_score = evaluator.compute_faithfulness(res["final_answer"], res["fused_context"])

                # Header metrics
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Predicted Intent", res["router_intent"])
                m2.metric("SOP / BPMN Quotas", f"{res['k_sop']} SOP + {res['k_bpmn']} BPMN")
                m3.metric("Critic Audit Vector", "Ψ Passed" if res["audit_passed"] else f"Repaired ({res['repair_iterations']})")
                m4.metric("Faithfulness Score", f"{faith_score:.3f}")
                m5.metric("End-to-End Latency", f"{total_dur:.2f}s")

                st.markdown("### 📋 Final Verified Process Answer")
                st.success(res["final_answer"])

                # Expandable multi-phase trace
                st.markdown("### 🔍 Transparent 6-Phase Runtime Execution Trace")

                with st.expander("🔹 Phase 1: Query Understanding Agent (QUA) Normalization", expanded=True):
                    st.markdown(f"**Raw Query:** `{res['query']}`")
                    st.markdown(f"**Normalized Query ($q'$):** `{res['normalized_query']}`")
                    st.caption("Applied acronym expansion ($D_{acronym}$) and grounded entity mentions to canonical BPMN task labels.")

                with st.expander("🔹 Phase 2: Adaptive Intent Router Agent & Dynamic Quotas", expanded=True):
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.markdown(f"**Identified Operational Intent:** `{res['router_intent']}`")
                        st.markdown(f"**Assigned Quotas $k(r)$:** `{res['k_sop']}` SOP passages + `{res['k_bpmn']}` BPMN graph hops")
                    with c2:
                        probs = res["router_probabilities"]
                        prob_df = pd.DataFrame({
                            "Intent": list(probs.keys()),
                            "Softmax Probability": [probs[k] for k in probs.keys()]
                        })
                        st.bar_chart(prob_df.set_index("Intent"), height=160)

                with st.expander("🔹 Phase 3 & 4: Dual Retrieval & Cross-Modal Context Fusion ($C_{fused}$)", expanded=False):
                    st.markdown(f"**Retrieved SOP Sections:** {len(res['sop_chunks'])} passages")
                    for t in res["retrieved_sop_titles"]:
                        st.markdown(f"- 📄 `{t}`")
                    st.markdown("---")
                    st.markdown("**Unified Fused Context ($C_{fused}$):**")
                    st.code(res["fused_context"], language="markdown")

                with st.expander("🔹 Phase 5 & 6: Grounded Generation, Adversarial Critic Audit & Self-Repair", expanded=True):
                    c_psi, c_rep = st.columns([1.2, 1])
                    with c_psi:
                        st.markdown("**Critic Audit Vector $\\Psi = \\langle \\psi_{fact}, \\psi_{proc}, \\psi_{actor}, \\psi_{comp} \\rangle$:**")
                        psi = res["final_psi"]
                        st.write(pd.DataFrame({
                            "Audit Dimension": [
                                "Factual Entailment (ψ_fact)",
                                "Process Precedence (ψ_proc)",
                                "Actor Attribution (ψ_actor)",
                                "Checklist Completeness (ψ_comp)"
                            ],
                            "Score (0.0 - 1.0)": [
                                f"{psi['psi_fact']:.2f}",
                                f"{psi['psi_proc']:.2f}",
                                f"{psi['psi_actor']:.2f}",
                                f"{psi['psi_comp']:.2f}"
                            ],
                            "Threshold (τ)": ["0.85", "0.85", "0.85", "0.85"],
                            "Compliance Status": [
                                "✅ Pass" if psi["psi_fact"] >= 0.85 else "⚠️ Violation",
                                "✅ Pass" if psi["psi_proc"] >= 0.85 else "⚠️ Inversion",
                                "✅ Pass" if psi["psi_actor"] >= 0.85 else "⚠️ Confusion",
                                "✅ Pass" if psi["psi_comp"] >= 0.85 else "⚠️ Incomplete"
                            ]
                        }))

                    with c_rep:
                        st.markdown(f"**Self-Repair Iterations Completed:** `{res['repair_iterations']}` (max: 2)")
                        if res["repair_history"]:
                            for h in res["repair_history"]:
                                st.caption(f"Iteration {h['iteration']}: Passed = {h['passed']}")
                                if h.get("violations"):
                                    for v in h["violations"]:
                                        st.caption(f"⚠️ *{v}*")
                        st.markdown("**Draft Answer ($A_{draft}$):**")
                        st.text_area("Raw Draft before Critic Audit:", value=res["draft_answer"][:400] + "...", height=110, disabled=True)

            else:
                # Other baselines
                with st.spinner("Running Baseline Architecture..."):
                    if "Static Hybrid" in eval_architecture:
                        res = baselines.run_static_hybrid_rag(query_input)
                    elif "Vector RAG" in eval_architecture:
                        res = baselines.run_vector_rag(query_input)
                    elif "Graph RAG" in eval_architecture:
                        res = baselines.run_graph_rag(query_input)
                    else:
                        res = baselines.run_zero_shot(query_input)

                    faith_score = evaluator.compute_faithfulness(res["answer"], res["context"])

                st.subheader(f"Results for {res['architecture']}")
                st.markdown(f"**Latency:** `{res['latency_seconds']:.2f}s` | **Faithfulness:** `{faith_score:.3f}`")
                st.warning(res["answer"])

                with st.expander("Retrieved Context Provided to Model:", expanded=False):
                    st.code(res["context"], language="markdown")

# =========================================================================
# VIEW 2: EMPIRICAL BENCHMARK & STATISTICAL RESULTS
# =========================================================================
elif app_mode == "📊 Benchmark & Statistical Results":
    st.markdown('<div class="main-header">📊 ICICoS ProcessQA Master Benchmark Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comprehensive Empirical Evaluation across All 5 Architectures & Diagnostic Metrics</div>', unsafe_allow_html=True)

    tab_t3, tab_t4, tab_t5, tab_t6, tab_t7, tab_figs = st.tabs([
        "Table III: Primary Benchmark",
        "Table IV: Category Breakdown",
        "Table V: BPMN Reasoning Diagnostics",
        "Table VI: Router Token Budget",
        "Table VII: Statistical Hypothesis Testing",
        "🖼️ Publication Figures (1-4)"
    ])

    with tab_t3:
        st.markdown("### Table III: Primary Benchmark Experimental Results ($N = 20$ Stratified Core Testbed)")
        st.markdown("Comprehensive evaluation across Standard RAG Literature Metrics:")
        t3_path = RESULTS_DIR / "final_comparison_table.csv"
        if t3_path.exists():
            df3 = pd.read_csv(t3_path)
            st.dataframe(df3, use_container_width=True)
        else:
            st.warning("Table III CSV not found.")

    with tab_t4:
        st.markdown("### Table IV: Operational Performance Breakdown across Process Categories")
        st.markdown("Breakdown of Proposed Architecture performance across Document, Workflow, Gateway, and Hybrid queries:")
        t4_path = RESULTS_DIR / "operational_category_breakdown.csv"
        if t4_path.exists():
            df4 = pd.read_csv(t4_path)
            st.dataframe(df4, use_container_width=True)

    with tab_t5:
        st.markdown("### Table V: Procedural BPMN Reasoning Diagnostic Benchmark")
        st.markdown("Level 2 Diagnostic Metrics: Process Path Accuracy (PPA), Gateway Decision Accuracy (GDA), and Actor Attribution Accuracy (AAA):")
        t5_path = RESULTS_DIR / "bpmn_diagnostic_results.csv"
        if t5_path.exists():
            df5 = pd.read_csv(t5_path)
            st.dataframe(df5, use_container_width=True)

    with tab_t6:
        st.markdown("### Table VI: Router Efficiency & Context Budget Optimization ($N = 338$)")
        st.markdown("Token consumption comparison between Static Uniform Concatenation and Softmax Adaptive Dynamic Quotas:")
        t6_path = RESULTS_DIR / "router_efficiency_results.csv"
        if t6_path.exists():
            df6 = pd.read_csv(t6_path)
            st.dataframe(df6, use_container_width=True)

    with tab_t7:
        st.markdown("### Table VII: Paired Statistical Hypothesis Testing (Proposed vs. Static Hybrid)")
        st.markdown("Rigorous validation using Wilcoxon Signed-Rank tests, Paired Student's t-tests, and 10,000-resample Bootstrap 95% Confidence Intervals:")
        t7_path = RESULTS_DIR / "statistical_hypothesis_results.csv"
        if t7_path.exists():
            df7 = pd.read_csv(t7_path)
            st.dataframe(df7, use_container_width=True)

    with tab_figs:
        st.markdown("### High-Resolution Publication Figures (300 DPI)")
        c1, c2 = st.columns(2)
        with c1:
            fig1 = RESULTS_DIR / "fig1_benchmark_faithfulness_hallucination.png"
            if fig1.exists():
                st.image(str(fig1), caption="Figure 1: Primary Grounding & Hallucination Rate across 5 Architectures", use_container_width=True)
            
            fig3 = RESULTS_DIR / "fig3_latency_decomposition.png"
            if fig3.exists():
                st.image(str(fig3), caption="Figure 3: End-to-End Latency Decomposition across 6 Phases", use_container_width=True)

        with c2:
            fig2 = RESULTS_DIR / "fig2_bpmn_diagnostic_comparison.png"
            if fig2.exists():
                st.image(str(fig2), caption="Figure 2: Procedural BPMN Reasoning Diagnostic Comparison (PPA, GDA, AAA)", use_container_width=True)

            fig4 = RESULTS_DIR / "fig4_router_token_efficiency.png"
            if fig4.exists():
                st.image(str(fig4), caption="Figure 4: Context Budget Savings (-12.9% Token Reduction)", use_container_width=True)

# =========================================================================
# VIEW 3: COMPONENT ABLATION STUDY
# =========================================================================
elif app_mode == "🔬 Component Ablation Study":
    st.markdown('<div class="main-header">🔬 Systematic Component Ablation Study</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Table IX: Isolating Contributions of the Router, BPMN Graph Retriever, and Adversarial Critic</div>', unsafe_allow_html=True)

    t9_path = RESULTS_DIR / "ablation_study_results.csv"
    if t9_path.exists():
        df9 = pd.read_csv(t9_path)
        st.dataframe(df9, use_container_width=True)

    st.markdown("---")
    st.markdown("### 💡 Component Failure Modality Analysis")
    
    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        st.error("❌ Ablation 1: w/o Adaptive Router")
        st.markdown("""
        * **Failure Mechanism**: *Modality Dilution*
        * **Impact**: Fixed uniform retrieval quotas ($\langle 5, 3 \rangle$) waste valuable context budget on non-salient modalities.
        * **Symptom**: Pure document queries are flooded with irrelevant process steps, while procedural questions receive insufficient graph depth.
        """)

    with col_a2:
        st.error("❌ Ablation 2: w/o BPMN Graph Retriever")
        st.markdown("""
        * **Failure Mechanism**: *Topological Scrambling*
        * **Impact**: Process Consistency Score (PCS) collapses by **-41.7% (0.942 $\to$ 0.549)**.
        * **Symptom**: Flat text embeddings fail to preserve directed chronological precedence ($u \leadsto_G v$). Opposing XOR branches conflate into contradictory sequences.
        """)

    with col_a3:
        st.error("❌ Ablation 3: w/o Critic Verification")
        st.markdown("""
        * **Failure Mechanism**: *Hallucination Leakage*
        * **Impact**: Hallucination increases to **25.8%**, with role confusion bypassing output filters.
        * **Symptom**: Single-pass generation lacks self-repair feedback $\Delta_{repair}$, causing unverified checklist omissions and swimlane misattributions.
        """)

# =========================================================================
# VIEW 4: KNOWLEDGE SPACE & BPMN EXPLORER
# =========================================================================
elif app_mode == "📚 Knowledge Space & BPMN Explorer":
    st.markdown('<div class="main-header">📚 Dual-Modal Enterprise Knowledge Space</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspection of Standard Operating Procedures (SOPs), BPMN DiGraphs, and Evaluation Corpus</div>', unsafe_allow_html=True)

    tab_sop, tab_bpmn, tab_corpus = st.tabs([
        "📄 Standard Operating Procedures (SOPs)",
        "🕸️ BPMN XML Workflow Models",
        "📂 Benchmark Dataset Explorer (N = 338)"
    ])

    with tab_sop:
        st.markdown("### Authentic ICICoS Conference Standard Operating Procedures")
        sop_files = list(Path(SOP_DIR).glob("*.md"))
        selected_sop = st.selectbox("Choose SOP Document:", [f.name for f in sop_files])
        if selected_sop:
            with open(Path(SOP_DIR) / selected_sop, "r", encoding="utf-8") as f:
                st.markdown(f.read())

    with tab_bpmn:
        st.markdown("### Authentic OMG ISO/IEC 19510 XML BPMN Process Models")
        bpmn_files = list(Path(BPMN_DIR).glob("*.bpmn"))
        selected_bpmn = st.selectbox("Choose BPMN Workflow:", [f.name for f in bpmn_files])
        if selected_bpmn:
            with open(Path(BPMN_DIR) / selected_bpmn, "r", encoding="utf-8") as f:
                content = f.read()
                st.code(content[:2500] + "\n... [truncated]", language="xml")

    with tab_corpus:
        st.markdown("### ICICoS ProcessQA Evaluation Corpus ($N = 338$)")
        if dataset:
            df_ds = pd.DataFrame(dataset)
            cat_filter = st.multiselect(
                "Filter by Operational Category:",
                options=df_ds["category"].unique().tolist(),
                default=df_ds["category"].unique().tolist()
            )
            filtered_df = df_ds[df_ds["category"].isin(cat_filter)]
            st.dataframe(filtered_df[["id", "category", "query", "ground_truth_answer", "expected_intent"]], use_container_width=True)
