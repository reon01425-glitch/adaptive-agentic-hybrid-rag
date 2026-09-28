# Adaptive Agentic Hybrid RAG with BPMN-Grounded Graph Reasoning

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App%20Live-FF4B4B.svg)](http://localhost:8501)
[![Conference Benchmark](https://img.shields.io/badge/Benchmark-ICICoS%20ProcessQA%20(N%3D338)-purple.svg)](#)

> **Official Implementation & Reproducibility Package** for the research paper:  
> *"Adaptive Agentic Hybrid RAG with BPMN-Grounded Graph Reasoning for Explainable Enterprise Process Question Answering"* (Lintang et al., 2026).

---

## 📌 Executive Summary

Traditional Retrieval-Augmented Generation (RAG) paradigms face critical failure modes when applied to regulated enterprise workflows:
1. **Topological Scrambling**: Unstructured metric vector spaces treat directional steps as symmetrically related text passages ($dist(u,v) = dist(v,u)$), scrambling chronological sequence order.
2. **Happy-Path Hallucination Bias**: Models collapse exclusive XOR decision alternatives, ignoring prerequisite audit barriers, grace periods, and exception handling paths.
3. **Role Confusion**: Swimlane authorizations across organizational actors (Author, Treasurer, Secretariat, TPC Chair, Reviewer) are conflated.

This framework resolves these inductive limitations by establishing a **Dual-Modal Enterprise Knowledge Space** $\mathcal{K} = (\mathcal{K}_{sop}, \mathcal{K}_{bpmn})$, orchestrating 3 architectural layers and 6 runtime execution phases.

```
       Dual-Modal Knowledge Space: K = (K_sop, K_bpmn)
  ┌────────────────────────────────────────────────────────┐
  │                                                        │
  │  [K_sop: AST Chunks]             [K_bpmn: DiGraph G]   │
  │   - Author Guidelines             - Registration Flow  │
  │   - Payment & SPJ Policies        - Peer Review Flow   │
  │   - Turnitin Limits               - Camera-Ready Flow  │
  │                                                        │
  └───────────────────────────┬────────────────────────────┘
                              │
  ┌───────────────────────────▼────────────────────────────┐
  │         Multi-Agent Orchestration & Retrieval          │
  │                                                        │
  │  Phase 1: Query Understanding Agent (QUA - D_acronym)  │
  │  Phase 2: Adaptive Intent Router (Softmax Quotas k(r)) │
  │  Phase 3: Dual Retrieval (Dense Cosine + DiGraph DFS)  │
  │                                                        │
  └───────────────────────────┬────────────────────────────┘
                              │
  ┌───────────────────────────▼────────────────────────────┐
  │   Context Fusion, Grounded Generation & Critic Loop    │
  │                                                        │
  │  Phase 4: Cross-Modal Context Fusion (C_fused)         │
  │  Phase 5: Grounded LLM Generator (Draft A_draft)       │
  │  Phase 6: Adversarial Critic Verification (Ψ >= 0.85)  │
  │           └─ Closed-Loop Self-Repair (Δ_repair, N <= 2)│
  │                                                        │
  └────────────────────────────────────────────────────────┘
```

---

## 🏆 Key Experimental Results

Evaluated on the **ICICoS ProcessQA** benchmark suite ($N = 338$ complete corpus; $N = 20$ stratified core benchmark across 4 operational categories):

* **100% Actor Attribution Accuracy (AAA = 1.000)**: Completely eliminates swimlane role diffusion across all 4 operational categories.
* **98.8% Gateway Decision Accuracy (GDA = 0.988)**: Eliminates happy-path bias on conditional branching and remediation cycles.
* **87.5% Process Path Accuracy (PPA = 0.875)**: Enforces directed graph transitions without sequence inversions.
* **-12.9% Context Token Reduction**: The Softmax Adaptive Intent Router saves 35,360 tokens over static concatenation across $N = 338$ queries with **99.1% lenient routing accuracy**.
* **Essential Graph Grounding**: Component ablation demonstrates that removing the BPMN Graph Retriever causes a severe **-41.7% collapse in Process Consistency Score (PCS: 0.942 → 0.549)**.

---

## 📁 Repository Structure

```
├── app.py                      # Interactive Streamlit Web App (localhost:8501)
├── run_experiments.py          # Master benchmark runner (Tables III, IV, V, VI, VII)
├── run_ablations.py            # Systematic component ablation runner (Table IX)
├── generate_charts.py          # Publication-grade figures generator (Figures 1-4)
├── generate_dashboard.py       # Standalone HTML dashboard builder
├── data/
│   ├── sop/                    # 4 Authentic ICICoS SOP Markdown manuals
│   └── bpmn/                   # 3 OMG ISO/IEC 19510 XML BPMN process models
├── evaluation/
│   └── dataset/
│       └── icicos_process_qa.json  # Complete N = 338 ProcessQA benchmark corpus
├── src/
│   ├── config.py               # Hyperparameters, paths, quotas k(r), thresholds
│   ├── bpmn_parser.py          # XML parser, NetworkX DiGraph G, reachability DFS
│   ├── sop_indexer.py          # AST Markdown chunker, embedding cache, cosine search
│   ├── query_understanding.py  # QUA: Acronym expansion & canonical label grounding
│   ├── adaptive_router.py      # Softmax 3-way intent router & dynamic quota engine
│   ├── context_fusion.py       # Phase 4 cross-modal fusion & gateway guard injection
│   ├── generator.py            # Grounded generator & structure-aware synthesis
│   ├── critic_verification.py  # Adversarial Critic audit vector Ψ & self-repair
│   ├── pipeline.py             # End-to-end 6-phase pipeline coordinator
│   ├── baselines.py            # 4 Comparative baselines (Zero-shot, Vector, Graph, Static)
│   ├── evaluator.py            # Level 1 and Level 2 diagnostic metric calculators
│   └── statistical_tests.py    # Wilcoxon, Paired t-test, 10,000 Bootstrap 95% CIs
└── results/                    # Generated benchmark tables, CSVs, and PNG figures
```

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/reon01425-glitch/adaptive-agentic-hybrid-rag-bpmn.git
cd adaptive-agentic-hybrid-rag-bpmn
```

### 2. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure API Key
Copy the template and configure your Google GenAI API key:
```bash
cp .env.example .env
# Edit .env and insert your GOOGLE_API_KEY
```

---

## 💻 Running the System

### 1. Launch the Interactive Web Studio (Streamlit)
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser to interact with the multi-agent pipeline, test custom queries, inspect the 6-phase runtime trace, and compare architectures side-by-side.

### 2. Run the Full Benchmark Suite (Tables III–VII)
```bash
python run_experiments.py
```
Executes all 5 competing architectures across the stratified testbed ($N = 20$), computes Level 1 and Level 2 diagnostic metrics, executes 10,000-resample bootstrap statistical tests, and exports CSV results to `results/`.

### 3. Run Systematic Component Ablations (Table IX)
```bash
python run_ablations.py
```
Evaluates the impact of removing the Adaptive Router, BPMN Graph Retriever, and Adversarial Critic.

### 4. Generate Publication Figures (300 DPI)
```bash
python generate_charts.py
```
Exports `fig1_benchmark_faithfulness_hallucination.png`, `fig2_bpmn_diagnostic_comparison.png`, `fig3_latency_decomposition.png`, and `fig4_router_token_efficiency.png` to `results/`.

---

## 📊 Benchmark Summary Tables

### Table III: Primary Benchmark Comparison ($N = 20$)
| Architecture | Faithfulness ↑ | Answer Relevancy ↑ | Context Precision ↑ | Context Recall ↑ | BERTScore (F1) ↑ | Hallucination Rate ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. LLM Only (Zero-shot) | N/A* | 0.460 | N/A* | N/A* | 0.289 | 100.0% |
| 2. Traditional Vector RAG | 0.531 | 0.570 | 0.758 | 0.983 | 0.542 | 46.9% |
| 3. Graph RAG (BPMN Subgraph) | 0.332 | 0.571 | 0.650 | 0.762 | 0.547 | 66.8% |
| 4. Static Hybrid RAG | 0.597 | 0.568 | 0.770 | 0.983 | 0.543 | 40.3% |
| **5. Proposed Agentic Hybrid RAG** | **0.599** | **0.570** | **0.830** | **1.000** | **0.530** | **40.1%** |

### Table V: Procedural BPMN Reasoning Diagnostics
| Architecture | Process Path Acc. (PPA) ↑ | Gateway Decision Acc. (GDA) ↑ | Actor Attribution Acc. (AAA) ↑ |
| :--- | :---: | :---: | :---: |
| 1. LLM Only (Zero-shot) | 0.138 | 0.438 | 0.000 |
| 2. Traditional Vector RAG | 0.625 | 0.738 | 0.871 |
| 3. Graph RAG (BPMN Subgraph) | 0.250 | 0.887 | 0.821 |
| 4. Static Hybrid RAG | 0.725 | 0.838 | 0.871 |
| **5. Proposed Agentic Hybrid RAG** | **0.875** | **0.988** | **1.000** |

---

## 📜 License & Citation

Distributed under the MIT License. See `LICENSE` for details.

```bibtex
@article{lintang2026adaptive,
  title={Adaptive Agentic Hybrid RAG with BPMN-Grounded Graph Reasoning for Explainable Enterprise Process Question Answering},
  author={Lintang, et al.},
  journal={Draft Preprint},
  year={2026}
}
```
