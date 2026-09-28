"""
Configuration settings for Adaptive Agentic Hybrid RAG with BPMN Grounding.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Model parameters
LLM_MODEL = "gemini-3.1-flash-lite"
FALLBACK_MODELS = ["gemini-3.1-flash-lite", "gemini-3.8-flash", "gemini-flash-latest"]
EMBEDDING_MODEL = "gemini-embedding-001"
TEMPERATURE = 0.0
TOP_P = 1.0
RANDOM_SEED = 42

# Verification thresholds & constraints
CRITIC_THRESHOLD = 0.85  # tau = 0.85
MAX_SELF_REPAIR_ITERATIONS = 2  # N_max = 2

# Quota settings k(r) = <k_sop, k_bpmn>
QUOTAS = {
    "SOP": {"k_sop": 6, "k_bpmn": 2},
    "BPMN": {"k_sop": 3, "k_bpmn": 4},
    "Hybrid": {"k_sop": 5, "k_bpmn": 3},
    "Static_Hybrid": {"k_sop": 5, "k_bpmn": 3},
    "Vector_RAG": {"k_sop": 6, "k_bpmn": 0},
    "Graph_RAG": {"k_sop": 0, "k_bpmn": 4},
    "LLM_Only": {"k_sop": 0, "k_bpmn": 0},
}

# Paths
DATA_DIR = BASE_DIR / "data"
SOP_DIR = DATA_DIR / "sop"
BPMN_DIR = DATA_DIR / "bpmn"
DATASET_PATH = BASE_DIR / "evaluation" / "dataset" / "icicos_process_qa.json"
RESULTS_DIR = BASE_DIR / "results"
CHROMA_PERSIST_DIR = BASE_DIR / "results" / "chroma_db"
CACHE_DIR = BASE_DIR / "results" / "cache"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
