"""
SOP Textual Knowledge Space (K_sop) Indexer.
Performs Abstract Syntax Tree (AST) section-hierarchical chunking:
c_i = <Path(H1 > H2 > H3), Body_i, Meta_i>
Generates embeddings and provides persistent cosine vector retrieval.
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    from google import genai
except ImportError:
    genai = None

from src.config import GOOGLE_API_KEY, EMBEDDING_MODEL, SOP_DIR, CACHE_DIR

class SOPChunk:
    def __init__(self, chunk_id: str, path_hierarchy: str, title: str, content: str, doc_name: str, domain: str = "ICICoS"):
        self.chunk_id = chunk_id
        self.path_hierarchy = path_hierarchy
        self.title = title
        self.content = content
        self.doc_name = doc_name
        self.domain = domain
        self.embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "path_hierarchy": self.path_hierarchy,
            "title": self.title,
            "content": self.content,
            "doc_name": self.doc_name,
            "domain": self.domain
        }

class SOPIndexer:
    """Indexes SOP markdown documents into hierarchical AST chunks and vector embeddings."""

    def __init__(self, sop_dir: Path = SOP_DIR):
        self.sop_dir = sop_dir
        self.chunks: List[SOPChunk] = []
        self.cache_file = CACHE_DIR / "sop_embeddings_cache.json"
        self.embeddings_cache: Dict[str, List[float]] = {}
        self.client = None
        if GOOGLE_API_KEY and genai:
            try:
                self.client = genai.Client(api_key=GOOGLE_API_KEY)
            except Exception as e:
                print(f"Warning: Failed to initialize Google GenAI Client: {e}")

        self._load_cache()
        self.ingest_all_sops()

    def _load_cache(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.embeddings_cache = json.load(f)
            except Exception as e:
                self.embeddings_cache = {}

    def _save_cache(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.embeddings_cache, f)
        except Exception as e:
            pass

    def parse_markdown_ast(self, file_path: Path) -> List[SOPChunk]:
        """Parses markdown into hierarchical AST chunks preserving H1 > H2 > H3 paths."""
        text = file_path.read_text(encoding="utf-8")
        lines = text.split("\n")
        
        doc_h1 = file_path.stem.replace("_", " ").title()
        current_h1 = doc_h1
        current_h2 = ""
        current_h3 = ""
        
        chunks = []
        current_buffer = []
        chunk_idx = 0

        for line in lines:
            line_strip = line.strip()
            if line_strip.startswith("# "):
                if current_buffer:
                    c_text = "\n".join(current_buffer).strip()
                    if len(c_text) > 30:
                        hierarchy = f"{current_h1} > {current_h2}".strip(" >")
                        chunks.append(SOPChunk(
                            chunk_id=f"{file_path.stem}_{chunk_idx}",
                            path_hierarchy=hierarchy,
                            title=current_h2 or current_h1,
                            content=c_text,
                            doc_name=file_path.name,
                            domain="ICICoS"
                        ))
                        chunk_idx += 1
                    current_buffer = []
                current_h1 = line_strip[2:].strip()
                current_h2 = ""
                current_h3 = ""
            elif line_strip.startswith("## "):
                if current_buffer:
                    c_text = "\n".join(current_buffer).strip()
                    if len(c_text) > 30:
                        hierarchy = f"{current_h1} > {current_h2}".strip(" >")
                        chunks.append(SOPChunk(
                            chunk_id=f"{file_path.stem}_{chunk_idx}",
                            path_hierarchy=hierarchy,
                            title=current_h2 or current_h1,
                            content=c_text,
                            doc_name=file_path.name,
                            domain="ICICoS"
                        ))
                        chunk_idx += 1
                    current_buffer = []
                current_h2 = line_strip[3:].strip()
                current_h3 = ""
            elif line_strip.startswith("### "):
                if current_buffer:
                    c_text = "\n".join(current_buffer).strip()
                    if len(c_text) > 30:
                        hierarchy = f"{current_h1} > {current_h2} > {current_h3}".strip(" >")
                        chunks.append(SOPChunk(
                            chunk_id=f"{file_path.stem}_{chunk_idx}",
                            path_hierarchy=hierarchy,
                            title=current_h3 or current_h2,
                            content=c_text,
                            doc_name=file_path.name,
                            domain="ICICoS"
                        ))
                        chunk_idx += 1
                    current_buffer = []
                current_h3 = line_strip[4:].strip()
            else:
                current_buffer.append(line)

        if current_buffer:
            c_text = "\n".join(current_buffer).strip()
            if len(c_text) > 30:
                hierarchy = f"{current_h1} > {current_h2} > {current_h3}".strip(" >")
                chunks.append(SOPChunk(
                    chunk_id=f"{file_path.stem}_{chunk_idx}",
                    path_hierarchy=hierarchy,
                    title=current_h3 or current_h2 or current_h1,
                    content=c_text,
                    doc_name=file_path.name,
                    domain="ICICoS"
                ))

        return chunks

    def ingest_all_sops(self):
        """Ingests all SOP documents in SOP_DIR."""
        self.chunks = []
        for p in self.sop_dir.glob("*.md"):
            doc_chunks = self.parse_markdown_ast(p)
            self.chunks.extend(doc_chunks)

        # Generate / load embeddings
        for c in self.chunks:
            full_text = f"Path: {c.path_hierarchy}\nSection: {c.title}\n{c.content}"
            c.embedding = self.get_embedding(full_text)

    def get_embedding(self, text: str) -> List[float]:
        """Fetches embedding with persistent local cache."""
        thash = hashlib.md5(text.encode("utf-8")).hexdigest()
        if thash in self.embeddings_cache:
            return self.embeddings_cache[thash]

        if self.client:
            try:
                res = self.client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=text[:2000]
                )
                emb = res.embeddings[0].values
                self.embeddings_cache[thash] = emb
                self._save_cache()
                return emb
            except Exception as e:
                pass

        # Deterministic pseudo-semantic fallback embedding if API fails
        # Generates normalized bag-of-words / character n-gram hashing
        dim = 256
        vec = np.zeros(dim, dtype=np.float32)
        words = re.findall(r"\w+", text.lower())
        for w in words:
            h = int(hashlib.sha256(w.encode("utf-8")).hexdigest(), 16) % dim
            vec[h] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        emb = vec.tolist()
        self.embeddings_cache[thash] = emb
        return emb

    def retrieve(self, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        """Retrieves top-k chunks using cosine similarity."""
        if not self.chunks or top_k <= 0:
            return []

        q_emb = np.array(self.get_embedding(query), dtype=np.float32)
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb = q_emb / q_norm

        scores = []
        q_tokens = set(re.findall(r"\w+", query.lower()))

        for c in self.chunks:
            c_emb = np.array(c.embedding, dtype=np.float32)
            c_norm = np.linalg.norm(c_emb)
            if c_norm > 0:
                c_emb = c_emb / c_norm
            cos_sim = float(np.dot(q_emb, c_emb))

            # Lexical keyword match boost
            c_text = (c.path_hierarchy + " " + c.title + " " + c.content).lower()
            overlap = sum(1 for tok in q_tokens if tok in c_text and len(tok) > 2)
            combined_score = 0.7 * cos_sim + 0.3 * min(overlap / max(1, len(q_tokens)), 1.0)

            scores.append((combined_score, c))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, chunk in scores[:top_k]:
            c_dict = chunk.to_dict()
            c_dict["similarity_score"] = round(score, 4)
            results.append(c_dict)

        return results
