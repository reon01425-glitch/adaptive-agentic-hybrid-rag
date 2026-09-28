"""
Cross-Modal Context Fusion Layer (Phase 4 in Paper).
Performs:
1. Node-to-Chunk Projection: maps unstructured SOP passages to relevant BPMN task nodes.
2. Decision Gateway Guard Injection: formats explicit boolean condition guards C(e).
3. Institutional Domain Boundary Enforcement: purges non-ICICoS knowledge chunks.
Outputs: Unified Fused Context C_fused.
"""

from typing import List, Dict, Any, Optional

class CrossModalContextFusion:
    """Fuses retrieved SOP textual passages and BPMN graph topology into C_fused."""

    def __init__(self, target_domain: str = "ICICoS"):
        self.target_domain = target_domain

    def enforce_domain_boundary(self, sop_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Strictly purges any chunks outside the verified enterprise domain."""
        valid = []
        for c in sop_chunks:
            if c.get("domain", "") == self.target_domain:
                valid.append(c)
        return valid

    def project_nodes_to_chunks(
        self, sop_chunks: List[Dict[str, Any]], bpmn_nodes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Associates SOP clauses to corresponding BPMN task nodes via keyword/role matching."""
        enriched_chunks = []
        for c in sop_chunks:
            c_copy = dict(c)
            c_text = (c.get("title", "") + " " + c.get("content", "")).lower()
            associated_tasks = []
            for n in bpmn_nodes:
                n_label = n.get("label", "").lower()
                n_role = n.get("role", "").lower()
                if any(w in c_text for w in n_label.split() if len(w) > 4):
                    associated_tasks.append(f"{n.get('label')} [{n.get('role')}]")
            c_copy["projected_bpmn_tasks"] = associated_tasks[:2]
            enriched_chunks.append(c_copy)
        return enriched_chunks

    def fuse(
        self,
        sop_chunks: List[Dict[str, Any]],
        bpmn_graph_text: str,
        bpmn_nodes: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Synthesizes unified fused context C_fused.
        """
        # 1. Domain boundary enforcement
        valid_chunks = self.enforce_domain_boundary(sop_chunks)

        # 2. Node-to-chunk projection
        if bpmn_nodes:
            valid_chunks = self.project_nodes_to_chunks(valid_chunks, bpmn_nodes)

        sections = []
        sections.append("###################################################################")
        sections.append(f"UNIFIED ENTERPRISE CONTEXT [DOMAIN: {self.target_domain}]")
        sections.append("###################################################################")

        # 3. Inject Structured Process Graph Topology with Gateway Guards
        if bpmn_graph_text.strip():
            sections.append("\n[STRUCTURED PROCESS KNOWLEDGE SPACE (K_bpmn)]")
            sections.append(bpmn_graph_text.strip())

        # 4. Inject Hierarchical Textual SOP Passages
        if valid_chunks:
            sections.append("\n[TEXTUAL SOP KNOWLEDGE SPACE (K_sop)]")
            for idx, c in enumerate(valid_chunks, 1):
                proj_info = ""
                if c.get("projected_bpmn_tasks"):
                    proj_info = f" | [Linked Workflow Nodes: {', '.join(c['projected_bpmn_tasks'])}]"
                sections.append(f"--- SOP Section {idx}: {c.get('path_hierarchy', '')} > {c.get('title', '')}{proj_info} ---")
                sections.append(c.get("content", "").strip())

        sections.append("###################################################################")
        return "\n".join(sections)
