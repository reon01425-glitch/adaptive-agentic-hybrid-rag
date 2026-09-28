"""
BPMN 2.0 XML Parser and Attributed Directed Reasoning Graph G = (V, E, L, S, C).
Transforms OMG ISO/IEC 19510 BPMN 2.0 specifications into NetworkX DiGraphs.
Preserves non-symmetric directional sequence flows, swimlanes, and boolean gateway guards.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
import networkx as nx

class BPMNProcessGraph:
    """
    Attributed Process Reasoning Graph G = (V, E, L, S, C)
    V = V_task U V_event U V_gateway
    E = Directed sequence transitions
    L = Canonical natural language labels
    S = Swimlane role allocations
    C = Boolean condition guards on gateway sequence edges
    """

    def __init__(self, bpmn_path: Optional[Path] = None):
        self.graph = nx.DiGraph()
        self.node_metadata: Dict[str, Dict[str, Any]] = {}
        self.edge_metadata: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.process_name: str = ""
        self.namespaces = {
            "bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL",
            "xsi": "http://www.w3.org/2001/XMLSchema-instance"
        }
        if bpmn_path:
            self.load_bpmn_xml(bpmn_path)

    def load_bpmn_xml(self, xml_file: Path) -> nx.DiGraph:
        """Parses OMG BPMN 2.0 XML file into NetworkX DiGraph."""
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Find process
        process_elem = root.find(".//bpmn:process", self.namespaces)
        if process_elem is None:
            process_elem = root.find(".//process")
        
        self.process_name = xml_file.stem
        if process_elem is not None and "name" in process_elem.attrib:
            self.process_name = process_elem.attrib["name"]

        # 1. Parse Swimlanes (Roles S(v))
        lane_map: Dict[str, str] = {}  # node_id -> role
        for lane in root.findall(".//bpmn:lane", self.namespaces):
            lane_name = lane.attrib.get("name", "Default Role")
            for flow_node in lane.findall("bpmn:flowNodeRef", self.namespaces):
                node_id = flow_node.text.strip() if flow_node.text else ""
                if node_id:
                    lane_map[node_id] = lane_name

        # 2. Parse Nodes (Activities, Gateways, Events)
        # Tasks / Activities
        for tag in ["task", "userTask", "serviceTask", "scriptTask", "manualTask"]:
            for elem in root.findall(f".//bpmn:{tag}", self.namespaces):
                nid = elem.attrib["id"]
                label = elem.attrib.get("name", nid)
                role = lane_map.get(nid, "System / Committee")
                self.graph.add_node(nid, type="task", label=label, role=role, process=self.process_name)
                self.node_metadata[nid] = {
                    "id": nid,
                    "type": "task",
                    "label": label,
                    "role": role,
                    "process": self.process_name
                }

        # Gateways (Exclusive XOR, Parallel AND, Inclusive OR)
        for tag in ["exclusiveGateway", "parallelGateway", "inclusiveGateway"]:
            for elem in root.findall(f".//bpmn:{tag}", self.namespaces):
                nid = elem.attrib["id"]
                label = elem.attrib.get("name", nid)
                gw_type = "XOR" if "exclusive" in tag.lower() else ("AND" if "parallel" in tag.lower() else "OR")
                role = lane_map.get(nid, "Decision Engine")
                self.graph.add_node(nid, type="gateway", gateway_type=gw_type, label=label, role=role, process=self.process_name)
                self.node_metadata[nid] = {
                    "id": nid,
                    "type": "gateway",
                    "gateway_type": gw_type,
                    "label": label,
                    "role": role,
                    "process": self.process_name
                }

        # Events (Start, End, Intermediate)
        for tag in ["startEvent", "endEvent", "intermediateCatchEvent", "intermediateThrowEvent"]:
            for elem in root.findall(f".//bpmn:{tag}", self.namespaces):
                nid = elem.attrib["id"]
                label = elem.attrib.get("name", nid)
                event_type = "start" if "start" in tag.lower() else ("end" if "end" in tag.lower() else "intermediate")
                role = lane_map.get(nid, "Lifecycle Event")
                self.graph.add_node(nid, type="event", event_type=event_type, label=label, role=role, process=self.process_name)
                self.node_metadata[nid] = {
                    "id": nid,
                    "type": "event",
                    "event_type": event_type,
                    "label": label,
                    "role": role,
                    "process": self.process_name
                }

        # 3. Parse Directed Sequence Flows & Gateway Boolean Guards C(e)
        for flow in root.findall(".//bpmn:sequenceFlow", self.namespaces):
            fid = flow.attrib.get("id", "")
            src = flow.attrib.get("sourceRef", "")
            tgt = flow.attrib.get("targetRef", "")
            fname = flow.attrib.get("name", "")

            # Condition expression
            cond_elem = flow.find("bpmn:conditionExpression", self.namespaces)
            condition = cond_elem.text.strip() if cond_elem is not None and cond_elem.text else fname

            if src in self.graph and tgt in self.graph:
                self.graph.add_edge(src, tgt, id=fid, condition=condition, label=fname)
                self.edge_metadata[(src, tgt)] = {
                    "flow_id": fid,
                    "condition": condition,
                    "label": fname
                }

        return self.graph

    def is_reachable(self, u: str, v: str) -> bool:
        """Verifies topological reachability u ~>_G v via DFS."""
        if u not in self.graph or v not in self.graph:
            return False
        return nx.has_path(self.graph, u, v)

    def get_path(self, u: str, v: str) -> List[str]:
        """Returns shortest directed procedural path between u and v."""
        if self.is_reachable(u, v):
            return nx.shortest_path(self.graph, u, v)
        return []

    def get_successors(self, node_id: str) -> List[Dict[str, Any]]:
        """Returns immediate successor steps with transition guards."""
        if node_id not in self.graph:
            return []
        successors = []
        for succ in self.graph.successors(node_id):
            edge_data = self.graph.get_edge_data(node_id, succ)
            succ_meta = dict(self.node_metadata.get(succ, {}))
            succ_meta["guard"] = edge_data.get("condition", "")
            successors.append(succ_meta)
        return successors

    def get_predecessors(self, node_id: str) -> List[Dict[str, Any]]:
        """Returns immediate prerequisite steps."""
        if node_id not in self.graph:
            return []
        preds = []
        for pred in self.graph.predecessors(node_id):
            pred_meta = dict(self.node_metadata.get(pred, {}))
            preds.append(pred_meta)
        return preds

    def get_subgraph_around_nodes(self, node_ids: List[str], hops: int = 2) -> Dict[str, Any]:
        """
        Extracts k-hop neighborhood around target nodes.
        Preserves directed edges, swimlanes, and boolean guards.
        """
        active_nodes: Set[str] = set()
        for nid in node_ids:
            if nid in self.graph:
                active_nodes.add(nid)
                # Expand successors and predecessors up to hops
                current = {nid}
                for _ in range(hops):
                    next_nodes = set()
                    for n in current:
                        next_nodes.update(self.graph.successors(n))
                        next_nodes.update(self.graph.predecessors(n))
                    active_nodes.update(next_nodes)
                    current = next_nodes

        subg = self.graph.subgraph(active_nodes)
        
        nodes_list = []
        for n in subg.nodes():
            meta = self.node_metadata.get(n, {})
            nodes_list.append(meta)

        edges_list = []
        for u, v in subg.edges():
            edata = subg.get_edge_data(u, v)
            u_label = self.node_metadata.get(u, {}).get("label", u)
            v_label = self.node_metadata.get(v, {}).get("label", v)
            u_role = self.node_metadata.get(u, {}).get("role", "")
            v_role = self.node_metadata.get(v, {}).get("role", "")
            cond = edata.get("condition", "")
            edges_list.append({
                "from_id": u,
                "from_label": u_label,
                "from_role": u_role,
                "to_id": v,
                "to_label": v_label,
                "to_role": v_role,
                "guard": cond
            })

        return {
            "nodes": nodes_list,
            "edges": edges_list,
            "total_nodes": len(nodes_list),
            "total_edges": len(edges_list)
        }

    def format_subgraph_text(self, subgraph_data: Dict[str, Any]) -> str:
        """Formats graph topology into structured prompt text for generation."""
        lines = ["=== BPMN PROCESS WORKFLOW TOPOLOGY ==="]
        lines.append("ROLES & SWIMLANES:")
        roles_seen = set()
        for n in subgraph_data.get("nodes", []):
            role = n.get("role")
            if role and role not in roles_seen:
                roles_seen.add(role)
                lines.append(f"• Swimlane Role: {role}")

        lines.append("\nCHRONOLOGICAL SEQUENCE & TRANSITION GUARDS:")
        for e in subgraph_data.get("edges", []):
            guard_str = f" [GUARD: {e['guard']}]" if e.get("guard") else ""
            lines.append(
                f"• ({e['from_role']}) '{e['from_label']}' ──{guard_str}──> ({e['to_role']}) '{e['to_label']}'"
            )

        lines.append("=======================================")
        return "\n".join(lines)


class MultiBPMNRepository:
    """Manages all enterprise BPMN workflow models."""
    def __init__(self, bpmn_dir: Path):
        self.bpmn_dir = bpmn_dir
        self.processes: Dict[str, BPMNProcessGraph] = {}
        self.load_all()

    def load_all(self):
        for bpmn_file in self.bpmn_dir.glob("*.bpmn"):
            p_graph = BPMNProcessGraph(bpmn_file)
            self.processes[bpmn_file.stem] = p_graph

    def get_all_nodes(self) -> List[Dict[str, Any]]:
        nodes = []
        for p_name, p_graph in self.processes.items():
            for nid, meta in p_graph.node_metadata.items():
                meta_copy = dict(meta)
                meta_copy["bpmn_model"] = p_name
                nodes.append(meta_copy)
        return nodes

    def query_subgraph(self, matched_node_ids: List[str], hops: int = 2) -> str:
        all_subgraphs = []
        for p_name, p_graph in self.processes.items():
            valid_ids = [nid for nid in matched_node_ids if nid in p_graph.graph]
            if valid_ids:
                sub_data = p_graph.get_subgraph_around_nodes(valid_ids, hops=hops)
                formatted = p_graph.format_subgraph_text(sub_data)
                all_subgraphs.append(f"Model: {p_name}\n" + formatted)
        
        if not all_subgraphs:
            # Fallback to returning primary registration workflow
            for p_name, p_graph in self.processes.items():
                if "reg" in p_name:
                    all_nodes = list(p_graph.graph.nodes())[:6]
                    sub_data = p_graph.get_subgraph_around_nodes(all_nodes, hops=1)
                    return p_graph.format_subgraph_text(sub_data)
            return ""

        return "\n\n".join(all_subgraphs)
