from typing import List, Dict, Optional
from app.node import Node


def collect_all_nodes(root: Node, visited: Optional[set] = None) -> List[Node]:

    if visited is None:
        visited = set()

    nodes = []

    if root in visited:
        return nodes

    visited.add(root)
    nodes.append(root)

    # Recursively collect parent nodes
    for parent in root.parents:
        if parent is not None:
            nodes.extend(collect_all_nodes(parent, visited))

    return nodes


def find_node_by_name(root: Node, name: str) -> Optional[Node]:

    all_nodes = collect_all_nodes(root)

    for node in all_nodes:
        if node.name == name:
            return node

    return None


def find_nodes_by_type(root: Node, node_type: str) -> List[Node]:

    all_nodes = collect_all_nodes(root)

    return [node for node in all_nodes if node.type == node_type]


def get_node_info(node: Node, detailed: bool = False) -> str:

    info = []
    info.append(f"Node: {node.name}")
    info.append(f"  Type: {node.type}")
    info.append(f"  ID: {node.id}")

    if node.value is not None:
        info.append(f"  Value shape: {node.value.shape}")
        info.append(f"  Value dtype: {node.value.dtype}")
        if detailed and node.value.size <= 20:
            info.append(f"  Value:\n{node.value}")
    else:
        info.append(f"  Value: None")

    if node.grad is not None:
        info.append(f"  Gradient shape: {node.grad.shape}")
        if detailed and node.grad.size <= 20:
            info.append(f"  Gradient:\n{node.grad}")
    else:
        info.append(f"  Gradient: None")

    info.append(f"  Parents: {len(node.parents)}")
    if detailed and node.parents:
        for i, parent in enumerate(node.parents):
            if parent is not None:
                info.append(f"    [{i}] {parent.name} (type: {parent.type})")

    return "\n".join(info)


def get_nodes_by_layer(root: Node, layer_prefix: str) -> Dict[str, Node]:

    all_nodes = collect_all_nodes(root)

    layer_nodes = {}
    for node in all_nodes:
        if node.name.startswith(layer_prefix):
            layer_nodes[node.name] = node

    return layer_nodes

