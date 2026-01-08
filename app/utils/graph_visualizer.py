from app.node import Variable, Node

def print_computational_graph(node: Node, visited=None, indent=0):

    if visited is None:
        visited = set()

    if id(node) in visited:
        print(" " * indent + f"↻ {node.name} (already visited)")
        return

    visited.add(id(node))

    val = f"node value: {node.value}" if node.value is not None else ""
    val_shape = f"shape: {node.value.shape}" if node.value is not None else ""
    grad = f"grad: {node.grad}" if node.grad is not None else ""
    grad_shape = f"shape: {node.grad.shape}" if node.grad is not None else ""
    print(" " * indent + f"├─{node.name}: (values: {val_shape}), (grad: {grad_shape})")
    for parent in node.parents:
        print_computational_graph(parent, visited, indent + 1)