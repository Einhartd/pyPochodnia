from typing import List
from app.node import Variable, Node


class Sequential:

    def __init__(self, layers: List = None, name: str = "Sequential"):

        self.name = name
        self.layers = layers if layers is not None else []

    def add(self, layer):

        self.layers.append(layer)

    def forward(self, x: Node) -> Node:

        output = x

        for layer in self.layers:
            output = layer(output)

        return output

    def __call__(self, x: Node) -> Node:

        return self.forward(x)

    def parameters(self) -> List[Node]:

        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self):

        for layer in self.layers:
            layer.zero_grad()

    def summary(self):

        print(f"\n{'='*70}")
        print(f"Model: {self.name}")
        print(f"{'='*70}")
        print(f"{'Layer (type)':<30} {'Output Shape':<20} {'Params':<15}")
        print(f"{'-'*70}")

        total_params = 0

        for i, layer in enumerate(self.layers):
            layer_type = layer.__class__.__name__
            layer_name = getattr(layer, 'name', f'layer_{i}')

            # Count parameters
            layer_params = 0
            if hasattr(layer, 'parameters'):
                for param in layer.parameters():
                    if param.value is not None:
                        layer_params += param.value.size

            # Get output shape (if available)
            if hasattr(layer, 'output_size'):
               output_shape = str(layer.output_size)
            else:
                output_shape = "-"

            display_name = f"{layer_name} ({layer_type})"
            print(f"{display_name:<30} {output_shape:<20} {layer_params:<15}")
            total_params += layer_params

        print(f"{'='*70}")
        print(f"Total parameters: {total_params}")
        print(f"{'='*70}\n")

