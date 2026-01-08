# pyPochodnia

A minimal computational graph library with automatic differentiation (autograd) for neural networks, built from scratch in pure NumPy.

## Overview

pyPochodnia is a PyTorch-like deep learning framework implemented from scratch in Python and NumPy. Features:

- Computational graph construction
- Automatic differentiation (backpropagation)
- MLP neural networks
- Educational tool for understanding deep learning frameworks

## Installation

```bash
git clone https://github.com/yourusername/pyPochodnia.git
cd pyPochodnia
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
from app.node import Variable, Constant
from app.node.operations.arithmetic import Add, Multiply

# Create variables
x = Variable(value=np.array([1.0, 2.0, 3.0]), requires_grad=True)
w = Variable(value=np.array([2.0, 2.0, 2.0]), requires_grad=True)
b = Constant(value=np.array([1.0, 1.0, 1.0]))

# Build graph: y = x * w + b
mul_node = Multiply(x, w)
result = Add(mul_node, b)

# Forward pass
output = result.forward()
print(f"Output: {output}")

# Backward pass
result.backward()
print(f"Gradient x: {x.grad}")
print(f"Gradient w: {w.grad}")
```

## Components

### Nodes
- **Variable** - Trainable parameters with gradients
- **Constant** - Fixed values (no gradients)

### Operations
- **Arithmetic**: Add, Subtract, Multiply, Power, MatMul
- **Activations**: ReLU, Sigmoid, Tanh, Softmax
- **Loss Functions**: MSELoss, CrossEntropyLoss

### Layers & Models
- **Dense** - Fully-connected layer with optional bias
- **Sequential** - Stack layers into a model

### Optimizers
- **SGD** - Stochastic Gradient Descent (with momentum)
- **Adam** - Adaptive Moment Estimation

