# pyPochodnia

Biblioteka do tworzenia grafów obliczeniowych z automatycznym różniczkowaniem (autograd) dla sieci neuronowych MLP w czystym NumPy.

## Cel projektu

pyPochodnia to minimalistyczna implementacja frameworka do deep learningu, podobna do PyTorch, ale napisana od podstaw w Pythonie i NumPy. Projekt umożliwia:

- Tworzenie grafów obliczeniowych
- Automatyczne różniczkowanie (backward propagation)
- Budowanie i trenowanie sieci MLP
- Edukację na temat jak działają frameworki deep learningowe

## Struktura projektu

```
pyPochodnia/
├── app/
│   ├── node/                    # Podstawowe węzły grafu
│   │   ├── node.py             # Klasa bazowa Node
│   │   ├── constant.py         # Węzeł stałej
│   │   ├── variable.py         # Węzeł zmiennej
│   │   └── operations/         # Operacje
│   │       ├── arithmetic/     # Operacje arytmetyczne
│   │       │   ├── add.py      # Dodawanie
│   │       │   ├── subtract.py # Odejmowanie
│   │       │   ├── multiply.py # Mnożenie
│   │       │   ├── divide.py   # Dzielenie
│   │       │   ├── power.py    # Potęgowanie
│   │       │   └── matmul.py   # Mnożenie macierzowe
│   │       ├── activation.py   # Funkcje aktywacji (ReLU, Sigmoid, Tanh, Softmax)
│   │       └── loss.py         # Funkcje straty (MSE, CrossEntropy)
│   ├── layers/                  # Warstwy sieci
│   │   └── dense.py            # Warstwa fully-connected
│   ├── models/                  # Modele
│   │   └── mlp.py              # Multi-Layer Perceptron
│   └── optimizers/              # Optymalizatory
│       └── optimizer.py        # SGD, Adam
├── examples/                    # Przykłady użycia
│   ├── example_mlp_xor.py      # Problem XOR
│   └── example_mlp_regression.py # Regresja liniowa
├── tests/                       # Testy jednostkowe
│   ├── test_nodes.py           # Testy węzłów
│   └── test_arithmetic_operations.py # Testy operacji
└── main.py                      # Główny plik
```

## 🚀 Instalacja

```bash
# Klonowanie repozytorium
git clone https://github.com/yourusername/pyPochodnia.git
cd pyPochodnia

# Instalacja zależności
pip install numpy pandas pydantic pytest
```

## Przykłady użycia

### Podstawowe operacje

```python
import numpy as np
from app.node import Variable, Constant
from app.node.operations.arithmetic import Add, Multiply

# Tworzenie zmiennych
x = Variable(value=np.array([1.0, 2.0, 3.0]), requires_grad=True)
w = Variable(value=np.array([2.0, 2.0, 2.0]), requires_grad=True)
b = Constant(value=np.array([1.0, 1.0, 1.0]))

# Budowanie grafu: y = x * w + b
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

### Uruchomienie przykładów

```bash
# Problem XOR
python examples/example_mlp_xor.py

# Regresja liniowa
python examples/example_mlp_regression.py
```

## Dostępne komponenty

### Węzły (Nodes)
- **Variable**: Węzeł przechowujący dane z opcjonalnym gradientem
- **Constant**: Węzeł stałej (bez gradientu)

### Operacje arytmetyczne
- **Add**: Dodawanie (a + b)
- **Subtract**: Odejmowanie (a - b)
- **Multiply**: Mnożenie element-wise (a * b)
- **Divide**: Dzielenie (a / b)
- **Power**: Potęgowanie (a^b)
- **MatMul**: Mnożenie macierzowe (a @ b)

### Funkcje aktywacji
- **ReLU**: f(x) = max(0, x)
- **Sigmoid**: f(x) = 1 / (1 + exp(-x))
- **Tanh**: f(x) = tanh(x)
- **Softmax**: Normalizacja prawdopodobieństw

### Funkcje straty
- **MSELoss**: Mean Squared Error
- **CrossEntropyLoss**: Cross Entropy dla klasyfikacji wieloklasowej
- **BinaryCrossEntropyLoss**: Binary Cross Entropy dla klasyfikacji binarnej

### Warstwy
- **Dense**: Warstwa fully-connected (linear) z opcjonalnym bias

### Modele
- **MLP**: Multi-Layer Perceptron z konfigurowalnymi warstwami i aktywacjami

### Optymalizatory
- **SGD**: Stochastic Gradient Descent (z opcjonalnym momentum)
- **Adam**: Adaptive Moment Estimation

