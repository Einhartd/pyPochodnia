import numpy as np
from app.node import Variable, Constant
from app.models import Sequential
from app.layers import Dense, Activation
from app.node.operations.loss import MSELoss
from app.optimizers import Adam


def main():
    print("Przykład: Trening MLP na problemie XOR (Sequential API)")

    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=np.float32)

    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ], dtype=np.float32)

    print("\nDane treningowe (XOR):")
    for i in range(len(X)):
        print(f"  X: {X[i]} -> y: {y[i]}")

    model = Sequential([
        Dense(2, 8, weight_init='xavier', name='hidden_layer'),
        Activation('tanh', name='tanh_activation'),
        Dense(8, 1, weight_init='xavier', name='output_layer'),
        Activation('sigmoid', name='sigmoid_activation')
    ], name='xor_Sequential')

    model.summary()

    optimizer = Adam(model.parameters(), learning_rate=0.01)

    # Training loop
    print("\nRozpoczynanie treningu...\n")
    epochs = 5000
    print_every = 500

    for epoch in range(epochs):
        total_loss = 0.0

        # Train on entire batch
        x_var = Variable(X, name="x_batch")
        y_target = Constant(y, name="y_batch")

        # Forward pass
        output = model(x_var)

        # Compute loss
        loss = MSELoss(output, y_target, name="loss")
        loss_value = loss.forward()
        total_loss = loss_value

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()
        optimizer.zero_grad()

        # Print progress
        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.6f}")

    # Test the trained model
    print("\n" + "="*70)
    print("Testowanie wytrenowanego modelu:")
    print("="*70)

    for i in range(len(X)):
        x_var = Variable(X[i:i+1], name=f"test_x_{i}")
        output = model(x_var)
        pred = output.forward()

        print(f"X: {X[i]} -> Predykcja: {pred[0][0]:.4f}, Oczekiwane: {y[i][0]}")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()

