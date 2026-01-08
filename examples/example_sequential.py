import numpy as np
from app.node import Variable, Constant
from app.models import Sequential
from app.layers import Dense, Activation
from app.node.operations.loss import MSELoss
from app.optimizers import Adam
from app.utils import accuracy, classification_report

def main():
    print("Example: MLP in XOR problem")

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

    print("\nTraining dataset (XOR):")
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
    print("\nTrain start...\n")
    epochs = 5000
    print_every = 500

    # Train on entire batch
    x_var = Constant(X, name="x_batch")
    y_target = Constant(y, name="y_batch")

    output = model(x_var)

    for epoch in range(epochs):
        total_loss = 0.0

        # Compute loss
        loss = MSELoss(output, y_target, name="loss")
        loss_value = loss.forward()
        total_loss = loss_value

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()
        optimizer.zero_grad()

        # Print progress with metrics
        if (epoch + 1) % print_every == 0:
            # Get current predictions
            current_pred = output.value
            train_acc = accuracy(y, current_pred, threshold=0.5)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.6f}, Accuracy: {train_acc:.4f}")

    # Test the trained model
    print("\n" + "="*70)
    print("Test of trained model:")
    print("="*70)

    predictions = []
    for i in range(len(X)):
        x_var = Variable(X[i:i+1], name=f"test_x_{i}")
        output = model(x_var)
        pred = output.forward()
        predictions.append(pred[0][0])

        print(f"X: {X[i]} -> Y_pred: {pred[0][0]:.4f}, Y: {y[i][0]}")

    # Calculate and display final metrics
    predictions = np.array(predictions).reshape(-1, 1)
    print("\n" + classification_report(y, predictions, threshold=0.5))
    print()


if __name__ == "__main__":
    main()

