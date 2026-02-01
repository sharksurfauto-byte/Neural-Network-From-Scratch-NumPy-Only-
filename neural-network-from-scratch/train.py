
import numpy as np
from data.mnist_loader import load_and_preprocess_mnist
from optimizers.sgd import gradient_descent
from layers.linear import initialize_parameters, forward_propagation # Only needed for initial param sizes
from utils.metrics import get_predictions, get_accuracy

def train_model():
    print("Loading and preprocessing MNIST data...")
    X_train, y_train, X_test, y_test = load_and_preprocess_mnist()

    input_size = X_train.shape[0] # 784
    hidden_size = 10 # Example hidden layer size
    output_size = 10 # 10 classes for MNIST

    # Note: initialize_parameters is now called inside gradient_descent
    # We only need the shapes here for consistency in calling gradient_descent

    learning_rate = 0.5
    iterations = 2000

    print(f"
Starting training with learning rate={learning_rate}, iterations={iterations}...")
    W1, B1, W2, B2 = gradient_descent(X_train, y_train, learning_rate, iterations, input_size, hidden_size, output_size)
    print("
Training complete.")

    print("
Evaluating on test set...")
    # Forward propagate with the learned parameters on the test set
    Z1_test, A1_test, Z2_test, A2_test = forward_propagation(W1, B1, W2, B2, X_test)
    test_predictions = get_predictions(A2_test)
    test_accuracy = get_accuracy(test_predictions, y_test)
    print(f"Test Accuracy = {test_accuracy}")

if __name__ == "__main__":
    train_model()
