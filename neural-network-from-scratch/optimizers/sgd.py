
import numpy as np
from layers.linear import initialize_parameters, forward_propagation, backward_propagation, update_parameters
from utils.metrics import get_predictions, get_accuracy

def gradient_descent(X, Y, alpha, iterations, input_size, hidden_size, output_size):
  W1, B1, W2, B2 = initialize_parameters(input_size, hidden_size, output_size)
  m = X.shape[1] # Number of samples

  for i in range(iterations):
    Z1, A1, Z2, A2 = forward_propagation(W1, B1, W2, B2, X)
    dW1, dB1, dW2, dB2 = backward_propagation(W1, B1, W2, B2, Z1, A1, Z2, A2, X, Y, m)
    W1, B1, W2, B2 = update_parameters(W1, B1, W2, B2, dW1, dB1, dW2, dB2, alpha)

    if (i % 250) == 0:
      predictions = get_predictions(A2)
      accuracy = get_accuracy(predictions, Y)
      print(f"Iteration number: {i}")
      print(f"Accuracy = {accuracy}")
  return W1, B1, W2, B2
