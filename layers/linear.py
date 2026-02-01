
import numpy as np
from .activations import ReLU, softmax_calculator # Assuming activations.py is in the same directory

def initialize_parameters(input_size, hidden_size, output_size):
  W1 = np.random.rand(hidden_size, input_size) - 0.5
  B1 = np.random.rand(hidden_size, 1) - 0.5
  W2 = np.random.rand(output_size, hidden_size) - 0.5
  B2 = np.random.rand(output_size, 1) - 0.5
  return W1, B1, W2, B2

def forward_propagation(W1, B1, W2, B2, X):
  Z1 = W1.dot(X) + B1
  A1 = ReLU(Z1)
  Z2 = W2.dot(A1) + B2
  A2 = softmax_calculator(Z2)
  return Z1, A1, Z2, A2

def backward_propagation(W1, B1, W2, B2, Z1, A1, Z2, A2, X, Y, m):
  one_hot_Y = one_hot_converter(Y, A2.shape[0]) # Need to pass number of classes
  dZ2 = A2 - one_hot_Y
  dW2 = 1 / m * dZ2.dot(A1.T)
  dB2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
  dZ1 = W2.T.dot(dZ2) * (Z1 > 0)
  dW1 = 1 / m * dZ1.dot(X.T)
  dB1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
  return dW1, dB1, dW2, dB2

def update_parameters(W1, B1, W2, B2, dW1, dB1, dW2, dB2, learning_rate):
  W1 = W1 - learning_rate * dW1
  B1 = B1 - learning_rate * dB1
  W2 = W2 - learning_rate * dW2
  B2 = B2 - learning_rate * dB2
  return W1, B1, W2, B2

def one_hot_converter(Y, num_classes):
  one_hot_Y = np.zeros((num_classes, Y.size))
  one_hot_Y[Y, np.arange(Y.size)] = 1
  return one_hot_Y
