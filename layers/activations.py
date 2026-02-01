
import numpy as np

def ReLU(X):
  return np.maximum(X, 0)

def softmax_calculator(Z):
  # Ensure Z is 2D array for correct broadcasting, if it's 1D, reshape it
  if Z.ndim == 1:
    Z = Z.reshape(-1, 1)
  exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True)) # For numerical stability
  return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
