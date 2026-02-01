
# Neural Network From Scratch (NumPy Only)

A complete implementation of a feedforward neural network from first principles, including manual backpropagation, gradient checking, and optimization — without using any deep learning frameworks.

## Why this project exists

Most deep learning work relies on automatic differentiation.
This project was built to re-derive and verify backpropagation manually, ensuring a first-principles understanding of how neural networks actually learn.

<img width="526" height="466" alt="image" src="https://github.com/user-attachments/assets/4750891b-73fa-44e6-83be-4c172b00d271" />


## What’s implemented

*   Fully connected neural network (MLP)
*   Manual forward and backward passes
*   Numerically stable Softmax + Cross-Entropy
*   Stochastic Gradient Descent (from scratch)
*   Gradient checking via finite differences
*   Mini-batch training & shuffling
*   Loss and accuracy tracking

**No PyTorch. No TensorFlow. No autograd.**

### Architecture
784 → 128 → 64 → 10
(ReLU activations, Softmax inside loss)

### Correctness Verification

Before scaling, the model was intentionally overfitted on a tiny dataset:

*   100 MNIST samples
*   100% training accuracy

**Confirms correctness of:**

*   Gradient computation
*   Backpropagation chain rule
*   Parameter updates

(This step acts as a unit test for the entire network.)

### Results (MNIST)

| Dataset     | Train Acc | Val Acc |
|-------------|-----------|---------|
| 20k samples | 98%       | 96%     |

Plots available in `/plots`.

## Key Learnings

*   Difference between full-batch and mini-batch GD
*   Learning rate impact on convergence
*   Diagnosing underfitting vs optimization failure
*   Numerical stability in softmax
*   Practical gradient debugging

## Limitations

*   No convolutional layers
*   No automatic differentiation
*   No data augmentation

(These are intentional — this project focuses on core learning mechanics.)

## Why this matters

This project serves as a reference implementation for understanding:

*   Backpropagation
*   Optimization dynamics
*   Training stability

It also forms the foundation for more advanced architectures such as Transformers and Vision Transformers.

## How to Run

1.  Ensure you have Python 3.x installed.
2.  Install necessary libraries: `pip install numpy tensorflow`
3.  Navigate to the `neural-network-from-scratch` directory.
4.  Run the training script: `python train.py`
