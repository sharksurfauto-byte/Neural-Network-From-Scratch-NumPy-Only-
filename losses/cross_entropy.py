
import numpy as np

def cross_entropy_loss(A2, Y, epsilon=1e-12):
    """Compute cross-entropy loss.
    A2: softmax output probabilities (num_classes, num_samples)
    Y: true labels (num_samples,)
    """
    m = Y.shape[0]
    # Clip predictions to avoid log(0)
    A2_clipped = np.clip(A2, epsilon, 1. - epsilon)

    # One-hot encode Y
    num_classes = A2.shape[0]
    one_hot_Y = np.zeros((num_classes, m))
    one_hot_Y[Y, np.arange(m)] = 1

    loss = -np.sum(one_hot_Y * np.log(A2_clipped)) / m
    return loss

def cross_entropy_backward(A2, Y):
    """Compute the gradient of cross-entropy loss with respect to Z2.
    A2: softmax output probabilities (num_classes, num_samples)
    Y: true labels (num_samples,)
    """
    num_classes = A2.shape[0]
    m = Y.shape[0]
    one_hot_Y = np.zeros((num_classes, m))
    one_hot_Y[Y, np.arange(m)] = 1

    dZ2 = A2 - one_hot_Y
    return dZ2
