"""
Activation functions and their derivatives used in the neural network.
These functions define how the output of each neuron is transformed and 
how gradients are propagated during backpropagation.
"""
import numpy as np

# ------------------------
# ReLU
# ------------------------
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)


# ------------------------
# Sigmoid
# ------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


# ------------------------
# Tanh
# ------------------------
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


# ------------------------
# Softmax (for output layer)
# ------------------------
def softmax(x):
    """
    Applies the softmax function to the input.
    Used in the final layer to get class probabilities.
    """
    # Numerically stable version (to avoid getting large values)
    shift_x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shift_x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
