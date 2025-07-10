"""
Contains the Layer and Network classes:
- Layer: represents one fully connected layer
- Network: container for multiple layers with forward and backward passes
"""

import numpy as np
from activations import relu, relu_derivative

class Layer:
    """
    Represents a fully connected layer in the neural network.
    Handles weight/bias initialization, forward pass, and backward pass.
    """

    def __init__(self, input_size, output_size, activation_fxn, activation_der):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(-0.1, 0.1, size = (input_size, output_size))
        self.bias = np.zeros((1, output_size))
        self.activation = activation_fxn
        self.activation_der = activation_der

    def forward(self, input_data):
        """
        Computes Z = input @ weights + bias, then applies activation.
        Stores intermediate values for backpropagation.
        """
        z = input_data @ self.weights + self.bias 
        a = self.activation(z)

        self.a = a
        self.z = z
        self.input = input_data

        return a
    
    def backward(self, grad_output):
        """
        Computes gradients of weights, biases, and inputs using backpropagation.
        """
        if self.activation_der is not None:
            grad_z = grad_output * self.activation_der(self.z)
        else:
            grad_z = grad_output
            
        # Updating weights and biases; finding gradient of the input.
        self.grad_weights = self.input.T @ grad_z
        self.grad_biases = np.sum(grad_z, axis=0, keepdims=True)
        grad_input = grad_z @ self.weights.T    # Gradient wrt input of current layer

        return grad_input 

class Network:
    """
    Represents the full neural network with multiple layers.
    Handles full forward and backward propagation.
    """

    def __init__(self):
        self.layers = []

    def add(self, layer):
        """Adds a new layer to the network."""
        self.layers.append(layer)
    
    def forward(self, input_data):
        """Performs a full forward pass through all layers."""
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, grad_output):
        """Performs backpropagation through all layers in reverse."""
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)