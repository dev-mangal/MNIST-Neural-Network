"""
Implements two loss functions:
1. Mean Squared Loss (MSE)
2. Cross Entropy Loss (CEL)

Each loss class supports forward and backward propagation.
"""

import numpy as np

class MSELoss:
    """Mean Squared Error Loss for testing the network."""

    def __init__(self):
        self.predicted = None
        self.target = None

    def forward(self, predicted, target):
        """
        Computes mean squared error between predicted and target values.
        """
        # Save for backward propagation.
        self.predicted = predicted
        self.target = target
        return np.mean(np.square(predicted - target))
    
    def backward(self, predicted, target):
        """
        Derivative of MSE loss with respect to predicted output.
        """
        return 2 * (predicted - target) / predicted.shape[0]


class CrossEntropyLoss:
    """Cross Entropy Loss for classification with softmax output."""
    def __init__(self):
        self.predicted = None
        self.target = None

    def forward(self, predicted, target):
        """
        Computes cross-entropy loss between predicted probabilities and one-hot labels.
        Applies clipping for numerical stability.
        """
        # Save for backward propagation.
        self.predicted = predicted
        self.target = target

        # Numerical stability: clip predicted values to avoid log(0).
        predicted_clipped = np.clip(predicted, 1e-12, 1. - 1e-12)

        # Compute cross-entropy loss.
        loss = -np.sum(target * np.log(predicted_clipped), axis=1)  # loss per sample
        return np.mean(loss)  # average over batch
    
    def backward(self, predicted, target):
        """
        Derivative of cross-entropy loss with respect to softmax output.
        
        When using Softmax activation in the output layer along with Cross Entropy Loss,
        the gradient simplifies to (predicted - target) due to their combined derivatives. 
        This avoids the need to compute the full softmax Jacobian.
        """
        return (predicted - target) / predicted.shape[0]
