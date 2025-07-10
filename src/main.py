"""
Main training and testing script for the MNIST neural network project.

Performs:
- Network creation
- Training with batching
- Loss calculation and weight updates
- Final test predictions
- CSV export of results
- Optional visual prediction viewer

Need to import separate activation functions and loss functions before using them to test the network,
if changing the default ones.
"""

import numpy as np
np.random.seed(42)  # For reproductibility of weight initialization, can be removed to experiment on training

import pandas as pd
import time
import os
import matplotlib.pyplot as plt

from model import Network, Layer
from activations import relu, relu_derivative, softmax
from loss import CrossEntropyLoss, MSELoss
from mnist_loader import input_train, input_test, target_train_onehot, target_train
from visualizer import preview_predictions
from metrics import plot_training_metrics

# Get the root project directory (two levels up from this file), data and images folder directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMG_DIR = os.path.join(BASE_DIR, "images")

# Create and define network architecture.
net = Network()
net.add(Layer(784, 128, relu, relu_derivative))
net.add(Layer(128, 64, relu, relu_derivative))
net.add(Layer(64, 10, softmax, None))

loss_fxn = CrossEntropyLoss()

# Training settings
epochs = 40
learning_rate = 0.15
total_samples = input_train.shape[0]
batch_size = 128

"""Optional: Uncomment for viewing accuracy and loss tracking per epoch"""
# Tracking lists for graphing
# accuracies = []
# losses = []

# Training loop
start_time = time.time() # Start timer
print("Training started...")

for epoch in range(epochs):
    num_batches = total_samples // batch_size

    for i in range(0, total_samples, batch_size):
        input_batch = input_train[i : i + batch_size]
        target_batch = target_train_onehot[i : i + batch_size]

        output = net.forward(input_batch)
        loss = loss_fxn.forward(output, target_batch)

        grad_loss = loss_fxn.backward(output, target_batch)
        net.backward(grad_loss)

        # Updation of weights and biases in each layer
        for layer in net.layers: # type : Layer
            layer.weights -= layer.grad_weights * learning_rate
            layer.bias -= layer.grad_biases * learning_rate

    """
    Optional: Uncomment to view accuracy and loss tracking per epoch
    Needs tracking lists code block uncommented
    """
    # output_train = net.forward(input_train)
    # predicted_labels_train = np.argmax(output_train, axis=1)
    # acc = np.mean(predicted_labels_train == target_train) * 100
    # accuracies.append(acc)

    # full_loss = loss_fxn.forward(output_train, target_train_onehot)
    # losses.append(full_loss)

    # print(f"Epoch {epoch + 1}/{epochs}: Accuracy = {acc:.6f}%, Loss = {full_loss:.6f}")

end_time = time.time()  # End timer
total_time = end_time - start_time
print(f"\nTraining completed in {total_time:.2f} seconds.")

"""
Optional: Uncomment to view accuracy vs epoch and loss vs epoch graphs for training
Needs accuracy and loss tracking per epoch code block uncommented
"""
# plot_training_metrics(accuracies, losses, IMG_DIR)

# Final test output predictions
output_test = net.forward(input_test)
predicted_labels_test = np.argmax(output_test, axis = 1)

# Export to CSV file
submission_df = pd.DataFrame({
    "ImageID" : np.arange(1, len(predicted_labels_test) + 1),
    "Predicted_Label" : predicted_labels_test
})
save_sub_path = os.path.join(DATA_DIR, "Predicted_Label.csv")
submission_df.to_csv(save_sub_path, index=False)
print(f"\nPredictions saved to: {save_sub_path}\n")

"""Optional: Uncomment this to view image predictions using an interactive visualizer"""
# preview_predictions(net, input_test)