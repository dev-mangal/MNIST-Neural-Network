"""
Main training and testing script for the MNIST neural network project.

Performs:
- Network creation
- Training with batching
- Loss calculation and weight updates
- Final test predictions
- CSV export of results
- Optional visual prediction viewer

Need to import separate activation functions and loss functions before using them to test the network
"""

import numpy as np
import pandas as pd
import time
import os
from model import Network, Layer
from activations import relu, relu_derivative, softmax
from loss import CrossEntropyLoss, MSELoss
from mnist_loader import input_train, input_test, target_train_onehot, target_train
from model import Network, Layer
from visualizer import preview_predictions

# Get the root project directory (two levels up from this file)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

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

    """Uncomment to view losses per epoch"""
    # print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")

end_time = time.time()  # End timer
total_time = end_time - start_time
print(f"Training completed in {total_time:.2f} seconds.")

"""Uncomment to view the final training accuracy of the network"""
# output_train = net.forward(input_train)
# predicted_labels_train = np.argmax(output_train, axis=1)
# correct_all = np.sum(predicted_labels_train == target_train)
# total_all = len(target_train)
# accuracy_all = (correct_all / total_all) * 100
# print(f"\nFinal training accuracy on full dataset: {accuracy_all:.8f}%")

# Final test output predictions
output_test = net.forward(input_test)
predicted_labels_test = np.argmax(output_test, axis = 1)

# Export to CSV file
submission_df = pd.DataFrame({
    "ImageID" : np.arange(1, len(predicted_labels_test) + 1),
    "Predicted_Label" : predicted_labels_test
})
submission_df.to_csv(os.path.join(DATA_DIR, "Predicted_Label.csv"), index=False)
print("Predictions saved to Predicted_Label.csv\nIt is available in the data folder")

"""Optional: Uncomment this to view image predictions using an interactive visualizer"""
# preview_predictions(net, input_test)