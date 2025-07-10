"""
Provides a function to visualize training accuracy and loss over epochs.
Call this from main.py after training is done.
"""

import matplotlib.pyplot as plt
import os

def plot_training_metrics(accuracies, losses, img_dir):
    """
    Plots accuracy and loss per epoch and saves the graph.

    Parameters:
    - accuracies: list of accuracy values per epoch
    - losses: list of loss values per epoch
    - img_dir: path to the directory to save the graph
    """
    plt.figure(figsize=(12, 5))

    # Accuracy vs Epoch
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o', color='blue')
    plt.title("Accuracy vs. Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)

    # Loss vs Epoch
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(losses) + 1), losses, marker='x', color='red')
    plt.title("Loss vs. Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.tight_layout()
    save_path = os.path.join(img_dir, "training_metrics.png")
    plt.savefig(save_path)
    plt.show()
    print(f"Training graph saved to: {save_path}")
