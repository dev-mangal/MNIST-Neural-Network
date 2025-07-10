"""
Loads all the training and testing data and stores them in numpy arrays.
"""
import numpy as np
import pandas as pd
import os

# Define base/data paths relative to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Load train data
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
input_train = train_df.iloc[:, 1:].values       # all columns except the first
target_train = train_df.iloc[:, 0].values       # first column (labels)

# Normalize input pixels to range [0, 1]
input_train = input_train.astype(np.float32) / 255.0

# Load test data
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
input_test = test_df.values.astype(np.float32) / 255.0

"""
Convert the required output array into one hot encoded style array of shape [42000, 10].
All the digits other than the output digit (with value 1) will have the value 0.
"""
def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

target_train_onehot = one_hot(target_train)