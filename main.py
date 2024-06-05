import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

print(torch.__version__)

# agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# create some data using linreg formula: y = weight * x + bias
weight = 0.7
bias = 0.99

# create range values
start = 0
end = 1
step = 0.02

# create X and y (feautures, labels)
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias
print(X[:10], y[:10])

# split data
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
len(X_train), len(y_train)