
import numpy as np

# Load MNIST dataset
data = np.load("mnist.npz")

x_train = data["x_train"]
y_train = data["y_train"]

x_test = data["x_test"]
y_test = data["y_test"]

# Flatten training images
x_train_flat = x_train.reshape(x_train.shape[0], -1)

# Flatten testing images
x_test_flat = x_test.reshape(x_test.shape[0], -1)

# Display shapes
print("ORIGINAL DATASET")
print("Training images:", x_train.shape)
print("Testing images:", x_test.shape)

print("\nFLATTENED DATASET")
print("Training features:", x_train_flat.shape)
print("Testing features:", x_test_flat.shape)

print("\nLABEL SHAPES")
print("Training labels:", y_train.shape)
print("Testing labels:", y_test.shape)
