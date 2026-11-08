
import numpy as np

# Load dataset
data = np.load("mnist.npz")

x_train = data["x_train"]
x_test = data["x_test"]

# Flatten images
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

# Normalize pixel values
x_train_normalized = x_train_flat / 255.0
x_test_normalized = x_test_flat / 255.0

# Display results
print("BEFORE NORMALIZATION")
print("Minimum:", x_train_flat.min())
print("Maximum:", x_train_flat.max())

print("\nAFTER NORMALIZATION")
print("Minimum:", x_train_normalized.min())
print("Maximum:", x_train_normalized.max())

print("\nNormalized first 10 pixel values:")
print(x_train_normalized[0][:10])

print("\nNormalized dataset shape:")
print(x_train_normalized.shape)
