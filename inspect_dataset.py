
import numpy as np

# Load the MNIST dataset
data = np.load("mnist.npz")

# Display available arrays
print("Available arrays:", data.files)

# Extract the arrays
x_train = data["x_train"]
y_train = data["y_train"]

x_test = data["x_test"]
y_test = data["y_test"]

# Display dataset information
print("\nTRAINING DATA")
print("Image shape:", x_train.shape)
print("Label shape:", y_train.shape)

print("\nTESTING DATA")
print("Image shape:", x_test.shape)
print("Label shape:", y_test.shape)

# Display pixel information
print("\nPIXEL INFORMATION")
print("Minimum pixel value:", x_train.min())
print("Maximum pixel value:", x_train.max())

# Display first image label
print("\nFirst training image label:", y_train[0])
