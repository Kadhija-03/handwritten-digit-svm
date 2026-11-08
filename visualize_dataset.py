
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = np.load("mnist.npz")

x_train = data["x_train"]
y_train = data["y_train"]

# Display first 10 images
plt.figure(figsize=(12, 4))

for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()
# Display numerical pixel values of the first image
print("First image pixel matrix:")
print(x_train[0])

print("\nImage shape:", x_train[0].shape)
print("Actual label:", y_train[0])


# Flatten the first image
first_image = x_train[0]

flattened_image = first_image.reshape(-1)

print("\nORIGINAL IMAGE")
print("Shape:", first_image.shape)

print("\nFLATTENED IMAGE")
print("Shape:", flattened_image.shape)

print("\nFirst 20 pixel values:")
print(flattened_image[:20])
