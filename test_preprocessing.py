
import numpy as np

from src.preprocessing import normalize_pixels

# Create one sample image with 784 pixel features
sample = np.zeros((1, 784), dtype=np.uint8)

# Set the first three pixel values
sample[0, :3] = [0, 128, 255]

# Normalize the data
result = normalize_pixels(sample)

print("Original pixels:", sample[0, :3])
print("Normalized pixels:", result[0, :3])
print("Output shape:", result.shape)
