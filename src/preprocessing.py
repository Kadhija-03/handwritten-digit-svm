
import numpy as np


def normalize_pixels(images):

    images = np.asarray(images, dtype=np.float32)

    if images.ndim != 2 or images.shape[1] != 784:
        raise ValueError("Expected images with 784 features.")

    if not np.isfinite(images).all():
        raise ValueError("Images contain invalid values.")

    if images.min() < 0 or images.max() > 255:
        raise ValueError("Pixel values must be between 0 and 255.")

    return images / 255.0
