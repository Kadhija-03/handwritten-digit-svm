
import numpy as np


def load_data():
    # Load MNIST dataset
    data = np.load("mnist.npz")

    x_train = data["x_train"]
    y_train = data["y_train"]

    x_test = data["x_test"]
    y_test = data["y_test"]

    # Flatten images into 784 features
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_data()

    print("Training data shape:", x_train.shape)
    print("Training labels shape:", y_train.shape)

    print("Testing data shape:", x_test.shape)
    print("Testing labels shape:", y_test.shape)
