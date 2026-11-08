
from sklearn.model_selection import train_test_split
from src.data import load_data

# Load dataset
x_train, y_train, x_test, y_test = load_data()

# Split training data into training and validation sets
x_train_part, x_val, y_train_part, y_val = train_test_split(
    x_train,
    y_train,
    test_size=0.15,
    random_state=42,
    stratify=y_train
)

print("Original training dataset:", x_train.shape)
print("New training subset:", x_train_part.shape)
print("Validation subset:", x_val.shape)
print("Official testing dataset:", x_test.shape)
