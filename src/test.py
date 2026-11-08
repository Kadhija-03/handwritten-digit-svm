
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from src.data import load_data


# Load the official MNIST test dataset
x_train, y_train, x_test, y_test = load_data()


# Load the selected trained model
model = joblib.load("artifacts/best_model.joblib")


# Predict digits using the test dataset
print("Evaluating the best model on official test data...")

y_pred = model.predict(x_test)


# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")


print("\nFinal Test Results")
print("------------------")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))


# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=range(10)
)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(cmap="Blues", ax=ax, values_format="d")

plt.title("MNIST - Final Test Confusion Matrix")
plt.tight_layout()

plt.savefig("reports/test_confusion_matrix.png", dpi=300)
plt.close()


# Generate classification report
report = classification_report(
    y_test,
    y_pred,
    digits=4
)

print("\nFinal Classification Report:")
print(report)


# Save classification report as an image
fig, ax = plt.subplots(figsize=(10, 7))
ax.axis("off")

ax.text(
    0.02,
    0.98,
    "Final Test Classification Report\n\n" + report,
    fontsize=12,
    family="monospace",
    verticalalignment="top"
)

plt.tight_layout()
plt.savefig("reports/test_classification_report.png", dpi=300)
plt.close()


print("\nFinal testing completed successfully!")
print("Test reports saved in: reports/")
