
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# Define report folder
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)


# Load saved validation predictions
data = np.load(REPORT_DIR / "validation_results.npz")

y_true = data["y_true"]
y_pred = data["y_pred"]


# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)


# Generate confusion matrix image
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=range(10)
)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(cmap="Blues", ax=ax, values_format="d")

plt.title("MNIST - SVM Confusion Matrix")
plt.tight_layout()

plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=300)
plt.close()


# Generate classification report
report = classification_report(
    y_true,
    y_pred,
    digits=4
)

print("\nClassification Report:")
print(report)


# Save classification report as an image
fig, ax = plt.subplots(figsize=(10, 7))
ax.axis("off")

ax.text(
    0.02,
    0.98,
    "Classification Report\n\n" + report,
    fontsize=12,
    family="monospace",
    verticalalignment="top"
)

plt.tight_layout()
plt.savefig(REPORT_DIR / "classification_report.png", dpi=300)
plt.close()


print("\nEvaluation completed successfully!")
print("Reports saved in:", REPORT_DIR)
