
import joblib
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.data import load_data
from src.preprocessing import normalize_pixels


# Create folders if they do not exist
ARTIFACT_DIR = Path("artifacts")
REPORT_DIR = Path("reports")

ARTIFACT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# Load MNIST dataset
x_train, y_train, x_test, y_test = load_data()


# Split training data into training and validation sets
x_train_part, x_val, y_train_part, y_val = train_test_split(
    x_train,
    y_train,
    test_size=0.15,
    random_state=42,
    stratify=y_train
)


# Define two SVM pipelines
models = {
    "Linear SVM": Pipeline([
        ("normalize", FunctionTransformer(
            normalize_pixels,
            validate=False
        )),
        ("svm", SVC(kernel="linear", C=1.0))
    ]),

    "RBF SVM": Pipeline([
        ("normalize", FunctionTransformer(
            normalize_pixels,
            validate=False
        )),
        ("svm", SVC(kernel="rbf", C=5.0, gamma="scale"))
    ])
}


# Train and evaluate both models
results = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(x_train_part, y_train_part)

    y_pred = model.predict(x_val)

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average="macro")
    recall = recall_score(y_val, y_pred, average="macro")
    f1 = f1_score(y_val, y_pred, average="macro")

    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    filename = name.lower().replace(" ", "_") + ".joblib"

    joblib.dump(model, ARTIFACT_DIR / filename)

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))


# Select the best model using macro F1-score
best_name = max(
    results,
    key=lambda name: results[name]["f1_score"]
)

best_model = models[best_name]

joblib.dump(best_model, ARTIFACT_DIR / "best_model.joblib")


# Save validation predictions for later evaluation
best_predictions = best_model.predict(x_val)

np.savez(
    REPORT_DIR / "validation_results.npz",
    y_true=y_val,
    y_pred=best_predictions
)


print("\nBest Model:", best_name)
print("Saved models in:", ARTIFACT_DIR)
print("Saved validation predictions in:", REPORT_DIR)
