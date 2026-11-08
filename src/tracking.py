
import mlflow
import mlflow.sklearn
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.data import load_data


# Configure MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Handwritten Digit Recognition - SVM")


# Load MNIST data
x_train, y_train, x_test, y_test = load_data()


# Recreate the same validation split used during training
x_train_part, x_val, y_train_part, y_val = train_test_split(
    x_train,
    y_train,
    test_size=0.15,
    random_state=42,
    stratify=y_train
)


# Model configurations
model_configs = {
    "Linear SVM": {
        "kernel": "linear",
        "C": 1.0,
        "gamma": "scale",
        "artifact": "artifacts/linear_svm.joblib"
    },
    "RBF SVM": {
        "kernel": "rbf",
        "C": 5.0,
        "gamma": "scale",
        "artifact": "artifacts/rbf_svm.joblib"
    }
}


# Track both trained models
for name, config in model_configs.items():

    print(f"\nTracking {name}...")

    # Load saved model
    model = joblib.load(config["artifact"])

    # Generate predictions for this model
    y_pred = model.predict(x_val)

    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(
        y_val, y_pred, average="macro"
    )
    recall = recall_score(
        y_val, y_pred, average="macro"
    )
    f1 = f1_score(
        y_val, y_pred, average="macro"
    )

    # Start MLflow experiment
    with mlflow.start_run(run_name=name):

        # Log parameters
        mlflow.log_param("model_name", name)
        mlflow.log_param("kernel", config["kernel"])
        mlflow.log_param("C", config["C"])
        mlflow.log_param("gamma", config["gamma"])
        mlflow.log_param("validation_samples", len(y_val))

        # Log evaluation metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log trained model
        mlflow.sklearn.log_model(
    model,
    name="model",
    skops_trusted_types=[
        "src.preprocessing.normalize_pixels"
    ]
)

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("F1-score:", round(f1, 4))
    print("Experiment saved successfully!")


print("\nMLflow tracking completed!")
