# Handwritten Digit Recognition Using SVM

## Project Overview

Handwritten Digit Recognition is a machine learning application that identifies handwritten digits (0–9) from uploaded images using Support Vector Machine (SVM) classification.

The project compares Linear and RBF SVM kernels and deploys the selected model through a FastAPI backend and Streamlit user interface. Docker is used for containerization and deployment.

## Objectives

- Prepare and preprocess handwritten digit images.
- Train and compare Linear and RBF SVM models.
- Evaluate model performance using classification metrics.
- Develop a REST API using FastAPI.
- Build an interactive interface using Streamlit.
- Containerize the application using Docker.
- Implement input validation, prediction statistics and error logging.

## Technologies Used

- Python
- NumPy and Pandas
- Scikit-learn
- Matplotlib
- OpenCV and Pillow
- FastAPI
- Streamlit
- MLflow
- Docker and Docker Compose
- Git and GitHub

## Dataset

The project uses the MNIST handwritten digit dataset.

- Training images: 60,000
- Testing images: 10,000
- Image dimensions: 28 × 28 pixels
- Number of classes: 10 (digits 0–9)
- Features per image: 784

## Data Preprocessing

1. Load the MNIST dataset.
2. Inspect the dataset and class distribution.
3. Flatten each 28 × 28 image into 784 features.
4. Normalize pixel values from 0–255 to 0–1.
5. Split the training data into training and validation subsets.
6. Preserve the official test dataset for final evaluation.

The image preprocessing pipeline also handles uploaded images through grayscale conversion, thresholding, cropping, resizing and centering.

## Model Development

Two Support Vector Machine models were trained and compared:

- Linear SVM
- RBF SVM

The RBF SVM was selected based on validation performance.

## Model Evaluation

The selected RBF SVM achieved the following results on the official MNIST test dataset:

| Metric | Score |
|---|---:|
| Accuracy | 98.26% |
| Macro Precision | 98.26% |
| Macro Recall | 98.25% |
| Macro F1-score | 98.25% |

Confusion matrices and classification reports are available in the `reports/` directory.

## Application Architecture

The application consists of:

- **Streamlit:** User interface for uploading handwritten digit images.
- **FastAPI:** REST API for image validation and prediction.
- **SVM Model:** Predicts the handwritten digit.
- **Docker:** Runs the API and Streamlit application in containers.
- **MLflow:** Tracks model parameters, metrics and artifacts.

## Project Structure

```text
handwritten-digit-svm/
│
├── artifacts/
│   ├── best_model.joblib
│   ├── linear_svm.joblib
│   ├── rbf_svm.joblib
│   └── model_metadata.json
│
├── images/
├── logs/
├── reports/
│   ├── classification_report.png
│   ├── confusion_matrix.png
│   ├── test_classification_report.png
│   └── test_confusion_matrix.png
│
├── src/
│   ├── api.py
│   ├── data.py
│   ├── evaluate.py
│   ├── image_preprocessing.py
│   ├── preprocessing.py
│   ├── test.py
│   ├── tracking.py
│   └── train.py
│
├── app.py
├── config.json
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── mnist.npz
```

## Installation and Setup

Clone the repository:

```bash
git clone https://github.com/Kadhija-03/handwritten-digit-svm.git
cd handwritten-digit-svm
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training and Evaluation

Run the following commands from the project root:

```bash
python -m src.data
python -m src.train
python -m src.evaluate
python -m src.test
python -m src.tracking
```

## Running the Application

### Using Docker Compose

Build and start both services:

```bash
docker compose up -d --build
```

Access the applications:

- Streamlit interface: http://localhost:8501
- FastAPI documentation: http://localhost:8000/docs
- API health check: http://localhost:8000/health
- Prediction statistics: http://localhost:8000/stats

Stop the containers:

```bash
docker compose down
```

### Monitoring

The application provides:

- Total prediction count.
- Successful prediction count.
- Failed prediction count.
- Persistent prediction statistics.
- Application error logging.
- Input file type and image dimension validation.

Prediction statistics are stored in `logs/prediction_stats.json`, while application logs are stored in `logs/application.log`.

## Model Versioning and Tracking

- Trained models are saved as Joblib artifacts.
- `config.json` stores application and preprocessing configuration.
- `artifacts/model_metadata.json` records selected model information and evaluation metrics.
- MLflow tracks training parameters, metrics and model artifacts.
- Git maintains source-code version history.

## Limitations

- The application is designed for handwritten digit classification (0–9).
- Prediction quality depends on image clarity and handwriting style.
- The model is trained on MNIST and may perform differently on images that significantly differ from the training dataset.
- The application does not guarantee correct predictions for every uploaded image.

## Author

**Kadhija S**

M.Tech Artificial Intelligence

TKM College of Engineering

## License

This project is developed for academic and educational purposes.
