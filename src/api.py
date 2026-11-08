
import logging
from pathlib import Path
from io import BytesIO

import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image

from src.image_preprocessing import preprocess_image


# Create logs directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=LOG_DIR / "application.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)

logger = logging.getLogger(__name__)

# Prediction statistics
import json

STATS_FILE = Path("logs/prediction_stats.json")

# Load previously saved statistics
if STATS_FILE.exists():
    with open(STATS_FILE, "r") as file:
        prediction_stats = json.load(file)
else:
    prediction_stats = {
        "total_predictions": 0,
        "successful_predictions": 0,
        "failed_predictions": 0
    }


def save_statistics():
    with open(STATS_FILE, "w") as file:
        json.dump(prediction_stats, file, indent=4)
# Load trained model
model = joblib.load("artifacts/best_model.joblib")

app = FastAPI(
    title="Handwritten Digit Recognition API",
    description="SVM-based handwritten digit recognition with monitoring",
    version="1.0"
)


@app.get("/")
def home():
    return {
        "message": "Handwritten Digit Recognition API is running!"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.get("/stats")
def get_statistics():
    return prediction_stats


@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):

    prediction_stats["total_predictions"] += 1

    try:
        logger.info("Prediction request received: %s", file.filename)

        if file.content_type not in ["image/jpeg", "image/png"]:
            raise ValueError("Only JPEG and PNG images are supported.")

        contents = await file.read()

        if not contents:
            raise ValueError("Uploaded file is empty.")

        image = Image.open(BytesIO(contents))

# Validate image dimensions
        width, height = image.size

        if width < 28 or height < 28:
            raise ValueError("Image dimensions must be at least 28x28 pixels.")

        if width > 5000 or height > 5000:
            raise ValueError("Image dimensions must not exceed 5000x5000 pixels.")

        image = image.convert("RGB")

        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)

        predicted_digit = int(prediction[0])

        prediction_stats["successful_predictions"] += 1
        save_statistics()

        logger.info(
            "Prediction successful. File: %s, Digit: %s",
            file.filename,
            predicted_digit
        )

        return {
            "filename": file.filename,
            "predicted_digit": predicted_digit
        }

    except Exception as e:

        prediction_stats["failed_predictions"] += 1
        save_statistics()

        logger.exception(
            "Prediction failed for file: %s",
            file.filename
        )

        raise HTTPException(
            status_code=400,
            detail=f"Error processing image: {str(e)}"
        )
