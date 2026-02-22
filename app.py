from fastapi import FastAPI, File, UploadFile, HTTPException, Request
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import time
import logging 
from fastapi.responses import PlainTextResponse

# Initialize FastAPI app
app = FastAPI(title="Pet Classifier API", description="API for Cats vs Dogs Classification")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pet-classifier")

_METRICS = {
    "requests_total": 0,
    "predict_requests_total": 0,
    "predict_failures_total": 0,
    "last_predict_latency_ms": 0.0,
}


# Load the model at startup
MODEL_PATH = os.getenv("MODEL_PATH", "models/baseline_cnn.h5")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define image size based on your M1 training script
IMG_SIZE = (224, 224)

@app.get("/health")
def health_check():
    """Health check endpoint to verify the service is running."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")
    return {"status": "healthy", "model": "loaded"}

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Metrics endpoint to expose service metrics."""
    lines = [f"{k} {v}" for k, v in _METRICS.items()]
    return "\n".join(lines) + "\n"

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware to track metrics."""
    _METRICS["requests_total"] += 1
    return await call_next(request)

start = time.time()
_METRICS["predict_requests_total"] += 1
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Prediction endpoint: Accepts an image and returns Cat/Dog probability."""
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPEG or PNG.")

    try:
        # 1. Read the uploaded image
        contents = await file.read()
        # image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 2. Preprocess the image
        # image = image.resize(IMG_SIZE)
        # img_array = tf.keras.preprocessing.image.img_to_array(image)
        # img_array = np.expand_dims(img_array, axis=0) # Create a batch of 1
        
        # Note: Your model already has layers.Rescaling(1./255) inside it, 
        # so we pass the raw 0-255 array directly to model.predict!

        img_array = preprocess_image(contents, IMG_SIZE)

        # 3. Make prediction
        prediction = model.predict(img_array)
        probability = float(prediction[0][0])
        
        # Keras image_dataset_from_directory sorts alphabetically (Cats=0, Dogs=1)
        # class_label = "Dog" if probability > 0.5 else "Cat"
        
        # If it's a Cat, the model outputs a low number (e.g., 0.1). 
        # We invert it (1 - 0.1 = 0.9) so the user sees a "90% confidence it's a Cat".
        # confidence = probability if class_label == "Dog" else (1.0 - probability)

        class_label, confidence = format_prediction(probability)

        latency_ms = (time.time() - start) * 1000
        _METRICS["last_predict_latency_ms"] = float(latency_ms)
        logger.info(
            "prediction_done filename=%s label=%s confidence=%.4f latency_ms=%.2f",
            file.filename, class_label, float(confidence), latency_ms
        )

        return {
            "filename": file.filename,
            "prediction": class_label,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        
        _METRICS["predict_failures_total"] += 1
        logger.exception("prediction_failed filename=%s err=%s", getattr(file, "filename", "unknown"), str(e))

        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")