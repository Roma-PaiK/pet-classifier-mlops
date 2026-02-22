from fastapi import FastAPI, File, UploadFile, HTTPException
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from utils import preprocess_image, format_prediction

# Initialize FastAPI app
app = FastAPI(title="Pet Classifier API", description="API for Cats vs Dogs Classification")

# Load the model at startup
MODEL_PATH = "models/baseline_cnn.h5"
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

        return {
            "filename": file.filename,
            "prediction": class_label,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")