import numpy as np
import tensorflow as tf
from PIL import Image
import io

def preprocess_image(image_bytes: bytes, target_size=(224, 224)) -> np.ndarray:
    """Data pre-processing function: Converts raw bytes to normalized tensor."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def format_prediction(probability: float) -> tuple:
    """Model utility function: Converts raw probability into label and confidence."""
    class_label = "Dog" if probability > 0.5 else "Cat"
    confidence = probability if class_label == "Dog" else (1.0 - probability)
    return class_label, round(confidence, 4)