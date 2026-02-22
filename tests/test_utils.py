import pytest
import numpy as np
import io
from PIL import Image
from utils import preprocess_image, format_prediction

def test_preprocess_image():
    """Test the data pre-processing function"""
    # Create a fake 500x500 red image in memory
    img = Image.new('RGB', (500, 500), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()

    # Run the function
    processed = preprocess_image(img_bytes, target_size=(224, 224))

    # Assert it reshaped correctly for the CNN (Batch, Height, Width, Channels)
    assert processed.shape == (1, 224, 224, 3)
    assert isinstance(processed, np.ndarray)

def test_format_prediction():
    """Test the model utility/inference function"""
    # Test Cat probability (e.g., model outputs 0.1)
    label, conf = format_prediction(0.1)
    assert label == "Cat"
    assert conf == 0.9

    # Test Dog probability (e.g., model outputs 0.95)
    label, conf = format_prediction(0.95)
    assert label == "Dog"
    assert conf == 0.95