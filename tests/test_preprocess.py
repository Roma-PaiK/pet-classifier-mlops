from PIL import Image
import numpy as np
from src.preprocess import preprocess_pil_image

def test_preprocess_shape():
    img = Image.new("RGB", (500, 300))
    arr = preprocess_pil_image(img)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1, 224, 224, 3)
