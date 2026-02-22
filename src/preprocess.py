from PIL import Image
import numpy as np

IMG_SIZE = (224, 224)

def preprocess_pil_image(image: Image.Image) -> np.ndarray:
    """Convert to RGB, resize to 224x224, return numpy batch (1, 224, 224, 3)."""
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr
