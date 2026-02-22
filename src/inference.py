import numpy as np

def postprocess_prediction(pred: np.ndarray) -> dict:
    """pred expected shape (1,1). threshold 0.5."""
    probability = float(pred[0][0])
    label = "Dog" if probability > 0.5 else "Cat"
    confidence = probability if label == "Dog" else (1.0 - probability)
    return {"prediction": label, "confidence": round(confidence, 4)}
