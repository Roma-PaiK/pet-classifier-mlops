import numpy as np
from src.inference import postprocess_prediction

def test_postprocess_cat():
    out = postprocess_prediction(np.array([[0.1]], dtype=float))
    assert out["prediction"] == "Cat"
    assert 0.8 < out["confidence"] <= 1.0

def test_postprocess_dog():
    out = postprocess_prediction(np.array([[0.9]], dtype=float))
    assert out["prediction"] == "Dog"
    assert 0.8 < out["confidence"] <= 1.0
