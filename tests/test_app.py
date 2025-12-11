import pytest

# Import predict() from your app (change if needed)
from app import predict

def test_predict_returns_valid_class_id():
    """Ensure that predict() returns a valid non-empty class ID."""
    result = predict("test input")

    # The returned value should not be None or empty
    assert result is not None
    assert str(result).strip() != ""


