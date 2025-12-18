import sys
import os
import pytest

# Add project root to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import predict


def test_predict_returns_valid_class_id():
    result = predict("test input")
    assert result is not None
    assert str(result).strip() != ""
