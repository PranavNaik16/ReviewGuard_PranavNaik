import pytest
import sys
import os
from fastapi.testclient import TestClient

# Add the backend/api directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../backend/api'))
from main import app

@pytest.fixture
def client():
    """Create test client for API"""
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture
def sample_reviews():
    """Sample reviews for testing"""
    return {
        "fraud": {
            "text": "best ever best ever best ever amazing 5 stars best product ever best ever best ever",
            "user_id": 999,
            "rating": 5
        },
        "legit": {
            "text": "This product works as expected. Good value for money. Delivery was fast.",
            "user_id": 111,
            "rating": 4
        },
        "mixed": {
            "text": "Okay product. Does the job but nothing special.",
            "user_id": 222,
            "rating": 3
        }
    }