import pytest
from fastapi.testclient import TestClient
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../backend/api'))

def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["api"] == "healthy"
    assert "model" in data
    assert "database" in data
    assert "redis" in data

def test_batch_detection_single(client, sample_reviews):
    """Test batch detection with single review"""
    payload = {
        "reviews": [sample_reviews["fraud"]]
    }
    response = client.post("/api/reviews/detect/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 1
    result = data["results"][0]
    assert "review_id" in result
    assert "score" in result
    assert "explanation" in result
    assert "status" in result

def test_batch_detection_multiple(client, sample_reviews):
    """Test batch detection with multiple reviews"""
    payload = {
        "reviews": [
            sample_reviews["fraud"],
            sample_reviews["legit"],
            sample_reviews["mixed"]
        ]
    }
    response = client.post("/api/reviews/detect/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 3
    
    # Check fraud review has high score
    fraud_result = data["results"][0]
    assert fraud_result["score"] > 0.7
    assert fraud_result["status"] == "quarantined"
    assert len(fraud_result["explanation"]) > 0
    
    # Check legit review has low score
    legit_result = data["results"][1]
    assert legit_result["score"] < 0.1
    assert legit_result["status"] == "approved"

def test_batch_detection_empty(client):
    """Test batch detection with empty list"""
    payload = {"reviews": []}
    response = client.post("/api/reviews/detect/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 0

def test_batch_detection_invalid_input(client):
    """Test batch detection with invalid input"""
    payload = {"invalid": "data"}
    response = client.post("/api/reviews/detect/batch", json=payload)
    assert response.status_code == 422  # Validation error

def test_submit_review(client, sample_reviews):
    """Test single review submission"""
    payload = sample_reviews["legit"]
    response = client.post("/api/reviews/submit", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "review_id" in data
    assert "score" in data
    assert "status" in data
    assert data["score"] < 0.1

def test_drift_status(client):
    """Test drift monitoring endpoint"""
    response = client.get("/api/drift/status")
    assert response.status_code == 200
    data = response.json()
    assert "drift_detected" in data
    assert "current_p_value" in data
    assert "recent_drifts" in data

def test_get_review_not_found(client):
    """Test get non-existent review"""
    response = client.get("/api/reviews/non-existent-id")
    assert response.status_code == 404

def test_health_check_failure(monkeypatch):
    """Test health check when services are down"""
    # Mock MongoDB connection failure
    import main
    monkeypatch.setattr(main, 'db', None)
    
    with TestClient(main.app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["database"] == False