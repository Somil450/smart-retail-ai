from fastapi.testclient import TestClient
import os
import sys

# Add parent directory to sys.path to allow importing app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]

def test_dashboard_stats():
    response = client.get("/dashboard/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_visits" in data
    assert "returning_customers" in data

def test_chatbot():
    response = client.post("/chatbot/", json={"message": "What are your hours?"})
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data

def test_analyze_sentiment():
    response = client.post("/nlp/analyze-sentiment", json={"text": "I really love this store!"})
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data

# Note: testing the vision endpoints requires a sample image file upload,
# which is slightly more complex in a simple script, but the endpoints exist.
