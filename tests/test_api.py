"""
Integration Tests for ToxicBuddy 2.0 FastAPI Endpoints
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_direct_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_analyze_message_endpoint():
    payload = {
        "text": "Shut up and do your job properly, you fool!",
        "sender": "Alice",
        "requested_tones": ["Neutral", "Friendly", "Professional", "Constructive"]
    }
    response = client.post("/api/analyze/message", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_toxic"] is True
    assert data["overall_score"] > 0.4
    assert "Neutral" in data["rewrites"]
    assert "Constructive" in data["rewrites"]

def test_rewrite_endpoint():
    payload = {
        "text": "This is garbage code.",
        "target_tone": "Professional"
    }
    response = client.post("/api/rewrite", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["target_tone"] == "Professional"
    assert data["changes_applied"] is True

def test_analyze_conversation_endpoint():
    payload = {
        "title": "Team Sprint Review Chat",
        "messages": [
            {"sender": "Alice", "text": "Great job on the demo today team!"},
            {"sender": "Bob", "text": "Your feature was garbage and ruined the release!"},
            {"sender": "Alice", "text": "Stop being so rude and useless!"}
        ]
    }
    response = client.post("/api/analyze/conversation", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "conversation_id" in data
    assert data["total_messages"] == 3
    assert data["toxic_messages_count"] >= 1
    assert "Alice" in data["participant_summary"]
    assert "Bob" in data["participant_summary"]
    assert len(data["flagged_messages"]) >= 1

    # Fetch saved report
    conv_id = data["conversation_id"]
    report_res = client.get(f"/api/conversations/{conv_id}/report")
    assert report_res.status_code == 200
    report_data = report_res.json()
    assert report_data["conversation_id"] == conv_id

def test_user_stats_endpoint():
    user_id = "Alice"
    response = client.get(f"/api/users/{user_id}/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == user_id
    assert "risk_level" in data
