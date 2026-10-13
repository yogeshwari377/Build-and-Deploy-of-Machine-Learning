"""
EduPredict – Unit & API Tests
=============================
Test suite for Flask backend (app.py) using pytest.

Tests cover:
  1. Health check endpoint (GET /health).
  2. Frontend index page (GET /).
  3. Input validation & error handling on POST /predict:
     - Non-JSON / empty requests.
     - Missing required fields.
     - Invalid data types.
     - Out-of-bounds numeric ranges.
     - Invalid categorical values.
  4. End-to-end model inference on POST /predict:
     - Dynamic prediction (never hardcoded).
     - Support for both snake_case and exact dataset column names.
  5. 503 Service Unavailable when model is not loaded.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Add project root to sys.path so app can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app, validate_prediction_input
import app as app_module


class DummyModel:
    """A realistic mock pipeline for testing predict endpoint without needing disk weights."""
    def predict(self, df: pd.DataFrame):
        # A simple linear formula simulating a trained model:
        # Performance Index ~ 2.8 * Hours + 1.0 * Previous + 0.5 * Sleep + 0.4 * Papers + (1.5 if Yes else 0)
        hours = df["Hours Studied"].values
        prev = df["Previous Scores"].values
        extra = np.where(df["Extracurricular Activities"].values == "Yes", 1.5, 0.0)
        sleep = df["Sleep Hours"].values
        papers = df["Sample Question Papers Practiced"].values
        pred = (2.8 * hours) + (0.8 * prev) + (0.5 * sleep) + (0.4 * papers) + extra
        return np.clip(pred, 10.0, 100.0)


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_model(monkeypatch):
    """Fixture that injects a dummy ML model into app_module._model."""
    dummy = DummyModel()
    monkeypatch.setattr(app_module, "_model", dummy)
    return dummy


# ==============================================================================
# 1. Health & Home Route Tests
# ==============================================================================
def test_health_check(client):
    """GET /health should return 200 OK and health status JSON."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "service" in data


def test_index_page(client):
    """GET / should return 200 OK and render the HTML interface."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"EduPredict" in response.data
    assert b"Hours Studied" in response.data
    assert b"style.css" in response.data
    assert b"script.js" in response.data


def test_static_css_served(client):
    """GET /static/style.css should return 200 OK and valid CSS content."""
    response = client.get("/static/style.css")
    assert response.status_code == 200
    assert b"--primary-purple" in response.data


def test_static_js_served(client):
    """GET /static/script.js should return 200 OK and valid JS content."""
    response = client.get("/static/script.js")
    assert response.status_code == 200
    assert b"predictionForm" in response.data


# ==============================================================================
# 2. Input Validation Unit Tests
# ==============================================================================
def test_validate_input_valid_snake_case():
    payload = {
        "hours_studied": 7,
        "previous_scores": 85,
        "extracurricular_activities": "Yes",
        "sleep_hours": 7,
        "sample_question_papers_practiced": 4,
    }
    cleaned, err = validate_prediction_input(payload)
    assert err is None
    assert cleaned["Hours Studied"] == 7.0
    assert cleaned["Previous Scores"] == 85.0
    assert cleaned["Extracurricular Activities"] == "Yes"
    assert cleaned["Sleep Hours"] == 7.0
    assert cleaned["Sample Question Papers Practiced"] == 4.0


def test_validate_input_valid_dataset_columns():
    payload = {
        "Hours Studied": 5,
        "Previous Scores": 70,
        "Extracurricular Activities": "no",
        "Sleep Hours": 8,
        "Sample Question Papers Practiced": 2,
    }
    cleaned, err = validate_prediction_input(payload)
    assert err is None
    assert cleaned["Extracurricular Activities"] == "No"  # normalized


def test_validate_input_missing_field():
    payload = {
        "hours_studied": 5,
        "previous_scores": 70,
        # missing extracurricular_activities
        "sleep_hours": 8,
        "sample_question_papers_practiced": 2,
    }
    cleaned, err = validate_prediction_input(payload)
    assert cleaned is None
    assert "extracurricular_activities" in err


def test_validate_input_invalid_ranges():
    # Negative hours
    _, err1 = validate_prediction_input({"hours_studied": -1, "previous_scores": 50, "extracurricular_activities": "Yes", "sleep_hours": 6, "sample_question_papers_practiced": 1})
    assert "hours_studied" in err1

    # Score > 100
    _, err2 = validate_prediction_input({"hours_studied": 4, "previous_scores": 105, "extracurricular_activities": "Yes", "sleep_hours": 6, "sample_question_papers_practiced": 1})
    assert "previous_scores" in err2

    # Invalid categorical
    _, err3 = validate_prediction_input({"hours_studied": 4, "previous_scores": 70, "extracurricular_activities": "Maybe", "sleep_hours": 6, "sample_question_papers_practiced": 1})
    assert "extracurricular_activities" in err3

    # Negative papers
    _, err4 = validate_prediction_input({"hours_studied": 4, "previous_scores": 70, "extracurricular_activities": "Yes", "sleep_hours": 6, "sample_question_papers_practiced": -3})
    assert "sample_question_papers_practiced" in err4


# ==============================================================================
# 3. POST /predict Route Validation Tests
# ==============================================================================
def test_predict_non_json(client, mock_model):
    """POST /predict without JSON content type should return 400."""
    response = client.post("/predict", data="not json", content_type="text/plain")
    assert response.status_code == 400
    assert response.get_json()["status"] == "fail"


def test_predict_missing_payload(client, mock_model):
    """POST /predict with empty JSON should return 400."""
    response = client.post("/predict", json={})
    assert response.status_code == 400
    assert response.get_json()["status"] == "fail"


def test_predict_validation_error(client, mock_model):
    """POST /predict with out-of-range value should return 400 with helpful error."""
    response = client.post("/predict", json={
        "hours_studied": 50, # invalid > 24
        "previous_scores": 80,
        "extracurricular_activities": "Yes",
        "sleep_hours": 8,
        "sample_question_papers_practiced": 2
    })
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "fail"
    assert "hours_studied" in data["error"]


# ==============================================================================
# 4. POST /predict Dynamic Inference Tests (Never Hardcoded)
# ==============================================================================
def test_predict_dynamic_success(client, mock_model):
    """POST /predict should return dynamic prediction based on input features."""
    student_a = {
        "hours_studied": 2,
        "previous_scores": 50,
        "extracurricular_activities": "No",
        "sleep_hours": 6,
        "sample_question_papers_practiced": 1,
    }
    res_a = client.post("/predict", json=student_a)
    assert res_a.status_code == 200
    data_a = res_a.get_json()
    assert data_a["status"] == "success"
    score_a = data_a["prediction"]

    student_b = {
        "hours_studied": 9,
        "previous_scores": 95,
        "extracurricular_activities": "Yes",
        "sleep_hours": 8,
        "sample_question_papers_practiced": 8,
    }
    res_b = client.post("/predict", json=student_b)
    assert res_b.status_code == 200
    data_b = res_b.get_json()
    score_b = data_b["prediction"]

    # Student B should score significantly higher than Student A (proving dynamic calculation)
    assert score_b > score_a
    assert data_a["inputs"]["Hours Studied"] == 2.0
    assert data_b["inputs"]["Hours Studied"] == 9.0


def test_predict_unloaded_model(client, monkeypatch):
    """POST /predict should return 503 if model is not loaded."""
    monkeypatch.setattr(app_module, "_model", None)
    monkeypatch.setattr(app_module, "load_trained_model", lambda: None)

    response = client.post("/predict", json={
        "hours_studied": 5,
        "previous_scores": 70,
        "extracurricular_activities": "Yes",
        "sleep_hours": 7,
        "sample_question_papers_practiced": 3,
    })
    assert response.status_code == 503
    data = response.get_json()
    assert data["status"] == "error"
    assert "model is not loaded" in data["error"].lower()
