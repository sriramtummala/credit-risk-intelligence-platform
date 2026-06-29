"""
Unit tests for the Credit Risk Scoring API (api/main.py).

Run from project root:
    pytest tests/test_api.py -v

The TestClient triggers the FastAPI lifespan, which loads the real model from
models/champion_model.joblib. Tests are therefore integration-style for the
predict endpoint (real model, no mocking) and pure unit-style for validation.
"""
import pytest
from fastapi.testclient import TestClient

from api.main import app

# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Start the app (triggers lifespan → model load) once for the whole module."""
    with TestClient(app) as c:
        yield c


# Reusable valid payload — low-risk customer who pays in full every month
VALID_LOW_RISK = {
    "LIMIT_BAL": 200000, "SEX": 1, "EDUCATION": 1, "MARRIAGE": 2, "AGE": 35,
    "PAY_0": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
    "BILL_AMT1": 50000, "BILL_AMT2": 45000, "BILL_AMT3": 40000,
    "BILL_AMT4": 35000, "BILL_AMT5": 30000, "BILL_AMT6": 25000,
    "PAY_AMT1": 50000, "PAY_AMT2": 45000, "PAY_AMT3": 40000,
    "PAY_AMT4": 35000, "PAY_AMT5": 30000, "PAY_AMT6": 25000,
}

# High-risk: 2-month delay, nearly maxed-out card, zero recent payment
VALID_HIGH_RISK = {
    "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
    "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
    "BILL_AMT1": 18900, "BILL_AMT2": 19200, "BILL_AMT3": 19100,
    "BILL_AMT4": 18700, "BILL_AMT5": 17900, "BILL_AMT6": 17200,
    "PAY_AMT1": 0, "PAY_AMT2": 500, "PAY_AMT3": 500,
    "PAY_AMT4": 500, "PAY_AMT5": 500, "PAY_AMT6": 500,
}


# ── /health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_status_is_healthy(self, client):
        r = client.get("/health")
        assert r.json()["status"] == "healthy"

    def test_model_is_loaded(self, client):
        r = client.get("/health")
        assert r.json()["model_loaded"] is True


# ── /model-info ────────────────────────────────────────────────────────────────

class TestModelInfo:
    def test_returns_200(self, client):
        r = client.get("/model-info")
        assert r.status_code == 200

    def test_contains_auc(self, client):
        r = client.get("/model-info")
        data = r.json()
        assert "auc_roc" in data
        assert data["auc_roc"] > 0.5

    def test_contains_git_commit(self, client):
        r = client.get("/model-info")
        assert r.json()["git_commit"] not in ("", None)

    def test_n_features_is_51(self, client):
        r = client.get("/model-info")
        assert r.json()["n_features"] == 51


# ── /predict — success cases ───────────────────────────────────────────────────

class TestPredictSuccess:
    def test_valid_payload_returns_200(self, client):
        r = client.post("/predict", json=VALID_LOW_RISK)
        assert r.status_code == 200

    def test_response_contains_risk_score(self, client):
        r = client.post("/predict", json=VALID_LOW_RISK)
        assert "risk_score" in r.json()

    def test_risk_score_between_0_and_1(self, client):
        r = client.post("/predict", json=VALID_LOW_RISK)
        score = r.json()["risk_score"]
        assert 0.0 <= score <= 1.0

    def test_low_risk_customer_gets_approve(self, client):
        r = client.post("/predict", json=VALID_LOW_RISK)
        data = r.json()
        assert data["action"] == "Approve"
        assert data["risk_band"] == "Low Risk"

    def test_high_risk_customer_gets_decline(self, client):
        r = client.post("/predict", json=VALID_HIGH_RISK)
        data = r.json()
        assert data["action"] == "Decline"
        assert data["risk_band"] == "High Risk"

    def test_high_risk_score_above_threshold(self, client):
        r = client.post("/predict", json=VALID_HIGH_RISK)
        assert r.json()["risk_score"] > 0.30

    def test_response_includes_threshold_used(self, client):
        r = client.post("/predict", json=VALID_LOW_RISK)
        assert r.json()["threshold_used"] == pytest.approx(0.30)

    def test_response_includes_model_version(self, client):
        r = client.post("/predict", json=VALID_LOW_RISK)
        assert r.json()["model_version"] != ""


# ── /predict — validation (422 cases) ─────────────────────────────────────────

class TestPredictValidation:
    def test_negative_age_returns_422(self, client):
        bad = {**VALID_LOW_RISK, "AGE": -5}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_age_below_18_returns_422(self, client):
        bad = {**VALID_LOW_RISK, "AGE": 17}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_zero_limit_bal_returns_422(self, client):
        # LIMIT_BAL must be > 0 (gt=0)
        bad = {**VALID_LOW_RISK, "LIMIT_BAL": 0}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_negative_limit_bal_returns_422(self, client):
        bad = {**VALID_LOW_RISK, "LIMIT_BAL": -1000}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_pay0_out_of_range_returns_422(self, client):
        bad = {**VALID_LOW_RISK, "PAY_0": 99}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_invalid_sex_returns_422(self, client):
        bad = {**VALID_LOW_RISK, "SEX": 5}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_negative_pay_amt_returns_422(self, client):
        bad = {**VALID_LOW_RISK, "PAY_AMT1": -100}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_missing_limit_bal_returns_422(self, client):
        missing = {k: v for k, v in VALID_LOW_RISK.items() if k != "LIMIT_BAL"}
        r = client.post("/predict", json=missing)
        assert r.status_code == 422

    def test_missing_pay0_returns_422(self, client):
        missing = {k: v for k, v in VALID_LOW_RISK.items() if k != "PAY_0"}
        r = client.post("/predict", json=missing)
        assert r.status_code == 422

    def test_empty_body_returns_422(self, client):
        r = client.post("/predict", json={})
        assert r.status_code == 422

    def test_422_body_contains_field_detail(self, client):
        bad = {**VALID_LOW_RISK, "AGE": -1}
        r = client.post("/predict", json=bad)
        body = r.json()
        assert "detail" in body
        # At least one error should mention the AGE field
        fields = [str(e.get("loc", "")) for e in body["detail"]]
        assert any("AGE" in f for f in fields)

    def test_string_age_returns_422(self, client):
        bad = {**VALID_LOW_RISK, "AGE": "thirty"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422
