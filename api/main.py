"""
Credit Risk Scoring Service — FastAPI

Run (from project root):
    uvicorn api.main:app --reload

Swagger UI: http://127.0.0.1:8000/docs
ReDoc:       http://127.0.0.1:8000/redoc
"""
import json
import logging
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, model_validator

from src.features.engineer import engineer_features

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("credit_risk_api")

# ── Paths & constants ─────────────────────────────────────────────────────────
_MODEL_PATH    = PROJECT_ROOT / "models" / "champion_model.joblib"
_METADATA_PATH = PROJECT_ROOT / "models" / "champion_model_metadata.json"
_THRESHOLD     = 0.30
_DROP_COLS     = {"default", "risk_segment", "edu_label", "mar_label", "age_group"}

# ── App state (populated in lifespan) ─────────────────────────────────────────
_state: dict = {
    "model":        None,
    "metadata":     {},
    "feature_cols": [],
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and metadata once at startup; release at shutdown."""
    if not _MODEL_PATH.exists():
        raise RuntimeError(
            f"No model artifact found at {_MODEL_PATH}. "
            "Run `python main.py` from the project root to train and save the model."
        )
    _state["model"] = joblib.load(_MODEL_PATH)
    logger.info("Model loaded from %s", _MODEL_PATH)

    if _METADATA_PATH.exists():
        _state["metadata"] = json.loads(_METADATA_PATH.read_text(encoding="utf-8"))
        _state["feature_cols"] = _state["metadata"].get("feature_names", [])
        logger.info(
            "Metadata loaded — version=%s  AUC=%.4f  git=%s",
            _state["metadata"].get("version"),
            _state["metadata"].get("auc_roc", 0),
            _state["metadata"].get("git_commit"),
        )

    yield  # app runs here

    _state["model"] = None
    logger.info("Model unloaded — service shutting down")


app = FastAPI(
    title="Credit Risk Scoring Service",
    description=(
        "REST API for the **Credit Risk Intelligence Platform v1.0**.\n\n"
        "Accepts raw customer features and returns a risk score (0–1), "
        "risk band (Low / Medium / High), and an automated credit action "
        "(Approve / Review / Decline).\n\n"
        "**Operational threshold:** 0.30  \n"
        "**Champion model:** XGBoost (AUC-ROC 0.7725, Gini 0.5450)\n\n"
        "See `/model-info` for full training provenance."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class CustomerInput(BaseModel):
    """
    Raw credit card customer features (23 variables from UCI dataset schema).
    All monetary values are in New Taiwan Dollars (NT$).
    Payment status codes: -2=no consumption, -1=paid in full, 0=revolving credit,
    1=1-month delay, 2=2-month delay, … 8=8-month delay.
    """
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "High-risk customer (likely Decline)",
                    "value": {
                        "LIMIT_BAL": 20000,  "SEX": 2, "EDUCATION": 2,
                        "MARRIAGE": 1,       "AGE": 24,
                        "PAY_0":  2, "PAY_2":  2, "PAY_3": -1,
                        "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
                        "BILL_AMT1": 18900, "BILL_AMT2": 19200, "BILL_AMT3": 19100,
                        "BILL_AMT4": 18700, "BILL_AMT5": 17900, "BILL_AMT6": 17200,
                        "PAY_AMT1":    0, "PAY_AMT2":  500, "PAY_AMT3":  500,
                        "PAY_AMT4":  500, "PAY_AMT5":  500, "PAY_AMT6":  500,
                    },
                },
                {
                    "summary": "Low-risk customer (likely Approve)",
                    "value": {
                        "LIMIT_BAL": 200000, "SEX": 1, "EDUCATION": 1,
                        "MARRIAGE": 2,       "AGE": 35,
                        "PAY_0": -1, "PAY_2": -1, "PAY_3": -1,
                        "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
                        "BILL_AMT1": 50000, "BILL_AMT2": 45000, "BILL_AMT3": 40000,
                        "BILL_AMT4": 35000, "BILL_AMT5": 30000, "BILL_AMT6": 25000,
                        "PAY_AMT1": 50000, "PAY_AMT2": 45000, "PAY_AMT3": 40000,
                        "PAY_AMT4": 35000, "PAY_AMT5": 30000, "PAY_AMT6": 25000,
                    },
                },
            ]
        }
    }

    # Demographics
    LIMIT_BAL: float = Field(..., gt=0,      description="Credit limit (NT$)")
    SEX:       int   = Field(..., ge=1, le=2, description="1=Male, 2=Female")
    EDUCATION: int   = Field(..., ge=1, le=4, description="1=Grad school, 2=University, 3=High school, 4=Other")
    MARRIAGE:  int   = Field(..., ge=1, le=3, description="1=Married, 2=Single, 3=Other")
    AGE:       int   = Field(..., ge=18, le=100, description="Age in years")

    # Payment status (last 6 months)
    PAY_0: int = Field(..., ge=-2, le=8, description="Payment status last month")
    PAY_2: int = Field(..., ge=-2, le=8, description="Payment status 2 months ago")
    PAY_3: int = Field(..., ge=-2, le=8, description="Payment status 3 months ago")
    PAY_4: int = Field(..., ge=-2, le=8, description="Payment status 4 months ago")
    PAY_5: int = Field(..., ge=-2, le=8, description="Payment status 5 months ago")
    PAY_6: int = Field(..., ge=-2, le=8, description="Payment status 6 months ago")

    # Bill amounts (last 6 months, NT$)
    BILL_AMT1: float = Field(..., description="Bill statement last month (NT$)")
    BILL_AMT2: float = Field(..., description="Bill statement 2 months ago")
    BILL_AMT3: float = Field(..., description="Bill statement 3 months ago")
    BILL_AMT4: float = Field(..., description="Bill statement 4 months ago")
    BILL_AMT5: float = Field(..., description="Bill statement 5 months ago")
    BILL_AMT6: float = Field(..., description="Bill statement 6 months ago")

    # Payment amounts (last 6 months, NT$)
    PAY_AMT1: float = Field(..., ge=0, description="Amount paid last month (NT$)")
    PAY_AMT2: float = Field(..., ge=0, description="Amount paid 2 months ago")
    PAY_AMT3: float = Field(..., ge=0, description="Amount paid 3 months ago")
    PAY_AMT4: float = Field(..., ge=0, description="Amount paid 4 months ago")
    PAY_AMT5: float = Field(..., ge=0, description="Amount paid 5 months ago")
    PAY_AMT6: float = Field(..., ge=0, description="Amount paid 6 months ago")

    @model_validator(mode="after")
    def clean_education_marriage(self):
        """Bin undocumented EDUCATION/MARRIAGE codes — matches training-time cleaning."""
        if self.EDUCATION in (0, 5, 6):
            self.EDUCATION = 4
        if self.MARRIAGE == 0:
            self.MARRIAGE = 3
        return self


class PredictionResponse(BaseModel):
    risk_score:        float = Field(..., description="Probability of default (0–1)")
    risk_band:         str   = Field(..., description="Low Risk / Medium Risk / High Risk")
    action:            str   = Field(..., description="Approve / Review / Decline")
    action_detail:     str   = Field(..., description="Plain-English rationale for the action")
    threshold_used:    float = Field(..., description="Classification threshold applied")
    model_version:     str   = Field(..., description="Model version that made this decision")


class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    model_version: str


class ModelInfoResponse(BaseModel):
    model_name:     str
    version:        str
    training_date:  str
    git_commit:     str
    auc_roc:        float
    gini:           float
    threshold:      float
    recall:         float
    precision:      float
    f1:             float
    approval_rate:  float
    n_train:        int
    n_features:     int
    risk_bands:     dict


# ── Internal helpers ───────────────────────────────────────────────────────────

def _band_and_action(prob: float) -> tuple[str, str, str]:
    """Map a probability to (band, action, detail)."""
    if prob < 0.20:
        return (
            "Low Risk",
            "Approve",
            "Predicted default probability is below the low-risk threshold (0.20). "
            "Automatic approval at standard credit limit.",
        )
    elif prob < 0.50:
        return (
            "Medium Risk",
            "Review",
            "Predicted default probability is in the medium-risk range (0.20–0.50). "
            "Refer to a credit analyst for manual review; consider a reduced credit limit.",
        )
    else:
        return (
            "High Risk",
            "Decline",
            "Predicted default probability exceeds the high-risk threshold (0.50). "
            "Automatic decline. An adverse action notice must be issued under Regulation B.",
        )


def _run_model(customer: CustomerInput) -> float:
    """Apply feature engineering and return the default probability for one customer."""
    raw_df = pd.DataFrame([customer.model_dump()])

    # engineer_features() expects the same raw column structure as training data
    df_eng = engineer_features(raw_df)

    # Select features in the exact order the model was trained on
    if _state["feature_cols"]:
        X = df_eng[_state["feature_cols"]]
    else:
        # Fallback: drop known non-feature columns if metadata isn't loaded
        X = df_eng.drop(columns=[c for c in _DROP_COLS if c in df_eng.columns])

    return float(_state["model"].predict_proba(X)[0, 1])


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    tags=["Operations"],
)
def health_check():
    """Returns service status and whether the model is loaded. Used by load balancers and monitoring."""
    meta = _state["metadata"]
    return HealthResponse(
        status="healthy",
        model_loaded=_state["model"] is not None,
        model_version=meta.get("version", "unknown"),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Score a single credit applicant",
    tags=["Scoring"],
)
def predict(customer: CustomerInput):
    """
    Accept raw customer features, apply feature engineering, and return:
    - **risk_score** — predicted probability of default (0–1)
    - **risk_band** — Low Risk / Medium Risk / High Risk
    - **action** — Approve / Review / Decline
    - **action_detail** — plain-English reason for the action

    The operational threshold is **0.30** (documented in `docs/threshold_policy.md`).
    """
    if _state["model"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Check service startup logs.",
        )

    request_id = uuid.uuid4().hex[:8]
    logger.info(
        "[%s] /predict  AGE=%d  LIMIT_BAL=%.0f  PAY_0=%d  PAY_2=%d",
        request_id, customer.AGE, customer.LIMIT_BAL, customer.PAY_0, customer.PAY_2,
    )

    try:
        prob = _run_model(customer)
    except Exception as exc:
        logger.exception("[%s] Model inference failed: %s", request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model inference error. Contact the model owner.",
        ) from exc

    band, action, detail = _band_and_action(prob)
    meta = _state["metadata"]

    # Audit log — every credit decision must be traceable in a regulated environment
    logger.info(
        "[%s] DECISION  score=%.4f  band=%s  action=%s  model_version=%s  threshold=%.2f",
        request_id, prob, band, action, meta.get("version", "1.0"), _THRESHOLD,
    )

    return PredictionResponse(
        risk_score=round(prob, 4),
        risk_band=band,
        action=action,
        action_detail=detail,
        threshold_used=_THRESHOLD,
        model_version=meta.get("version", "1.0"),
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model metadata and training provenance",
    tags=["Operations"],
)
def model_info():
    """
    Returns the full training provenance for the deployed model:
    version, training date, git commit, and performance metrics.

    Downstream systems should call this endpoint to verify they are
    consuming the expected model version before making credit decisions.
    """
    if not _state["metadata"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata not available.",
        )

    m = _state["metadata"]
    return ModelInfoResponse(
        model_name=    m.get("model_name",    "credit_risk_xgb"),
        version=       m.get("version",       "1.0"),
        training_date= m.get("training_date", "unknown"),
        git_commit=    m.get("git_commit",    "unknown"),
        auc_roc=       m.get("auc_roc",       0.0),
        gini=          m.get("gini",          0.0),
        threshold=     m.get("threshold",     _THRESHOLD),
        recall=        m.get("recall",        0.0),
        precision=     m.get("precision",     0.0),
        f1=            m.get("f1",            0.0),
        approval_rate= m.get("approval_rate", 0.0),
        n_train=       m.get("n_train",       0),
        n_features=    m.get("n_features",    0),
        risk_bands=    m.get("risk_bands",    {}),
    )
