# Release Notes — v1.0.0-production

**Release date:** 2026-06-29  
**Status:** Production-ready  
**Model:** XGBoost champion (champion_xgb_20260629)

---

## Summary

Initial production release of the **Credit Risk Intelligence Platform** — an end-to-end ML system that scores credit card applicants for probability of default and serves decisions through a FastAPI REST service with full audit logging, Pydantic validation, and Docker containerization.

---

## What's New

### Machine Learning

| Item | Detail |
|------|--------|
| Champion model | XGBoost Classifier (`n_estimators=300`, `max_depth=6`, `learning_rate=0.05`) |
| Class imbalance | Handled via `scale_pos_weight=3.52` (22.1% positive rate in training data) |
| AUC-ROC | **0.7725** on held-out test set (6,000 rows, stratified 80/20 split) |
| Gini coefficient | **0.5450** |
| Recall at t=0.30 | **80.6%** — the priority metric (missed defaults cost ~10× false alarms) |
| Cost saving | t=0.30 saves **$715,000** vs. naive t=0.50 on a 6,000-row test set |
| Feature engineering | 24 raw features → 51 model features (payment ratios, delinquency streaks, utilization rates) |
| Challenger models | Logistic Regression (AUC 0.729) and Random Forest (AUC 0.757) retained for regulatory comparison |

### API Service (`api/main.py`)

- **`POST /predict`** — accepts 23 raw customer features, runs feature engineering, returns:
  - `risk_score` (float, 0–1)
  - `risk_band` (Low / Medium / High)
  - `action` (Approve / Review / Decline)
  - `action_detail` (plain-English rationale, Regulation B compliant)
  - `threshold_used` and `model_version` for full auditability
- **`GET /health`** — service liveness check; used by Docker `HEALTHCHECK` and load balancers
- **`GET /model-info`** — full training provenance: version, git commit, training date, all metrics
- **Pydantic v2 validation** with 422 responses for out-of-range inputs (AGE < 18, LIMIT_BAL ≤ 0, PAY codes > 8, etc.)
- **Structured audit logging** — every `/predict` call logs `request_id`, key input features, score, band, action, model version, and threshold
- **Swagger UI** at `/docs` with two pre-filled example payloads (low-risk and high-risk) — click "Try it out" to run instantly
- **`@model_validator`** reproduces training-time EDUCATION/MARRIAGE binning to prevent serving/training skew

### Infrastructure

- **`Dockerfile`** — `python:3.11-slim`; trains the model on first boot if no artifact is present; `HEALTHCHECK` every 30 s
- **`docker/entrypoint.sh`** — idempotent startup: checks for `models/champion_model.joblib` before deciding to train
- **GitHub Actions CI** (`.github/workflows/ci.yml`) — on every push to `main`:
  1. `flake8` linting (E9, F63, F7, F82 — runtime-breaking errors only)
  2. Full training pipeline (`python main.py --download`)
  3. 27-test pytest suite (integration + validation)
- **Pinned production deps** in `requirements.txt`; dev extras in `requirements-dev.txt`

### Governance & Explainability

- **Model card** (`docs/model_card.md`, 14 KB) — SR 11-7-aligned; covers intended use, training data, metrics, monitoring triggers, ethical considerations, and a compliance checklist
- **Threshold policy** (`docs/threshold_policy.md`) — documents t=0.30 recommendation with PR curve analysis and cost-benefit rationale
- **SHAP explainability** (`notebooks/07_shap_explainability.ipynb`) — global beeswarm, per-feature bar chart, individual waterfall plot; top driver is `max_delinquency` (mean |SHAP| = 0.389)
- **Explainability summary** (`docs/model_explainability_summary.md`) — plain-English narrative for non-technical Risk Managers
- **3-tier risk banding**: Low (<0.20, 7.7% observed default), Medium (0.20–0.50, 15.3%), High (>0.50, 47.4%)

---

## Monitoring Triggers (Automatic Retraining)

| Metric | Threshold |
|--------|-----------|
| AUC-ROC | < 0.72 |
| Gini coefficient | < 0.45 |
| Approval rate drift | ±10 percentage points vs. baseline |
| Population Stability Index (PSI) | > 0.25 |

---

## Known Issues & Limitations

| Issue | Severity | Mitigation |
|-------|----------|-----------|
| Training data is 2005 Taiwan credit card data | High | Model must be validated against any target population before production use; document in model card |
| Static threshold (t=0.30) not adaptive | Medium | Threshold should be re-optimised whenever the class prevalence shifts >5 pp |
| No API rate limiting or authentication | High | Deploy behind an API gateway (e.g., AWS API Gateway, Nginx) before public exposure |
| Single-model serving (no A/B or canary) | Medium | Challenger models (LR, RF) are available in the codebase; A/B framework is a v1.1 candidate |
| No database persistence for decisions | Low | Audit log is written to stdout only; ingest into a SIEM or decision database for regulated use |
| Docker model volume not persisted by default | Low | Mount `-v $(pwd)/models:/app/models` to avoid ~30 s retraining on every container restart |

---

## Deployment Instructions

### Option A — Local

```bash
git clone <repo-url>
cd credit-risk-intelligence-platform

python -m venv .venv
# Windows: .venv\Scripts\activate  |  macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
python main.py --download          # downloads UCI dataset and trains champion model
uvicorn api.main:app --port 8000
```

### Option B — Docker

```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 -v "$(pwd)/models:/app/models" credit-risk-api
```

First boot trains the model (~30 s). Subsequent runs with the volume mount skip training.

### Option C — Render.com

1. Push repository to GitHub
2. Render → New → Web Service → connect repo
3. Environment: Docker | Port: 8000 | Health Check: `/health`
4. Deploy

### Verification

```bash
# Service health
curl http://localhost:8000/health

# Model provenance
curl http://localhost:8000/model-info

# Score a high-risk applicant
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"LIMIT_BAL":20000,"SEX":2,"EDUCATION":2,"MARRIAGE":1,"AGE":24,
       "PAY_0":2,"PAY_2":2,"PAY_3":-1,"PAY_4":-1,"PAY_5":-2,"PAY_6":-2,
       "BILL_AMT1":18900,"BILL_AMT2":19200,"BILL_AMT3":19100,
       "BILL_AMT4":18700,"BILL_AMT5":17900,"BILL_AMT6":17200,
       "PAY_AMT1":0,"PAY_AMT2":500,"PAY_AMT3":500,
       "PAY_AMT4":500,"PAY_AMT5":500,"PAY_AMT6":500}'
# Expected: {"risk_score": ~0.85+, "action": "Decline", "risk_band": "High Risk", ...}

# Validation rejection (AGE below minimum)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"LIMIT_BAL":50000,"SEX":1,"EDUCATION":2,"MARRIAGE":2,"AGE":16,...}'
# Expected: HTTP 422 with field-level error detail
```

---

## Test Suite

```bash
pytest tests/ -v
# 27 tests — Health (3), ModelInfo (4), PredictSuccess (8), PredictValidation (12)
# Expected: 27 passed in ~2 s
```

---

## Component Versions

| Component | Version |
|-----------|---------|
| Python | 3.11 |
| FastAPI | 0.137.1 |
| Pydantic | 2.13.4 |
| XGBoost | 3.2.0 |
| scikit-learn | 1.9.0 |
| uvicorn | 0.49.0 |
| pytest | 9.1.1 |

---

## Roadmap — v1.1 Candidates

- [ ] GenAI/RAG risk assistant (LangChain + Claude API) over model card and threshold policy
- [ ] Evidently AI monitoring dashboard — data drift and PSI tracking
- [ ] A/B model serving (champion vs. challenger)
- [ ] Decision database persistence (PostgreSQL + SQLAlchemy)
- [ ] API authentication (JWT / API key via FastAPI `Depends`)
- [ ] Async batch scoring endpoint for bulk applicant lists
