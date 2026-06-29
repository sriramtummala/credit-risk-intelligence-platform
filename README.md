# Credit Risk Intelligence Platform

<!-- Replace {your-github-username} with your actual GitHub handle for the live CI badge -->
![CI Pipeline](https://github.com/{your-github-username}/credit-risk-intelligence-platform/actions/workflows/ci.yml/badge.svg)

An end-to-end ML platform for predicting credit card default probability, with explainability, SR 11-7 governance, and production-grade deployment — built to demonstrate readiness for quantitative risk and AI/ML roles in financial services.

### At a Glance

| | |
|---|---|
| **Champion model** | XGBoost — AUC-ROC **0.7725**, Gini **0.5450**, Recall **80.6%** at t=0.30 |
| **Business impact** | Threshold optimisation saves **$715K** per 6,000 applicants vs. naive baseline |
| **Feature pipeline** | 24 raw → 51 engineered features (delinquency streaks, utilisation rates, payment trends) |
| **API** | FastAPI service — `/predict`, `/health`, `/model-info`; Pydantic v2 validation; audit logging |
| **Tests** | 27 pytest tests (27/27 passing); GitHub Actions CI on every push |
| **Deployment** | Docker (`python:3.11-slim`); Render-compatible; model trains on first boot |
| **Governance** | SR 11-7 model card, SHAP explainability, adverse action reason codes, 3-tier risk bands |

```bash
# Try it in 3 commands
pip install -r requirements.txt
python main.py --download          # downloads UCI data and trains XGBoost (~30 s)
uvicorn api.main:app --port 8000   # Swagger UI at http://localhost:8000/docs
```

---

## Problem Statement

### Business Problem

Financial institutions face significant losses when credit card clients fail to make payments. Accurately predicting **Probability of Default (PD)** is critical for:

- **Risk Mitigation** — flag high-risk customers before default occurs
- **Operational Efficiency** — automate credit review for low-risk applicants
- **Regulatory Compliance** — meet model governance standards (SR 11-7, Basel III)

### Technical Objective

Build a production-grade intelligence platform that:

1. **Predicts Default** — XGBoost champion model trained on UCI Default of Credit Card Clients dataset
2. **Explains Risk** — SHAP-based transparency into why a customer is flagged high-risk
3. **Governs Models** — SR 11-7-aligned model card, threshold policy, and adverse action reason codes
4. **Serves Predictions** — FastAPI REST service with Pydantic validation and audit-trail logging
5. **Monitors Performance** — data drift and model degradation tracking via Evidently AI

---

## Dataset

**UCI Default of Credit Card Clients**
- 30,000 Taiwanese credit card clients (April–September 2005)
- Target: `default.payment.next.month` (binary: 0/1)
- Class imbalance: 22.1% positive (default)
- Features: credit limit, demographics, 6-month payment history, bill amounts, payment amounts
- Source: UCI ML Repository

---

## Architecture

```
  [UCI Dataset]
       │
       ▼
┌──────────────┐    ┌──────────────┐    ┌─────────────────────────┐
│  Data        │───▶│  Feature     │───▶│  Model Training         │
│  Ingestion   │    │  Engineering │    │  XGBoost (champion)     │
│  (requests)  │    │  24→51 cols  │    │  LR + RF (challengers)  │
└──────────────┘    └──────────────┘    └────────────┬────────────┘
                                                     │
                    ┌────────────────────────────────┘
                    │
                    ▼
         ┌──────────────────┐    ┌─────────────────────────┐
         │  Explainability  │    │  Threshold Optimisation  │
         │  SHAP + Adverse  │    │  PR curve, cost matrix   │
         │  Action Codes    │    │  Risk banding (3 tiers)  │
         └──────────────────┘    └─────────────────────────┘
                    │
                    ▼
         ┌──────────────────┐    ┌─────────────────────────┐
         │  FastAPI Service │    │  SR 11-7 Governance      │
         │  /predict        │    │  Model card, threshold   │
         │  /health         │    │  policy, explainability  │
         │  /model-info     │    │  summary                 │
         └──────────────────┘    └─────────────────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │  Docker / Render │
         │  Cloud Deploy    │
         └──────────────────┘
```

---

## Model Performance

> Held-out test set — 20% stratified split (6,000 rows), `random_state=42`.
> Recall is the priority metric: a missed default costs ~10× a false alarm.
> Champion threshold t=0.30 saves **$715K** vs. the naive t=0.50.

| Model | AUC-ROC | Gini | Precision | Recall | F1 |
|-------|---------|------|-----------|--------|----|
| Rule: PAY_0 ≥ 2 (baseline) | 0.624 | 0.248 | 0.52 | 0.42 | 0.46 |
| Logistic Regression (raw) | 0.716 | 0.432 | 0.52 | 0.61 | 0.56 |
| Logistic Regression (engineered) | 0.729 | 0.458 | 0.53 | 0.63 | 0.58 |
| Random Forest (engineered) | 0.757 | 0.514 | 0.55 | 0.69 | 0.61 |
| **XGBoost (champion) ✓** | **0.7725** | **0.5450** | **0.558** | **0.806** | **0.660** |

*Threshold: t=0.30 | scale_pos_weight=3.52 (accounts for 22.1% class imbalance)*

### Risk Band Summary (champion model, t=0.30)

| Band | Score Range | Observed Default Rate | Action |
|------|-------------|----------------------|--------|
| Low Risk | < 0.20 | 7.7% | Approve |
| Medium Risk | 0.20 – 0.50 | 15.3% | Manual Review |
| High Risk | > 0.50 | 47.4% | Decline |

### Top Predictive Features (SHAP global importance)

| Rank | Feature | Mean \|SHAP\| | Business Meaning |
|------|---------|--------------|-----------------|
| 1 | `max_delinquency` | 0.389 | Worst payment delay in 6 months |
| 2 | `pay_trend` | 0.201 | Improving vs. worsening payment pattern |
| 3 | `utilization_rate` | 0.187 | Bill balance / credit limit |
| 4 | `PAY_0` | 0.164 | Most recent payment status |
| 5 | `pay_consistency` | 0.142 | Volatility in monthly payments |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Git

### Local Development

```bash
# 1. Clone and create virtual environment
git clone <repo-url>
cd credit-risk-intelligence-platform
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements-dev.txt

# 3. Train the model (downloads UCI dataset automatically)
python main.py --download

# 4. Start the API server
uvicorn api.main:app --reload --port 8000

# 5. Run tests
pytest tests/ -v
```

### API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Model metadata
curl http://localhost:8000/model-info

# Score a customer
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 200000, "SEX": 1, "EDUCATION": 1, "MARRIAGE": 2, "AGE": 35,
    "PAY_0": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
    "BILL_AMT1": 50000, "BILL_AMT2": 45000, "BILL_AMT3": 40000,
    "BILL_AMT4": 35000, "BILL_AMT5": 30000, "BILL_AMT6": 25000,
    "PAY_AMT1": 50000, "PAY_AMT2": 45000, "PAY_AMT3": 40000,
    "PAY_AMT4": 35000, "PAY_AMT5": 30000, "PAY_AMT6": 25000
  }'
```

**Example response:**

```json
{
  "risk_score": 0.0412,
  "risk_band": "Low Risk",
  "action": "Approve",
  "threshold_used": 0.30,
  "model_version": "champion_xgb_20250619"
}
```

---

## Deployment

### Option A — Docker (Recommended)

The container trains the model on first boot (downloads the UCI dataset, ~30 s) then starts the FastAPI server. Subsequent runs reuse a mounted model volume to skip training.

#### Build and run

```bash
# Build the image
docker build -t credit-risk-api .

# Run (first boot trains the model; API starts after training)
docker run -p 8000:8000 credit-risk-api

# Persist the model so subsequent runs skip training (~30 s saved)
docker run -p 8000:8000 \
  -v "$(pwd)/models:/app/models" \
  credit-risk-api
```

#### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Port the uvicorn server listens on |

#### Verify the running container

```bash
curl http://localhost:8000/health
# {"status":"healthy","model_loaded":true}

curl http://localhost:8000/model-info
# {"model_type":"XGBClassifier","auc_roc":0.7725,"gini":0.545,"n_features":51,...}
```

### Option B — Render.com (Cloud)

1. Push this repository to GitHub.
2. In Render, click **New → Web Service** and connect your repo.
3. Set the following fields:

| Field | Value |
|-------|-------|
| Environment | Docker |
| Instance Type | Starter (512 MB RAM is sufficient) |
| Port | 8000 |
| Health Check Path | `/health` |

4. Click **Deploy**. Render builds the Docker image, the entrypoint script trains the model on first boot, and the service goes live.

> **Note:** Free-tier Render instances spin down after 15 minutes of inactivity. The `/health` endpoint wakes them in ~30 s.

### Option C — Local Production (no Docker)

```bash
# Install production dependencies only
pip install -r requirements.txt

# Train the model
python main.py --download

# Start the production server (no --reload)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2
```

---

## Project Structure

```
credit-risk-intelligence-platform/
├── api/
│   └── main.py                 # FastAPI app — /health, /predict, /model-info
├── data/
│   ├── raw/                    # Downloaded UCI dataset (gitignored)
│   └── processed/              # Feature-engineered parquet (gitignored)
├── docker/
│   └── entrypoint.sh           # Container startup: train if needed → start API
├── docs/
│   ├── model_card.md           # SR 11-7-aligned governance document (14 KB)
│   ├── threshold_policy.md     # Threshold optimization rationale & risk bands
│   └── model_explainability_summary.md  # Plain-English SHAP narrative
├── models/                     # Saved model artifacts (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_business_insights.ipynb
│   ├── 05_model_comparison.ipynb
│   ├── 06_threshold_optimization.ipynb  # PR curve, cost analysis, risk bands
│   └── 07_shap_explainability.ipynb     # SHAP beeswarm, waterfall, reason codes
├── reports/
│   └── figures/                # Generated plots (gitignored)
├── src/
│   ├── data_ingestion.py       # UCI download + parse
│   ├── feature_engineering.py  # 24→51 feature pipeline
│   ├── train.py                # Full training pipeline, saves model artifacts
│   └── evaluate.py             # Metrics: AUC, Gini, business cost, risk bands
├── tests/
│   └── test_api.py             # 27 pytest tests (integration + validation)
├── main.py                     # CLI entry point: python main.py --download
├── requirements.txt            # Production deps (pinned)
├── requirements-dev.txt        # Dev extras: pytest, jupyter, shap, evidently
├── Dockerfile                  # python:3.11-slim, trains on first boot
└── .dockerignore
```

---

## Role Alignment — Citi AI/ML Capabilities

| # | Capability | Platform Component |
|---|-----------|-------------------|
| 1 | Credit risk modeling (PD) | XGBoost champion (AUC=0.7725, Gini=0.545) |
| 2 | Feature engineering for financial data | Payment ratios, delinquency streaks, utilization rates (24→51 cols) |
| 3 | Model explainability & transparency | SHAP waterfall plots, global importance, adverse action codes |
| 4 | Regulatory model governance (SR 11-7) | Model card, threshold policy, explainability summary |
| 5 | Threshold optimization | PR curve analysis, cost matrix, 3-tier risk banding |
| 6 | Production ML pipelines | `src/train.py` + `src/evaluate.py` CLI pipeline |
| 7 | REST API for model serving | FastAPI + Pydantic v2, audit-trail logging, 27-test suite |
| 8 | Containerized deployment | Docker + `entrypoint.sh`, Render-compatible |
| 9 | Statistical validation | KS test, ROC-AUC, Gini coefficient, PSI thresholds |
| 10 | GenAI / LLM integration | RAG assistant over risk policy docs (Days 23–27) |

---

## CI/CD Pipeline

Every push and pull request against `main` triggers the GitHub Actions CI pipeline, which acts as a quality gate before any code reaches production.

### What the pipeline checks

| Step | Tool | What it validates |
|------|------|-------------------|
| Linting | `flake8` | No syntax errors, undefined names, or runtime-breaking issues |
| Model training | `python main.py` | Training pipeline runs end-to-end (downloads UCI data, fits XGBoost) |
| Tests | `pytest` | All 27 API tests pass (validation, health, predict, model-info) |

### Workflow file

`.github/workflows/ci.yml` runs on `ubuntu-latest` with Python 3.11:

```yaml
# Abbreviated — see the full file at .github/workflows/ci.yml
jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt pytest httpx flake8
      - run: flake8 . --select=E9,F63,F7,F82 --exclude=.venv,notebooks
      - run: python main.py --download   # trains the model (~30 s)
      - run: pytest tests/ -v --tb=short
```

### Interview talking point

> "I implemented a CI/CD pipeline using GitHub Actions. Every push triggers linting and the full 27-test suite. The pipeline trains the model from scratch in CI — proving the end-to-end pipeline is reproducible, not just the API. A red build blocks merges, so no broken code reaches production."

---

## Governance & Compliance

This platform is built with SR 11-7 Model Risk Management in mind:

- **[Model Card](docs/model_card.md)** — full governance document covering intended use, training data, performance metrics, monitoring triggers, ethical considerations, and a compliance checklist
- **[Threshold Policy](docs/threshold_policy.md)** — documents the t=0.30 recommendation with cost-benefit analysis and business rationale
- **[Explainability Summary](docs/model_explainability_summary.md)** — plain-English SHAP narrative for non-technical Risk Managers and regulators
- **[Release Notes v1.0](docs/release_notes_v1.md)** — full changelog, known issues, deployment instructions, and v1.1 roadmap

**Monitoring triggers (automatic retraining):**

| Metric | Retrain Threshold |
|--------|------------------|
| AUC-ROC | < 0.72 |
| Gini | < 0.45 |
| Approval rate | ±10 percentage points vs. baseline |
| PSI (Population Stability Index) | > 0.25 |
