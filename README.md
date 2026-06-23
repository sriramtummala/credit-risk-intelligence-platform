# Credit Risk Intelligence Platform

An end-to-end ML platform for predicting credit card default probability, with explainability, GenAI/RAG governance, and production-grade monitoring — built to demonstrate readiness for quantitative risk and AI/ML roles in financial services.

---

## Problem Statement

### Business Problem

Financial institutions face significant losses when credit card clients fail to make payments. Accurately predicting **Probability of Default (PD)** is critical for:

- **Risk Mitigation** — flag high-risk customers before default occurs
- **Operational Efficiency** — automate credit review for low-risk applicants
- **Regulatory Compliance** — meet model governance standards (SR 11-7, Basel III)

### Technical Objective

Build a production-grade intelligence platform that:

1. **Predicts Default** — ML model trained on UCI Default of Credit Card Clients dataset to predict next-month default
2. **Explains Risk** — SHAP-based transparency into why a customer is flagged high-risk
3. **Governs Models** — GenAI/RAG assistant that lets stakeholders query model documentation and risk policies in natural language
4. **Monitors Performance** — data drift and model degradation tracking in a production-like environment

---

## Dataset

**UCI Default of Credit Card Clients**
- 30,000 Taiwanese credit card clients (April–September 2005)
- Target: `default.payment.next.month` (binary: 0/1)
- Features: credit limit, demographics, payment history (6 months), bill amounts, payment amounts
- Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Credit Risk Intelligence Platform               │
└─────────────────────────────────────────────────────────────────┘

  [Raw Data]                                                        
      │                                                             
      ▼                                                             
┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  
│  Data        │───▶│  Feature     │───▶│  Model Training      │  
│  Ingestion   │    │  Engineering │    │  (XGBoost / LR / RF) │  
│  & Validation│    │  Pipeline    │    │  + Cross-Validation  │  
└──────────────┘    └──────────────┘    └──────────┬───────────┘  
                                                    │              
                    ┌───────────────────────────────┘              
                    │                                               
                    ▼                                               
         ┌──────────────────┐    ┌─────────────────────────┐      
         │  Explainability  │    │  Model Registry &        │      
         │  Layer (SHAP)    │    │  Versioning (MLflow)     │      
         └──────────────────┘    └─────────────────────────┘      
                    │                          │                   
                    ▼                          ▼                   
         ┌──────────────────┐    ┌─────────────────────────┐      
         │  REST API        │    │  Monitoring &            │      
         │  (FastAPI)       │    │  Drift Detection         │      
         └──────────────────┘    │  (Evidently AI)          │      
                    │            └─────────────────────────┘      
                    ▼                                               
         ┌──────────────────┐                                      
         │  GenAI/RAG       │                                      
         │  Risk Assistant  │                                      
         │  (LangChain +    │                                      
         │   Claude API)    │                                      
         └──────────────────┘                                      
```

---

## Role Alignment — Citi AI/ML Capabilities

| # | Capability | Platform Component |
|---|-----------|-------------------|
| 1 | Credit risk modeling (PD) | XGBoost classifier on UCI dataset |
| 2 | Feature engineering for financial data | Payment ratio features, delinquency streaks, utilization rates |
| 3 | Model explainability & transparency | SHAP waterfall plots, global feature importance |
| 4 | Regulatory model governance (SR 11-7) | Model card, validation report, MLflow lineage |
| 5 | GenAI / LLM integration | RAG assistant over risk policy documents |
| 6 | Production ML pipelines | Scikit-learn + MLflow pipelines |
| 7 | Data drift & model monitoring | Evidently AI dashboards |
| 8 | API design for model serving | FastAPI inference endpoint |
| 9 | Statistical validation | KS test, ROC-AUC, Gini coefficient, PSI |
| 10 | Cross-functional communication | Automated PDF risk report (business-readable) |

---

## Project Structure

```
credit-risk-intelligence-platform/
├── data/
│   ├── raw/                    # Original UCI dataset
│   └── processed/              # Feature-engineered datasets
├── notebooks/
│   ├── 01_eda.ipynb            # Day 1: Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_explainability.ipynb
│   └── 05_monitoring.ipynb
├── src/
│   ├── data/                   # Data loading & validation
│   ├── features/               # Feature engineering
│   ├── models/                 # Training, evaluation
│   ├── api/                    # FastAPI app
│   └── rag/                    # GenAI risk assistant
├── models/                     # Saved model artifacts
├── reports/
│   ├── figures/                # SHAP plots, drift reports
│   └── model_card.md           # SR 11-7-aligned model documentation
├── requirements.txt
└── README.md
```

---

## Model Performance Comparison

> Held-out test set — 20% stratified split (6,000 rows), `random_state=42`. Recall is the priority metric (False Negative costs ~10× a False Positive — see [docs/credit_risk_business_framing.md](docs/credit_risk_business_framing.md)).

| Model | Features | Precision | Recall | F1 | AUC-ROC | Gini | Train (s) |
|-------|----------|-----------|--------|----|---------|------|-----------|
| Rule: PAY_0 ≥ 2 | Raw | — | — | — | — | — | — |
| Logistic Regression | Raw | — | — | — | — | — | — |
| Logistic Regression | + Engineered | — | — | — | — | — | — |
| Random Forest | + Engineered | — | — | — | — | — | — |
| **XGBoost** ✓ | + Engineered | — | — | — | — | — | — |

*Run `notebooks/04_baseline_model.ipynb` then `notebooks/05_model_comparison.ipynb` to populate. XGBoost selected as primary model; Logistic Regression retained as regulatory challenger.*
