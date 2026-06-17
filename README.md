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

## 30-Day Build Roadmap

| Days | Focus | Deliverable |
|------|-------|-------------|
| 1–2 | Foundations & EDA | README, dataset exploration, class imbalance analysis |
| 3–5 | Feature Engineering | Payment behavior features, utilization ratios |
| 6–8 | Baseline Modeling | Logistic Regression + XGBoost, ROC/AUC/Gini |
| 9–11 | Explainability | SHAP global + local explanations |
| 12–14 | Model Governance | Model card, MLflow tracking |
| 15–18 | API + Deployment | FastAPI serving endpoint |
| 19–22 | Monitoring | Evidently drift dashboard |
| 23–27 | GenAI/RAG | LangChain RAG over SR 11-7 policy docs |
| 28–30 | Interview Prep | End-to-end demo, talking points |

---

## Quick Start (Day 1)

```bash
# Clone the repo
git clone https://github.com/<your-username>/credit-risk-intelligence-platform.git
cd credit-risk-intelligence-platform

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Launch Day 1 EDA notebook
jupyter lab notebooks/01_eda.ipynb
```

---

## Key Interview Talking Points

- **Why UCI dataset?** — Industry-standard benchmark for credit default, clean enough to focus on modeling decisions rather than data cleaning, but rich enough to demonstrate real feature engineering
- **Why XGBoost over LR for PD?** — Captures non-linear payment behavior patterns; LR kept as interpretable baseline per SR 11-7 requirements
- **How does this map to Basel III?** — PD model output feeds into EL = PD × LGD × EAD framework; SHAP explanations support adverse action notices
- **What is SR 11-7?** — Fed/OCC guidance on model risk management: requires independent validation, ongoing monitoring, and model inventory

---

*Built as part of a 30-day aggressive preparation plan for AI/ML roles in financial services.*
