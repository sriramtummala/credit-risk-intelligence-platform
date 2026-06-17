# Model Card — Credit Default Prediction Model

> **SR 11-7 Alignment:** This document serves as the model inventory record required under the Federal Reserve's SR 11-7 guidance on model risk management.

---

## Model Overview

| Field | Value |
|-------|-------|
| Model Name | Credit Default Classifier v1 |
| Model Type | Binary Classification |
| Version | TBD |
| Owner | TBD |
| Last Updated | TBD |
| Status | In Development |

---

## Intended Use

**Primary use:** Predict probability of credit card payment default in the next billing cycle.

**Intended users:** Credit risk analysts, automated underwriting systems.

**Out-of-scope uses:** This model is not intended for real-time transaction fraud detection or loan origination decisions without additional validation.

---

## Training Data

- **Dataset:** UCI Default of Credit Card Clients
- **Size:** 30,000 records
- **Time period:** April–September 2005
- **Geography:** Taiwan
- **Target:** `default.payment.next.month` (binary: 0/1)
- **Class balance:** ~77.9% no default / ~22.1% default

---

## Model Architecture

*To be completed during Days 6–8 (model training phase).*

---

## Performance Metrics

*To be completed during Days 6–8.*

| Metric | Value |
|--------|-------|
| AUC-ROC | TBD |
| Gini Coefficient | TBD |
| KS Statistic | TBD |
| Precision (threshold=0.5) | TBD |
| Recall (threshold=0.5) | TBD |
| F1 Score | TBD |

---

## Explainability

*SHAP analysis to be completed during Days 9–11.*

---

## Known Limitations & Biases

- Dataset is from Taiwan 2005 — demographic and economic patterns may not generalize to current US market
- Class imbalance (~22% default) requires careful threshold selection
- EDUCATION and MARRIAGE contain undocumented categories (0, 5, 6) that were binned

---

## Model Governance (SR 11-7)

| Requirement | Status |
|-------------|--------|
| Independent validation | Pending |
| Model inventory registration | This document |
| Ongoing monitoring plan | Evidently AI (Days 19–22) |
| Adverse action explainability | SHAP (Days 9–11) |
| Challenger model comparison | Pending |
