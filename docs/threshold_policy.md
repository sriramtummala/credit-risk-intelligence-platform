# Threshold & Risk Band Policy — Credit Default Classifier v1

> **Document type:** Business Policy  
> **SR 11-7 alignment:** Model Risk Management — Threshold Selection Rationale  
> **Status:** Draft — requires Credit Risk Committee sign-off before production use  
> **Last updated:** 2026-06-29  
> **Model reference:** XGBoost Champion (see `docs/model_card.md`)

---

## 1. Purpose

This document codifies the classification threshold and risk-band definitions used to translate raw default probabilities into automated credit decisions. It provides the justification required under SR 11-7 for any model-driven decisioning rule.

---

## 2. Recommended Operational Threshold

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Operational threshold** | **0.30** | Minimises total business cost; see Section 4 |
| Default (naïve) threshold | 0.50 | Maximises accuracy on balanced data — inappropriate for imbalanced credit data |
| Max-F1 threshold | ~0.35 | Maximises F1; less conservative than cost-optimal |

### Why 0.30 and not 0.50?

In credit-card underwriting, the asymmetric cost of outcomes makes a 0.50 threshold economically irrational:

- A **False Negative** (approving a customer who defaults) costs the bank the outstanding balance, loss given default recovery expenses, and operational write-off costs — estimated at **$5,000 per missed default**.
- A **False Positive** (declining a creditworthy customer) loses only the foregone annual interest revenue — estimated at **$500 per wrongly declined account**.

The 10:1 cost ratio means it is worth declining 10 good customers to prevent a single missed default.

The **pure cost-minimising** threshold on this dataset is **0.20** ($2,201,500 total cost), but the resulting approval rate of ~29% is operationally untenable for a retail bank. **t = 0.30** is the *constrained optimum*: it achieves Recall ≥ 0.80 and an approval rate of ~46.6% while remaining $715,000 cheaper than the naïve t = 0.50 default.

---

## 3. Risk Band Definitions

| Risk Band | Score Range | Recommended Action | Expected Default Rate |
|-----------|-------------|--------------------|-----------------------|
| **Low Risk** | 0.00 – 0.19 | **Automatic Approval** — issue standard credit limit per product guidelines | 7.7% (observed) |
| **Medium Risk** | 0.20 – 0.49 | **Manual Review** — credit analyst assessment; consider reduced limit, income verification, or collateral request | 15.3% (observed) |
| **High Risk** | 0.50 – 1.00 | **Automatic Decline** — send adverse action notice per Regulation B | 47.4% (observed) |

### Band rationale

| Band | Reasoning |
|------|-----------|
| Low Risk (< 0.20) | Model is highly confident of non-default. Automated approval reduces operational cost and improves customer experience. |
| Medium Risk (0.20–0.50) | Model uncertainty is elevated. Human review adds a control layer before a credit commitment is made. |
| High Risk (> 0.50) | Model predicts default more likely than not. Automatic decline is the prudent default. |

---

## 4. Business Cost Analysis Summary

The table below shows total business cost (FN losses + FP opportunity cost) at each threshold over the held-out test set of 6,000 customers.

| Threshold | Recall | Precision | Approval Rate | Total Cost | Notes |
|-----------|--------|-----------|---------------|-----------|-------|
| 0.20 | 0.899 | 0.280 | 29.1% | $2,201,500 | Pure cost minimum; approval rate untenable |
| **0.30** | **0.806** | **0.333** | **46.6%** | **$2,359,000** | **Recommended — constrained optimum** |
| 0.50 | 0.604 | 0.474 | 71.9% | $3,074,000 | Naïve default; $715K more expensive |
| 0.60 | 0.523 | 0.542 | 78.7% | $3,458,000 | Highest cost; too many missed defaults |

> Full cost curves are in `reports/figures/20_threshold_cost_analysis.png`.  
> Analysis notebook: `notebooks/06_threshold_optimization.ipynb`.

---

## 5. Model Performance at Recommended Threshold (t = 0.30)

| Metric | Value |
|--------|-------|
| Precision | 0.3333 |
| Recall | 0.8056 |
| F1 Score | 0.4715 |
| AUC-ROC | 0.7725 |
| Gini Coefficient | 0.5450 |
| Approval Rate | 46.6% |
| Business Cost (vs t=0.50 baseline) | −$715,000 |

---

## 6. Regulatory & Governance Notes

| Requirement | Compliance approach |
|-------------|---------------------|
| **SR 11-7 — Model Validation** | Threshold selection is documented with a cost-model justification and must be independently validated before production deployment. |
| **Regulation B (ECOA) — Adverse Action** | Any applicant declined via the High Risk band must receive a written notice specifying the principal reasons for denial within 30 days. |
| **Fair Lending** | The risk-band model must be tested for disparate impact across protected classes (age, sex, national origin) before deployment. Results must be logged in the model inventory. |
| **Model Monitoring** | The operational threshold and band boundaries must be re-evaluated quarterly or whenever the portfolio default rate shifts by more than ±2 percentage points. |

---

## 7. Escalation & Override Process

| Scenario | Escalation path |
|----------|----------------|
| Medium Risk applicant with strong income documentation | Credit analyst may approve at reduced limit; decision logged in LOS |
| High Risk applicant — relationship banking exception | Senior Credit Officer sign-off required; logged with override reason code |
| Threshold recalibration request | Model Risk Management team + Credit Policy Committee |

---

## 8. Revision History

| Date | Version | Author | Change |
|------|---------|--------|--------|
| 2026-06-29 | 0.1 | Credit Risk Analytics | Initial draft — Day 11 threshold optimization |

---

*This document is a draft artifact produced during model development. It does not represent an approved Citi policy. A qualified credit risk professional must review and approve all threshold and band definitions before any production use.*
