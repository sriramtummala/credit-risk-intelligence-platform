# Model Explainability Summary — Credit Default Classifier v1

> **Audience:** Non-technical Risk Managers, Credit Policy Committee, Internal Audit  
> **SR 11-7 alignment:** Model Risk Management — Explainability & Transparency  
> **Explainability method:** SHAP (SHapley Additive exPlanations) via TreeExplainer  
> **Last updated:** 2026-06-29  
> **Model reference:** XGBoost Champion (see `docs/model_card.md`)

---

## 1. What SHAP Tells Us

SHAP assigns every feature a numeric score for each individual prediction that answers:  
**"How much did this specific data point push the model's decision toward or away from default?"**

- A **positive SHAP value** means the feature *increased* the predicted default probability.  
- A **negative SHAP value** means the feature *decreased* the predicted default probability.  
- The magnitude shows how strongly the feature influenced that decision.

This makes SHAP legally meaningful: the features with the highest positive SHAP values for a declined applicant are the factual basis for the Adverse Action Notice under Regulation B.

---

## 2. What Drives This Model — Plain English

### Top 5 Global Drivers (across all 6,000 test customers)

| Rank | Feature | Plain-English Meaning | Effect | Mean SHAP |
|------|---------|----------------------|--------|-----------|
| 1 | **max_delinquency** *(engineered)* | Worst payment delay ever recorded | Higher = more likely to default | 0.3891 |
| 2 | **PAY_0** | Payment status last month (0=on time, 2=2-month delay…) | Higher = more likely to default | 0.2724 |
| 3 | **total_delinquencies** *(engineered)* | Total months with any payment delay | Higher = more likely to default | 0.2063 |
| 4 | **PAY_AMT1** | Amount actually paid last month | Higher (paying more) = *less* likely to default | 0.1162 |
| 5 | **util_ratio_m2** *(engineered)* | Credit utilisation ratio 2 months ago | Higher = more likely to default | 0.1019 |

> **Key finding:** Features 1, 3, and 5 are **engineered features** built from raw payment and balance columns — they do not exist in the raw dataset. Their presence in the top 5 global drivers directly validates the feature engineering work and demonstrates that derived signals carry information beyond what the raw variables express individually.

### Key Patterns in Plain English

1. **Payment behaviour dominates.** Three of the top five features measure payment delay history. A customer who has ever missed a payment by 2+ months is dramatically more likely to default, regardless of credit limit or income level. The *worst-ever* delay (`max_delinquency`) is more predictive than *current* delay (`PAY_0`) — past behaviour is the strongest signal.

2. **Paying more is protective.** `PAY_AMT1` has a *negative* SHAP direction — customers making larger actual payments are statistically safer. Someone actively paying down balances is unlikely to be heading toward default.

3. **Credit utilisation adds independent signal.** Even after controlling for payment history, a high balance-to-limit ratio further increases risk. A customer who is both delinquent *and* maxed out represents the highest-risk profile.

4. **Credit limit on its own is not the whole story.** A higher `LIMIT_BAL` is modestly protective (it indicates the origination process already assessed this customer as creditworthy) but is outweighed by recent payment behaviour.

---

## 3. Example Individual Explanation

### Highest-Risk Applicant in Test Set

> **Decision:** DECLINE — predicted default probability **97.6%**

| Reason | Feature | Customer Value | SHAP Impact |
|--------|---------|----------------|-------------|
| **Reason 1** | Recent payment delay (`PAY_0`) | 3 months past due | +0.888 |
| **Reason 2** | Worst-ever delinquency (`max_delinquency`) | 3 months past due | +0.690 |
| **Reason 3** | Historical utilisation pattern (`util_ratio_m5`) | Elevated | +0.452 |

**Adverse Action Notice draft text:**  
*"Your application for credit was denied for the following principal reasons: (1) Recent delinquency on existing accounts; (2) Prior payment delinquency history; (3) High ratio of credit balances to credit limits."*

> **Note for reviewers:** This customer's actual outcome in the dataset was **non-default** — a False Positive. This illustrates the cost trade-off documented in `docs/threshold_policy.md`: at t=0.30, the model declines some creditworthy customers to achieve the 80%+ recall needed to protect the portfolio. The cost-model analysis shows this trade-off saves $715,000 relative to the naïve t=0.50 default.

---

## 4. What the Model Does NOT Use

The following are **excluded from the feature set** and cannot appear in SHAP outputs:

- Age (ECOA-protected characteristic)
- Sex / Gender (ECOA-protected characteristic)
- Marital status (ECOA-protected characteristic)
- National origin (ECOA-protected characteristic)

All inputs are exclusively financial and behavioural. A formal disparate impact analysis across protected classes is required before production deployment (see `docs/model_card.md` Section 7).

---

## 5. Model Limitations — Honest Assessment

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Cross-sectional data snapshot | Cannot detect customers on a recovery trajectory | Incorporate payment trend direction; monitor cohort |
| Taiwan 2005 training data | Coefficient magnitudes may not transfer to a US portfolio | Recalibrate on US data before deployment |
| AUC-ROC = 0.77 | ~23% of decisions are uncertain in expectation | Accepted and quantified in the cost model; monitor monthly |
| Engineered features reduce raw-variable interpretability | Harder to explain `max_delinquency` to a regulator vs `PAY_0` | Feature construction logic documented in notebook comments |

---

## 6. Compliance Summary

| Requirement | How this model meets it |
|-------------|------------------------|
| **SR 11-7 — Model Transparency** | SHAP TreeExplainer provides mathematically exact attribution for every prediction; both global and individual explanations are on demand |
| **Regulation B — Adverse Action Reasons** | Top-3 positive SHAP values for any declined applicant serve as factual, feature-level reason codes |
| **Fair Lending — Disparate Impact** | Protected characteristics excluded from features; formal disparate impact test required pre-deployment |
| **Model Monitoring** | SHAP rank-order of top drivers should be monitored quarterly; a shift would indicate population drift |

---

## 7. Glossary for Non-Technical Readers

| Term | Meaning |
|------|---------|
| SHAP value | A number showing how much one feature pushed the model's prediction toward or away from default for a specific customer |
| Base rate | Where the model starts before looking at any individual's data (~50% for this model) |
| Waterfall plot | A chart showing step-by-step how each feature moves a single customer's score from the base rate to the final prediction |
| Adverse Action Notice | The legally required letter explaining why credit was denied |
| SR 11-7 | Federal Reserve guidance requiring banks to formally validate and document all models used in risk management |
| ECOA | Equal Credit Opportunity Act — prohibits credit discrimination on protected characteristics |

---

*This is a draft artifact produced during model development. It does not constitute an approved Adverse Action Notice template or a regulatory submission. Review by Compliance and Legal is required before production use.*
