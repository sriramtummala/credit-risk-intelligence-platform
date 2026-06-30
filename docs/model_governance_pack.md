# Model Governance Pack

**Model:** Credit Risk Intelligence Platform — XGBoost Champion Scorer  
**Version:** 1.0.0-production  
**Status:** Draft — requires independent model validation before regulated use  
**Owner:** Credit Risk Analytics  
**Last updated:** 2026-06-30

---

## 1. Purpose

This document serves as the primary governance reference for the Credit Risk Intelligence Platform. It consolidates model assumptions, limitations, methodology justification, implementation controls, and links to all supporting governance artefacts in a single navigable pack.

The pack is structured to satisfy the four pillars of **SR 11-7 Model Risk Management**:

| SR 11-7 Pillar | Where addressed in this pack |
|----------------|------------------------------|
| Conceptual soundness | §4 — Model Assumptions; §6 — Challenger Comparison |
| Ongoing monitoring | §8 — Document Index (monitoring policy, drift concepts) |
| Outcomes analysis | §6 — Performance comparison; §8 — model card metrics |
| Effective challenge | §6 — Challenger Model Comparison |

---

## 2. Document Index

All governance artefacts for this platform are listed below. Together they constitute the complete model documentation package for a model risk review.

| Document | Path | Contents |
|----------|------|----------|
| Model Card | [`docs/model_card.md`](model_card.md) | Intended use, training data, metrics, monitoring triggers, ethical considerations, SR 11-7 checklist |
| Threshold Policy | [`docs/threshold_policy.md`](threshold_policy.md) | t=0.30 justification, PR curve analysis, cost matrix, risk banding definitions |
| Monitoring Policy | [`docs/monitoring_policy.md`](monitoring_policy.md) | PSI thresholds, data quality limits, performance SLAs, incident response plan |
| Drift Concepts | [`docs/drift_concepts.md`](drift_concepts.md) | Feature / target / concept drift definitions, PSI formula, simulation results |
| Explainability Summary | [`docs/model_explainability_summary.md`](model_explainability_summary.md) | SHAP global and local explanations, adverse action codes, plain-English narrative |
| Release Notes | [`docs/release_notes_v1.md`](release_notes_v1.md) | What's new in v1.0.0, known issues, deployment instructions, v1.1 roadmap |
| Architecture | [`docs/architecture.md`](architecture.md) | Component diagram, design decisions, training/serving decoupling |
| **This document** | `docs/model_governance_pack.md` | Assumptions, limitations, challenger comparison, implementation controls |

**Monitoring reports** (generated weekly by `notebooks/09_evidently_monitoring.ipynb`):

| Report | Path | Contents |
|--------|------|----------|
| Feature Drift | `reports/data_drift.html` | Per-feature PSI and KS visualisations |
| Data Quality | `reports/data_quality.html` | Missing values, range checks, statistical summaries |
| Model Performance | `reports/model_performance.html` | AUC, precision, recall vs. reference population |

---

## 3. Governance Checklist

The following checklist maps SR 11-7 requirements to the artefacts in this pack. Each item must be signed off by the Model Owner and, where indicated, by an independent Model Risk reviewer.

| Requirement | Addressed | Evidence |
|-------------|-----------|---------|
| **Conceptual soundness** | | |
| Model logic is consistent with credit risk theory | ✓ | §4 — Assumptions; SHAP confirms payment delinquency is top driver |
| Training data is appropriate for the use case | ✓ | Model Card §2 — Data |
| Feature engineering is documented and reproducible | ✓ | `src/features/engineer.py`; Model Card §2.2 |
| Alternative methodologies were evaluated | ✓ | §6 — Challenger Comparison |
| **Ongoing monitoring** | | |
| Quantitative performance thresholds are defined | ✓ | Monitoring Policy §5 |
| Data quality monitoring is in place | ✓ | Monitoring Policy §3 |
| Drift detection methodology is documented | ✓ | Drift Concepts; Monitoring Policy §4 |
| Retraining trigger conditions are defined | ✓ | Monitoring Policy §7 |
| **Outcomes analysis** | | |
| Model performance on holdout data is documented | ✓ | Model Card §4; Release Notes |
| Business cost impact is quantified | ✓ | Threshold Policy §3 |
| Risk band calibration is validated | ✓ | Model Card §4.3 |
| **Effective challenge** | | |
| Challenger models were compared | ✓ | §6 — Challenger Comparison |
| Champion selection is justified on multiple criteria | ✓ | §6.2 — Selection Rationale |
| Limitations are documented honestly | ✓ | §5 — Model Limitations |
| **Implementation controls** | | |
| Input validation is enforced at the API boundary | ✓ | §7.1 — Pydantic v2 schema enforcement |
| Training/serving feature parity is guaranteed | ✓ | §7.2 — Shared `engineer_features()` |
| Every decision is auditable | ✓ | §7.4 — Audit-trail logging |
| Regulatory adverse action codes are supported | ✓ | §7.5 — SHAP-based reason codes |

**Items requiring independent sign-off before production use:**

- [ ] Independent model validation of champion vs. challengers
- [ ] Legal / Compliance review of Regulation B adverse action workflow
- [ ] IT Security review of API authentication and rate limiting (see Release Notes — Known Issues)
- [ ] Credit Risk Committee approval of t=0.30 threshold and cost model

---

## 4. Model Assumptions

These are the conditions that must hold for the model to produce valid decisions. Each assumption represents a risk to monitor; if an assumption is violated, the model may perform worse than its backtest metrics suggest.

### 4.1 Data Representativeness

**Assumption:** The UCI Taiwan credit card dataset (2005, 30,000 cardholders) is a structurally valid proxy for the target population in a stable economic environment.

**Basis:** The dataset exhibits realistic credit risk characteristics — a 22.1% default rate, strong predictive signal in payment delinquency history (`PAY_0`, mean |SHAP| = 0.389), and credit utilisation patterns consistent with published credit risk literature.

**Risk if violated:** If the target population differs materially in credit culture, product structure, or regulatory environment, the model's probability estimates will be miscalibrated. Independent validation on target-population data is required before any regulated use.

**Monitoring signal:** Feature PSI across all 23 input variables. PSI > 0.25 on two or more key features simultaneously suggests the production population may have drifted from the training reference.

---

### 4.2 Economic Stability

**Assumption:** The model was calibrated in a stable economic period. It does not incorporate macroeconomic covariates (unemployment rate, interest rates, GDP growth).

**Consequence:** The model cannot predict defaults caused by systemic economic shocks — a recession that raises the base default rate from 22% to 40% will cause the model's probability estimates to be systematically overconfident (it will underestimate risk for all applicants).

**Monitoring signal:** Target drift — if the realised default rate in the approved population shifts by more than ±5 percentage points from the 22.1% baseline, economic regime change is suspected.

**Mitigation available:** The classification threshold (t=0.30) can be tightened manually in response to observed target drift without retraining. See `docs/threshold_policy.md` §5.

---

### 4.3 Feature Relationship Stationarity

**Assumption:** The relationship between the model's input features and the probability of default is stable over time. Specifically, the strong predictive relationship between `PAY_0` (most recent payment status) and default (`corr ≈ +0.33` in training data) is assumed to persist.

**Risk if violated:** Concept drift — if regulations, payment processing changes, or product changes alter the *meaning* of payment status codes, `PAY_0` will retain its distribution while losing its predictive power. This is the hardest failure mode to detect (see `docs/drift_concepts.md` §3).

**Monitoring signal:** Rolling `corr(PAY_0, default)` on labelled outcomes. Correlation falling below +0.20 triggers concept drift investigation.

---

### 4.4 Binary Default Outcome

**Assumption:** Default is modelled as a binary outcome (0 = no default within the observation window, 1 = default). The model does not distinguish between early and late defaults, partial defaults, or recoveries.

**Consequence:** The model cannot be used for loss-given-default (LGD) estimation or expected credit loss (ECL) calculations without additional severity modelling.

---

### 4.5 Cost Model Stability

**Assumption:** The business cost of a false negative (missed default) is approximately **$5,000** and the cost of a false positive (unnecessary decline) is approximately **$500**, yielding a 10:1 FN:FP cost ratio. This ratio justifies the aggressive t=0.30 threshold.

**Risk if violated:** If the cost ratio changes materially — for example, because portfolio size, interest rates, or recovery rates change — the optimal threshold will shift. The cost model is documented in `docs/threshold_policy.md` and must be reviewed annually or following material portfolio changes.

---

### 4.6 Independent and Identically Distributed Production Data

**Assumption:** Production applicants are drawn independently from the same population as the training data. Batch scoring within a single time window does not introduce temporal autocorrelation (e.g., a single employer's mass layoff appearing as many related defaults).

**Consequence:** The model's calibration metrics assume IID data. Correlated defaults (systemic events, geographic clusters) will cause the model to underestimate portfolio-level risk even if individual predictions appear well-calibrated.

---

## 5. Model Limitations

The following limitations are documented honestly in accordance with SR 11-7's expectation that model developers identify where models are likely to perform poorly.

### 5.1 Temporal Staleness

The training data is from **2005** — approximately two decades old. Credit product structures, consumer behaviour, regulatory requirements, and economic baselines have changed significantly since then. The model must be treated as a proof-of-concept framework, not a deployable production model, until it is retrained on recent, relevant data.

**Severity:** High. This is the single most significant limitation.

---

### 5.2 Geographic Specificity

The model was trained exclusively on Taiwanese credit card customers. Credit culture, payment norms, regulatory context, and macroeconomic conditions in Taiwan in 2005 may differ materially from any target deployment market. Cross-border or cross-market application requires full revalidation.

**Severity:** High for any non-Taiwan deployment.

---

### 5.3 No Macroeconomic Features

The model has no awareness of unemployment rates, interest rate cycles, housing market conditions, or other macroeconomic indicators that are known to be strong predictors of systemic default risk. During a recession, the model will underestimate risk for all applicants because it cannot detect the economic regime change.

**Severity:** Medium — manageable through threshold adjustment and target drift monitoring, but not eliminable without model redesign.

---

### 5.4 Static Threshold

The classification threshold (t=0.30) is fixed in the model metadata. It was optimised for a 22.1% base default rate and a specific FN:FP cost ratio. If either input changes, the threshold is no longer optimal.

**Mitigation:** The threshold is externalised in `models/champion_model_metadata.json` and can be updated without redeploying code. See `docs/threshold_policy.md`.

---

### 5.5 No Credit Limit or Product Context

The model does not know the credit limit being *offered* — only the applicant's existing `LIMIT_BAL`. It cannot distinguish between a high-limit offer to a low-utilisation applicant and the same offer to a high-utilisation applicant, even if the credit risk differs materially.

---

### 5.6 Short Lookback Window

The model uses a 6-month payment history window (`PAY_0` through `PAY_6`, `BILL_AMT1–6`, `PAY_AMT1–6`). Applicants with thin files (recently onboarded, first-time borrowers) may receive less accurate scores because the model has less history to work with.

---

### 5.7 No API Authentication or Rate Limiting

The current implementation has no API key enforcement or rate limiting on `POST /predict`. In a production deployment, this exposes the model to adversarial probing (an attacker could submit thousands of applicants to reverse-engineer the model's decision boundary). An API gateway with authentication is required before public exposure.

**Mitigation documented in:** `docs/release_notes_v1.md` — Known Issues.

---

## 6. Challenger Model Comparison

SR 11-7 requires that model developers demonstrate they evaluated alternative methodologies and justified the selection of the final model. Three candidate models were trained and evaluated on an identical held-out test set (6,000 rows, stratified 80/20 split, seed=42).

### 6.1 Performance Comparison

All models were evaluated at the same classification threshold (t=0.30) on the same 6,000-row held-out test set.

| Model | AUC-ROC | Gini | Recall at t=0.30 | Interpretability | Training time |
|-------|---------|------|-----------------|-----------------|--------------|
| Logistic Regression | 0.729 | 0.458 | ~68% | High (linear coefficients) | < 1 s |
| Random Forest | 0.757 | 0.514 | ~74% | Medium (feature importances) | ~15 s |
| **XGBoost (Champion)** | **0.7725** | **0.5450** | **80.6%** | **High (SHAP TreeExplainer)** | **~30 s** |

Note: Recall is the priority metric. At a cost ratio of 10:1 (FN:FP), missing a default is ten times more costly than a false alarm. XGBoost delivers meaningfully higher recall at the same threshold — capturing approximately 6–12 additional defaults per 1,000 applicants compared to the challengers.

### 6.2 Champion Selection Rationale

**XGBoost was selected over Logistic Regression** because:

1. **AUC advantage (+4.3 pp):** A 4.3-percentage-point AUC improvement is material in credit scoring, where portfolio sizes are large and even small improvements in rank-ordering translate to significant business value.
2. **Recall advantage:** XGBoost captures approximately 80.6% of defaults at t=0.30 vs. ~68% for Logistic Regression. On a 6,000-row test set with a 22.1% base rate (~1,326 actual defaults), XGBoost correctly identifies roughly 1,069 vs. ~901 for Logistic Regression — 168 additional defaults caught.
3. **Non-linear interactions:** Payment delinquency patterns involve complex interactions (e.g., being 2+ months late *and* at high utilisation is disproportionately risky). XGBoost captures these; Logistic Regression cannot without manual feature engineering.
4. **Interpretability is preserved:** SHAP TreeExplainer provides local and global explanations identical in form to what a logistic regression coefficient analysis provides, eliminating the traditional interpretability trade-off argument.

**XGBoost was selected over Random Forest** because:

1. **AUC advantage (+1.5 pp):** XGBoost is consistently more accurate on structured tabular data due to gradient boosting's sequential error correction.
2. **Probability calibration:** XGBoost produces well-calibrated probability scores out of the box. Random Forests tend to produce compressed probability estimates (scores cluster around 0.2–0.8 rather than spanning the full range), which degrades threshold-based decision-making.
3. **Class imbalance handling:** `scale_pos_weight=3.52` directly encodes the FN:FP cost ratio into XGBoost's objective function. Random Forest has no native equivalent and requires post-hoc cost-sensitive resampling.

**The case for Logistic Regression is not dismissed:** In a conservative regulatory environment or where examiner explainability is paramount, Logistic Regression's lower AUC may be acceptable in exchange for simpler governance. The Logistic Regression model is retained in the codebase for ongoing regulatory comparison.

### 6.3 Hyperparameter Selection

The champion XGBoost model uses:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 300 | Tuned via early stopping; additional trees yielded diminishing returns |
| `max_depth` | 6 | Controls overfitting; depth > 6 showed validation loss increase |
| `learning_rate` | 0.05 | Conservative — pairs with 300 trees for stability |
| `scale_pos_weight` | 3.52 | Ratio of negatives to positives (22,141 / 6,290) in training set |
| `subsample` | 0.8 | Row-level subsampling for variance reduction |
| `colsample_bytree` | 0.8 | Feature-level subsampling; reduces feature dominance |

---

## 7. Implementation Controls

The following controls are embedded in the system to reduce the risk of model misuse, data quality errors, and serving/training divergence.

### 7.1 Input Validation — Pydantic v2 Schema Enforcement

**File:** `api/main.py`

Every `POST /predict` request is validated against a strict Pydantic v2 schema before the model is invoked. The schema enforces:

- `LIMIT_BAL > 0` (non-positive credit limits are rejected)
- `AGE ≥ 18` (minors cannot be scored)
- `PAY_0` – `PAY_6` ∈ [-2, 8] (valid payment status codes only)
- `SEX` ∈ {1, 2}
- `EDUCATION` ∈ {0, 1, 2, 3, 4, 5, 6}
- `MARRIAGE` ∈ {0, 1, 2, 3}
- All bill and payment amounts are non-negative

Invalid requests receive an HTTP 422 response with field-level error detail. The model is never invoked on invalid input.

### 7.2 Training/Serving Feature Parity

**File:** `src/features/engineer.py`

The function `engineer_features()` is the single source of truth for all feature transformations. It is called in **exactly two places**:

1. `src/train.py` — at training time to produce the 51-column feature matrix
2. `api/main.py` (via `_run_model()`) — at inference time before calling `predict_proba`

There is no separate "API preprocessing" step that could silently diverge from the training transformation. Any change to `engineer_features()` automatically applies to both sides.

The `feature_names` list in `models/champion_model_metadata.json` provides an additional guard: the API selects columns in training-time order before passing them to the model, ensuring column alignment even if `engineer_features()` output order changes in future.

### 7.3 Category Binning Parity — `model_validator`

**File:** `api/main.py`

The training data cleaning step bins undocumented `EDUCATION` codes {0, 5, 6} and `MARRIAGE` code {0} into an "Other" category. This binning is reproduced at inference time using a Pydantic v2 `@model_validator`, which runs before `engineer_features()` is called. Without this control, a request with `EDUCATION=5` would reach the model with a code that never appeared in training — a silent data leakage risk.

### 7.4 Audit-Trail Logging

**File:** `api/main.py`

Every `POST /predict` call produces a structured log entry containing:

| Field | Purpose |
|-------|---------|
| `request_id` | Unique UUID per request — enables tracing |
| `limit_bal`, `age`, `pay_0` | Key input features for audit reconstruction |
| `risk_score` | Raw model probability output |
| `risk_band` | Low / Medium / High |
| `action` | Approve / Review / Decline |
| `threshold_used` | The t value applied (read from metadata at startup) |
| `model_version` | The version string from `champion_model_metadata.json` |

In a regulated environment, every credit decision must be traceable (SR 11-7, ECOA / Regulation B). The log structure is designed for direct ingestion into a SIEM or decision database.

### 7.5 Adverse Action Reason Codes

**File:** `docs/model_explainability_summary.md`; `notebooks/07_shap_explainability.ipynb`

For any applicant receiving a "Decline" action, SHAP values are used to identify the top 3 features that most increased the predicted default probability. These constitute the **adverse action reason codes** required under Regulation B. The reason codes are derived from the same model that produced the score — there is no separate explanation model that could diverge.

Example: An applicant declined with `PAY_0=2`, `max_delinquency=3`, and high `util_ratio_avg` would receive:

> "Primary reasons for this decision: (1) Recent payment delay of 2 months, (2) Prior severe delinquency on record, (3) High credit utilisation."

### 7.6 Model Version Provenance

**File:** `models/champion_model_metadata.json`; `GET /model-info`

The deployed model carries full provenance metadata: the git commit hash at the time of training, the training date, all evaluation metrics, and the complete `feature_names` list. Any downstream consumer (auditor, examiner, downstream system) can call `GET /model-info` to verify exactly which model version is live and reproduce its training conditions from the commit hash.

### 7.7 CI/CD Quality Gate

**File:** `.github/workflows/ci.yml`

Every push to the `main` branch must pass a three-step quality gate before merging:

1. **`flake8` linting** — catches runtime-breaking errors (undefined names, syntax errors) in `src/` and `api/`
2. **Full training pipeline** (`python main.py --download`) — proves end-to-end reproducibility from raw data to trained artefact in a clean CI environment
3. **27-test pytest suite** — covers health, model-info, valid predictions, and 12 validation rejection cases

No code that fails this gate can reach the production model artefact.

---

## 8. Model Risk Assessment Summary

| Risk dimension | Current status | Residual risk | Recommended action |
|----------------|---------------|--------------|-------------------|
| Data vintage (2005) | High | High | Retrain on recent, target-market data before any regulated use |
| Geographic specificity (Taiwan) | High | High | Validate on target-market population |
| Macroeconomic blindness | High | Medium | Supplement with macro features or threshold adjustment policy |
| Interpretability | Low (SHAP available) | Low | — |
| Training/serving skew | Low (shared `engineer_features()`) | Low | — |
| Audit trail completeness | Low (structured logging) | Low | Persist logs to database for regulated use |
| API security | High (no authentication) | High | Deploy behind authenticated API gateway |
| Threshold optimality | Medium (cost model assumptions) | Medium | Annual cost model review; adaptive threshold on rate change |

**Overall model risk:** **Medium-High** — the platform demonstrates sound methodology and strong implementation controls, but the training data limitations (vintage and geography) mean it is not suitable for regulated credit decisions without retraining and independent validation.

---

## 9. Revision History

| Date | Version | Author | Change |
|------|---------|--------|--------|
| 2026-06-30 | 1.0 | Credit Risk Analytics | Initial draft |

---

*This document is a draft governance artefact and does not represent an approved model validation or credit policy. Independent model validation, legal review, and Credit Risk Committee approval are required before any production use in a regulated lending context.*
