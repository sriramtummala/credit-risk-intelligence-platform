# Model Card — Credit Risk Intelligence Platform v1.0

> **SR 11-7 Alignment:** This document is the model inventory record required under the Federal Reserve's SR 11-7 guidance on model risk management. It must be reviewed and approved by an independent Model Validation team before production deployment.

---

## Section 1 — Model Details

| Field | Value |
|-------|-------|
| **Model Name** | Credit Risk Intelligence Platform v1.0 |
| **Short Name** | CRIP-v1 |
| **Model Type** | Binary Classification — XGBoost Gradient Boosted Trees |
| **Version** | 1.0.0 |
| **Developer** | Sriram Tummala |
| **Development Date** | June 2026 |
| **Last Updated** | 2026-06-29 |
| **Status** | Development — pending independent validation |
| **Primary Language** | Python 3.11 |
| **Libraries** | XGBoost 2.x, scikit-learn, SHAP 0.52 |
| **Repository** | credit-risk-intelligence-platform |

### Hyperparameters (Champion Model)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 300 | Sufficient trees to stabilise loss without overfitting |
| `learning_rate` | 0.05 | Conservative step size; paired with 300 trees |
| `max_depth` | 6 | Controls complexity; standard for tabular credit data |
| `scale_pos_weight` | 3.52 | Corrects for 22% minority-class (default) imbalance |
| `eval_metric` | logloss | Calibration-aware; suitable for probability output |
| `random_state` | 42 | Reproducibility |

---

## Section 2 — Intended Use

### Primary Use
Predict the probability that a credit card holder will default on their payment in the next billing cycle. The model outputs a score between 0 and 1; scores are converted to credit decisions using the risk-banding policy in `docs/threshold_policy.md`.

### Intended Users
- Credit risk analysts reviewing borderline applications (Medium Risk band)
- Automated underwriting systems making instant pass/decline decisions (Low and High Risk bands)
- Model Risk Management teams conducting ongoing validation

### Out-of-Scope Uses
The following uses are **not supported** by this model and require separate development and validation:

- Commercial lending or business credit risk
- Mortgage or auto loan default prediction
- Real-time transaction fraud detection
- Macroeconomic or portfolio-level stress testing
- Any decision requiring fair-lending certification without completing the disparate impact analysis described in Section 7

---

## Section 3 — Training Data

### Source
**UCI Machine Learning Repository — Default of Credit Card Clients Dataset**

| Attribute | Detail |
|-----------|--------|
| Records | 30,000 credit card holders |
| Time period | April – September 2005 |
| Geography | Taiwan |
| Target variable | `default.payment.next.month` (1 = default, 0 = no default) |
| Class balance | 77.9% no default / **22.1% default** |
| Features (raw) | 23 (demographics + 6-month payment/bill/payment-amount history) |
| Features (after engineering) | 51 (input to model) |

### Train / Test Split

| Split | Size | Default Rate |
|-------|------|-------------|
| Training | 24,000 (80%) | 22.1% |
| Test (held out) | 6,000 (20%) | 22.1% (stratified) |

### Data Cleaning Steps (Day 4)

1. **Undocumented category binning:** `EDUCATION` categories {0, 5, 6} and `MARRIAGE` category {0} are not described in the dataset codebook. They were binned into an "Other" category rather than dropped, preserving all 30,000 records.
2. **Age range filter:** Retained ages 21–79; extreme outliers investigated and confirmed within plausible cardholder range.
3. **No rows dropped for missing values:** The dataset contains no null values after source loading.
4. **String label columns removed:** Derived label columns (`edu_label`, `mar_label`, `age_group`) were dropped before model training to prevent data leakage through categorical-to-numeric encoding.

### Engineered Features (Day 9 — Selected Highlights)

| Feature | Construction | Business Rationale |
|---------|-------------|-------------------|
| `max_delinquency` | `max(PAY_0 … PAY_6)` | Single worst payment ever — top global SHAP driver (0.389) |
| `total_delinquencies` | `sum(PAY_i > 0)` | Count of delinquent months — third global SHAP driver (0.206) |
| `util_ratio_m1` … `util_ratio_m6` | `BILL_AMT_i / LIMIT_BAL` | Monthly credit utilisation — standard risk factor |
| `bill_trend` | Slope of 6-month bill amounts | Captures whether balances are growing or shrinking |
| `avg_pay_amt` | Mean of `PAY_AMT_1` … `PAY_AMT_6` | Average payment effort over 6 months |
| `pay_to_bill_ratio_m1` | `PAY_AMT_1 / (BILL_AMT_1 + 1)` | Payment coverage ratio |

---

## Section 4 — Performance Metrics

All metrics are evaluated on the **held-out test set (6,000 customers)**. No test-set data was used during training or hyperparameter selection.

### Threshold-Independent Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **AUC-ROC** | **0.7725** | Model ranks a random defaulter above a random non-defaulter 77.3% of the time |
| **Gini Coefficient** | **0.5450** | Standard credit-risk rank-ordering measure (0=random, 1=perfect) |
| **Average Precision (AP)** | **0.5483** | Area under the Precision-Recall curve |

### Threshold-Dependent Metrics

| Threshold | Precision | Recall | F1 | Approval Rate | Business Cost |
|-----------|-----------|--------|----|---------------|---------------|
| **0.30 (recommended)** | **0.3333** | **0.8056** | **0.4715** | **46.6%** | **$2,359,000** |
| 0.50 (naive default) | 0.4742 | 0.6036 | 0.5312 | 71.9% | $3,074,000 |
| 0.20 (cost minimum) | 0.2803 | 0.8990 | 0.4274 | 29.1% | $2,201,500 |

> Business cost model: FN = $5,000 (loan loss), FP = $500 (lost interest revenue) per customer. Full cost analysis in `docs/threshold_policy.md`.

### Performance by Risk Band (at t = 0.30 operational threshold)

| Risk Band | Score Range | Portfolio % | Observed Default Rate | Action |
|-----------|-------------|-------------|----------------------|--------|
| Low Risk | 0.00 – 0.19 | 29.1% | **7.7%** | Automatic Approval |
| Medium Risk | 0.20 – 0.49 | 42.8% | **15.3%** | Manual Review |
| High Risk | 0.50 – 1.00 | 28.1% | **47.4%** | Automatic Decline |

### Model Comparison (Why XGBoost Won)

| Model | AUC-ROC | Recall (t=0.5) | F1 | Train Time |
|-------|---------|----------------|----|------------|
| Logistic Regression (baseline) | ~0.72 | ~0.64 | ~0.48 | <1s |
| Random Forest | ~0.76 | ~0.63 | ~0.50 | ~15s |
| **XGBoost (champion)** | **0.7725** | **0.6036** | **0.5312** | ~8s |

XGBoost was selected as the primary model for its best AUC-ROC and Recall. Logistic Regression is retained as the **challenger model** for regulatory baseline comparison and governance transparency.

---

## Section 5 — Explainability

### Method
SHAP (SHapley Additive exPlanations) with `TreeExplainer` — provides **mathematically exact** feature attributions for tree-based models with no approximation error.

### Top 10 Global Feature Drivers (Mean |SHAP Value|)

| Rank | Feature | Mean |SHAP| | Direction |
|------|---------|------------|-----------|
| 1 | max_delinquency *(engineered)* | 0.3891 | Higher increases default risk |
| 2 | PAY_0 | 0.2724 | Higher increases default risk |
| 3 | total_delinquencies *(engineered)* | 0.2063 | Higher increases default risk |
| 4 | PAY_AMT1 | 0.1162 | Higher **decreases** default risk |
| 5 | util_ratio_m2 *(engineered)* | 0.1019 | Higher increases default risk |
| 6 | LIMIT_BAL | 0.0954 | Higher decreases default risk |
| 7 | bill_trend *(engineered)* | 0.0801 | Increasing bills increases risk |
| 8 | avg_pay_amt *(engineered)* | 0.0711 | Higher decreases default risk |
| 9 | PAY_AMT2 | 0.0957 | Higher decreases default risk |
| 10 | PAY_AMT3 | 0.0673 | Higher decreases default risk |

**Notable:** 3 of the top 5 features are engineered — confirming that feature construction (Day 9) added material predictive signal beyond raw variables.

### Individual Explanations
SHAP waterfall plots are generated per-applicant on demand. For any declined applicant, the top 3 positive SHAP values constitute the **Adverse Action reason codes** required under Regulation B. See `docs/model_explainability_summary.md` for a worked example.

### Regulatory Explainability Deliverables

| Artefact | Location | Purpose |
|----------|----------|---------|
| Global summary plot | `reports/figures/22_shap_summary_beeswarm.png` | SR 11-7 model transparency |
| Global bar chart | `reports/figures/23_shap_bar_importance.png` | Executive / committee presentation |
| Individual waterfall | `reports/figures/24_shap_waterfall_high_risk.png` | Adverse action illustration |
| Explainability narrative | `docs/model_explainability_summary.md` | Non-technical risk manager brief |

---

## Section 6 — Caveats & Known Limitations

| Limitation | Severity | Mitigation |
|------------|----------|------------|
| **Geographic mismatch:** Training data is from Taiwan (2005); default behaviour and regulatory environment differ from the US market | High | Recalibrate coefficients on US portfolio data before deployment |
| **Temporal gap:** Data is 20 years old; consumer credit behaviour, macroeconomic conditions, and payment methods have changed significantly | High | Treat as a proof-of-concept; retrain on current data |
| **Economic shock blindness:** Model was trained in a stable economic period; it will degrade under conditions not seen in training (e.g., pandemic-level unemployment spikes) | Medium | Implement stress-testing triggers; monitor approval-rate drift |
| **Class imbalance:** 22% default rate required `scale_pos_weight` correction; model is sensitive to threshold choice | Medium | Documented in `docs/threshold_policy.md`; threshold reviewed quarterly |
| **No income or employment data:** The dataset contains no income, employment status, or debt-to-income ratio | Medium | These are standard underwriting inputs; absence limits real-world precision |
| **Static snapshot:** Each record is one point in time; model cannot detect customers whose financial situation is improving | Low | Incorporate payment trend engineered features (partially mitigated) |

---

## Section 7 — Ethical Considerations & Fair Lending

### Protected Characteristics Excluded
The following ECOA-protected characteristics are **not used** as model inputs:

- Age (excluded from final feature set)
- Sex / Gender (not in source dataset)
- Marital status (excluded from final feature set)
- National origin (not in source dataset)
- Race / Ethnicity (not in source dataset)

### Required Pre-Deployment Checks

| Check | Status | Description |
|-------|--------|-------------|
| Disparate Impact Analysis (DIA) | **REQUIRED — not completed** | Test approval rates across protected classes; must pass 80% rule |
| Adverse Impact Ratio | **REQUIRED — not completed** | Calculate for each demographic segment |
| Fair lending model testing | **REQUIRED — not completed** | Validate that protected attributes cannot be proxied by included features (e.g., ZIP code as proxy for race) |
| ECOA / Regulation B compliance review | **REQUIRED — not completed** | Legal sign-off on adverse action reason code language |

> **Deployment gate:** This model **must not** be used in production decisioning until all four checks above are completed and documented.

---

## Section 8 — Monitoring & Retraining Triggers

### Scheduled Monitoring

| Activity | Frequency | Owner |
|----------|-----------|-------|
| Model performance review (AUC, KS, Gini) | Monthly | Model Risk Management |
| Population Stability Index (PSI) on input features | Monthly | Credit Analytics |
| SHAP driver rank-order stability check | Quarterly | Credit Analytics |
| Full model validation review | Annual | Independent Model Validation |
| Threshold and risk-band boundary review | Quarterly | Credit Policy Committee |

### Automatic Retraining Triggers

Initiate a retraining and re-validation cycle if **any** of the following are observed:

| Trigger | Threshold | Rationale |
|---------|-----------|-----------|
| AUC-ROC degradation | Falls below **0.72** (from 0.7725 baseline) | 6.5% relative drop indicates meaningful performance loss |
| Gini coefficient drop | Falls below **0.45** (from 0.545 baseline) | Industry minimum for prime credit models |
| Approval rate drift | Shifts by **±10 percentage points** from baseline (46.6%) | Indicates population shift or economic change |
| Default rate shift | Increases by **more than 2 percentage points** in approved book | Portfolio deterioration exceeds model's risk calibration |
| PSI > 0.25 on any top-5 SHAP feature | Single-feature PSI breach | Input distribution has shifted materially |
| External event | Regulatory change, economic shock, product redesign | Qualitative trigger requiring Credit Policy assessment |

---

## Section 9 — Model Governance Log

| Date | Version | Event | Author |
|------|---------|-------|--------|
| 2026-06-17 | 0.1 | Initial model card stub created | S. Tummala |
| 2026-06-22 | 0.2 | Data quality and EDA completed | S. Tummala |
| 2026-06-24 | 0.3 | Baseline Logistic Regression trained | S. Tummala |
| 2026-06-26 | 0.4 | Feature engineering (51 features) completed | S. Tummala |
| 2026-06-27 | 0.5 | XGBoost champion selected; AUC=0.7725 | S. Tummala |
| 2026-06-29 | 0.6 | Threshold policy (t=0.30) documented | S. Tummala |
| 2026-06-29 | 0.7 | SHAP explainability completed; reason codes defined | S. Tummala |
| 2026-06-29 | 1.0 | Full model card — ready for independent validation review | S. Tummala |

---

## Section 10 — SR 11-7 Compliance Checklist

| SR 11-7 Requirement | Status | Evidence |
|--------------------|--------|----------|
| Model purpose documented | COMPLETE | Section 2 |
| Training data documented | COMPLETE | Section 3 |
| Performance metrics documented | COMPLETE | Section 4 |
| Explainability / transparency | COMPLETE | Section 5 + `docs/model_explainability_summary.md` |
| Limitations documented | COMPLETE | Section 6 |
| Fair lending / bias checks | PENDING | Section 7 — DIA required |
| Monitoring plan | COMPLETE | Section 8 |
| Independent validation | PENDING | Schedule with Model Risk Management |
| Model inventory registration | COMPLETE | This document |
| Ongoing challenger model | COMPLETE | Logistic Regression retained as challenger |
| Adverse action reason codes | COMPLETE | SHAP top-3 positive values per declined applicant |

---

*This model card is a living document. It must be updated whenever the model is retrained, the threshold is changed, or a material limitation is identified. Approval of this card by Model Risk Management constitutes formal entry into the model inventory.*
