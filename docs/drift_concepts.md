# Data & Model Drift — Concepts and Monitoring Rationale

## Why Monitoring Matters for Model Governance

A credit risk model is not a static artifact. The populations it was trained on, the economic conditions it learned from, and the behaviours it tried to predict all change over time. Deploying a model without monitoring is equivalent to flying a plane without instruments: you may be on course, or you may not — and you won't know until you crash.

For institutions subject to **SR 11-7** (the Fed and OCC's Model Risk Management guidance), ongoing monitoring is not optional:

> *"Outcomes analysis should be conducted regularly to assess model performance against benchmarks... banks should have a process for identifying and responding to model deterioration."*  
> — SR 11-7, Section IV

For a credit scoring model specifically, undetected drift directly translates to financial loss: approving customers who are riskier than the model thinks (false sense of safety), or declining customers who are actually creditworthy (lost revenue).

---

## The Three Types of Drift

### 1. Feature Drift (Data Drift)

**Definition:** The statistical distribution of one or more input features changes between training time and production.

**Credit risk example:** A period of economic expansion causes average credit limits (`LIMIT_BAL`) to increase by 30–50% across the applicant pool. The model was trained on a lower-limit population and may no longer be well-calibrated for high-limit customers.

**Detection metric:** Population Stability Index (PSI)

$$\text{PSI} = \sum_{i=1}^{n} \left( A_i - E_i \right) \ln\left( \frac{A_i}{E_i} \right)$$

where $A_i$ = actual (production) proportion in bin $i$, $E_i$ = expected (reference) proportion.

| PSI | Interpretation | Recommended Action |
|-----|---------------|-------------------|
| < 0.10 | No significant shift | Continue monitoring |
| 0.10 – 0.25 | Moderate shift | Investigate the feature; check upstream data pipelines |
| > 0.25 | Major shift | **Alert model owner; consider retraining** |

**Why PSI > 0.25 in our simulation:** Multiplying `LIMIT_BAL` by 1.5 with Gaussian noise creates a distribution that is far outside the training range, generating PSI ≈ 0.35–0.50.

---

### 2. Target Drift (Label Drift)

**Definition:** The distribution of the outcome variable changes — for credit risk, this means the default rate itself shifts.

**Credit risk example:** A sudden recession (e.g., mass layoffs, interest rate shock) causes the portfolio default rate to spike from 22% to 40%. The model was calibrated to a 22% base rate, so its probability estimates are systematically underestimating risk for the new economic regime.

**Detection metrics:**
- **Kolmogorov-Smirnov (KS) test** on the binary target distributions (p < 0.05 → drift detected)
- **Approval rate monitoring** — if the fraction of applicants being approved drifts by ±10 percentage points from baseline, the score distribution has shifted

**Why this matters:** Target drift is a leading indicator. A model that was performing well will begin to deteriorate *after* the target distribution shifts. Catching target drift early allows the risk team to tighten thresholds before the AUC metric visibly drops.

**What to do when detected:**
1. Immediately notify the Credit Risk Committee
2. Review whether to adjust the classification threshold downward (more conservative)
3. Initiate an expedited retraining with data from the new economic period

---

### 3. Concept Drift

**Definition:** The *relationship* between features and the target changes — the feature values stay the same but they no longer mean the same thing.

**Credit risk example:** A new regulatory grace period means that customers coded as `PAY_0 = 2` (2-month payment delay) are no longer at elevated default risk — the grace period has changed the meaning of the payment status codes. `PAY_0` was the top SHAP driver of our model (mean |SHAP| = 0.389). If it loses predictive power, the model degrades even if its input distribution looks normal.

**Why concept drift is hardest to catch:**
- PSI on `PAY_0` may be low (the feature values themselves haven't shifted)
- There is no distributional alarm — the model simply starts making worse decisions
- The only reliable detection is **outcome monitoring**: track realised default rates for approved vs. declined populations over time

**Detection metrics:**
- **AUC-ROC on a holdout** with recent labels (requires waiting for outcomes)
- **Correlation monitoring**: `corr(PAY_0, default)` should remain stable over rolling windows
- **SHAP stability**: if the rank order of top feature importances changes significantly, concept drift is likely

**In our simulation:** Randomising `PAY_0` for defaulters causes:
- `corr(PAY_0, default)` to collapse from ≈ +0.35 to near zero
- Model AUC to drop below the 0.72 retraining trigger

---

## Monitoring Architecture

```
Production data (daily batch)
        │
        ▼
┌───────────────────────┐     ┌─────────────────────────────┐
│  Feature Distribution │     │  Model Performance           │
│  PSI per feature      │     │  AUC on labelled holdout     │
│  Null rate monitoring │     │  Approval rate vs. baseline  │
│  Range / outlier check│     │  Default rate in approved    │
└───────────┬───────────┘     └─────────────┬───────────────┘
            │                               │
            ▼                               ▼
   PSI > 0.25?                     AUC < 0.72?
   Null% > 5%?                     Approval ± 10pp?
   KS p < 0.05?                    PSI > 0.25?
            │                               │
            └───────────┬───────────────────┘
                        ▼
               Alert → Model Owner → Credit Risk Committee
                        ▼
               Investigate → Retrain → Re-validate → Re-deploy
```

---

## Monitoring Triggers (from `docs/threshold_policy.md`)

| Metric | Trigger Value | Response |
|--------|--------------|---------|
| PSI (any feature) | > 0.25 | Investigate + retrain if persistent |
| AUC-ROC | < 0.72 | Expedited retraining |
| Gini coefficient | < 0.45 | Expedited retraining |
| Approval rate drift | ± 10 percentage points | Threshold review |
| Feature null rate | > 5% | Data pipeline investigation |

---

## Key Considerations

**Why monitoring is non-negotiable:** SR 11-7 requires ongoing performance monitoring for all models in a bank's model inventory. Beyond compliance, a credit scoring model that silently drifts is a financial liability — approving riskier customers than the model thinks, or declining creditworthy ones. Each of the three drift types requires a different detection method and a different response.

**Distinguishing drift types:** Feature drift is the easiest to detect (PSI on input distributions). Target drift is the most actionable — if the default rate is shifting, tighten thresholds immediately. Concept drift is the most dangerous: PSI looks fine, but the model's internal logic has become invalid. The only reliable signal is outcome monitoring with a labelled holdout.

**Simulation approach:** Simulating all three drift types against the test set before building the monitoring system allows PSI thresholds to be calibrated empirically. A 50% credit-limit shift produces PSI ≈ 0.45 (well above the 0.25 alert threshold), and corrupting the `PAY_0` → default relationship — the top SHAP driver — causes AUC to drop below the 0.72 retrain trigger. This confirms the monitoring system will fire when it should.

---

## Generated Artifacts

| File | Description |
|------|-------------|
| `notebooks/08_drift_simulation.ipynb` | Drift simulation notebook (PSI, KS, AUC degradation) |
| `data/simulated_production_data.csv` | Feature-drifted dataset for monitoring validation |
| `reports/figures/25_feature_drift_distributions.png` | LIMIT_BAL and AGE distribution comparison |
| `reports/figures/26_target_drift.png` | Default rate shift (reference vs. recession) |
| `reports/figures/27_concept_drift_auc.png` | AUC degradation under concept drift |
