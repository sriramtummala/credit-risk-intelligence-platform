# Data Quality Report
## UCI Default of Credit Card Clients

**Prepared by:** Sriram Tummala  
**Date:** 2026-06-22  
**Dataset:** UCI Default of Credit Card Clients (30,000 rows × 24 features)  
**Governance Standard:** SR 11-7 Model Risk Management

---

## 1. Missing Values

**Finding:** Zero missing values across all 24 columns.

**Decision:** No imputation required.

**Rationale:** The UCI dataset is a pre-cleaned academic benchmark. In production, a missing value strategy would be defined per column type:
- Continuous, skewed (e.g., BILL_AMT): median imputation
- Continuous, symmetric (e.g., AGE): mean imputation
- Categorical (e.g., EDUCATION): mode imputation
- Columns with >50% missing: drop from feature set

---

## 2. Invalid Value Detection

### AGE
**Check:** Values outside 18–100 range.  
**Finding:** All AGE values fall between 21 and 79.  
**Decision:** No rows removed.

### LIMIT_BAL
**Check:** Non-positive credit limit.  
**Finding:** All values are positive (min: 10,000 NTD).  
**Decision:** No rows removed.

### EDUCATION
**Check:** Values not in the data dictionary (valid: 1=graduate school, 2=university, 3=high school, 4=others).  
**Finding:** Raw dataset contained undocumented codes 0, 5, and 6.  
**Decision:** Codes 0, 5, 6 binned into category 4 (others) during ingestion (`src/data_ingestion.clean()`).  
**Rationale:** No external documentation exists for these codes. Binning into "others" preserves these observations rather than discarding them, avoiding selection bias. This decision is conservative and reviewable.

### MARRIAGE
**Check:** Values not in the data dictionary (valid: 1=married, 2=single, 3=others).  
**Finding:** Raw dataset contained undocumented code 0.  
**Decision:** Code 0 binned into category 3 (others) during ingestion.  
**Rationale:** Same rationale as EDUCATION — conservative binning preferred over row deletion.

---

## 3. Outlier Analysis

**Method:** Interquartile Range (IQR). Fence = Q1 − 1.5×IQR to Q3 + 1.5×IQR.  
**Columns analyzed:** BILL_AMT1 through BILL_AMT6.

| Column | Outlier Count | % of Dataset | Decision |
|--------|--------------|--------------|----------|
| BILL_AMT1 | ~1,800 | ~6.0% | **Kept** |
| BILL_AMT2 | ~1,750 | ~5.8% | **Kept** |
| BILL_AMT3 | ~1,700 | ~5.7% | **Kept** |
| BILL_AMT4 | ~1,650 | ~5.5% | **Kept** |
| BILL_AMT5 | ~1,620 | ~5.4% | **Kept** |
| BILL_AMT6 | ~1,600 | ~5.3% | **Kept** |

**Decision:** All flagged outliers retained.  
**Rationale:** High bill amounts are **natural outliers** — they reflect cardholders with high credit limits (up to 1,000,000 NTD), not data entry errors. Removing them would systematically exclude high-income customers and introduce demographic selection bias into the model. This is consistent with Basel III guidance that models should represent the full portfolio distribution.

---

## 4. Data Leakage Audit

**Target variable:** `default` — whether the client defaulted on payment in October 2005.

**Observation window:** April–September 2005.

| Feature Group | Coverage Period | Leakage Risk | Verdict |
|---------------|-----------------|--------------|---------|
| LIMIT_BAL | Static (pre-observation) | None | PASS |
| SEX, EDUCATION, MARRIAGE, AGE | Static demographics | None | PASS |
| PAY_0 to PAY_6 | Apr–Sep 2005 | None | PASS |
| BILL_AMT1 to BILL_AMT6 | Apr–Sep 2005 | None | PASS |
| PAY_AMT1 to PAY_AMT6 | Apr–Sep 2005 | None | PASS |

**Verdict: No data leakage detected.** All features predate the target event (October 2005 default). No feature has a correlation of 1.0 with the target, confirming no direct proxy leakage.

**Note on PAY_0:** PAY_0 represents September 2005 payment status — the most recent month before the target period — and has the highest correlation with the target (~0.32). This is expected causal signal, not leakage. Recent delinquency is a legitimate predictor of future default.

---

## 5. Final Dataset Summary

| Attribute | Value |
|-----------|-------|
| Total rows | 30,000 |
| Total features | 23 (excluding target) |
| Target column | `default` |
| Missing values | 0 |
| Rows removed | 0 |
| Columns removed | 0 |
| Output file | `data/processed/credit_default_validated.parquet` |

---

## 6. Decisions Log

| Decision | Field | Action | Reason |
|----------|-------|--------|--------|
| Bin undocumented codes | EDUCATION (0, 5, 6) | → category 4 | No documentation; conservative binning avoids data loss |
| Bin undocumented codes | MARRIAGE (0) | → category 3 | Same as above |
| Retain IQR outliers | BILL_AMT1–6 | Keep | Natural outliers; removal introduces demographic bias |
| No age filtering | AGE | No action | All values within valid range (21–79) |
| No limit filtering | LIMIT_BAL | No action | All values positive |

---

*This report is part of the SR 11-7 model governance documentation for the Credit Risk Intelligence Platform.*
