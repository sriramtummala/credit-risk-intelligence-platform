import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Column groups — defined once, used across all feature functions
BILL_COLS       = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
PAY_AMT_COLS    = ['PAY_AMT1',  'PAY_AMT2',  'PAY_AMT3',  'PAY_AMT4',  'PAY_AMT5',  'PAY_AMT6']
PAY_STATUS_COLS = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']


# ---------------------------------------------------------------------------
# Task 9.1 — Utilization Ratios
# ---------------------------------------------------------------------------

def add_utilization_features(df: pd.DataFrame) -> pd.DataFrame:
    """How much of the credit limit is being used each month."""
    df = df.copy()
    for i, col in enumerate(BILL_COLS, start=1):
        df[f'util_ratio_m{i}'] = df[col] / df['LIMIT_BAL'].replace(0, np.nan)

    df['util_ratio_avg'] = df[[f'util_ratio_m{i}' for i in range(1, 7)]].mean(axis=1)
    df['util_ratio_max'] = df[[f'util_ratio_m{i}' for i in range(1, 7)]].max(axis=1)
    return df


# ---------------------------------------------------------------------------
# Task 9.2 — Payment-to-Bill Ratios
# ---------------------------------------------------------------------------

def add_payment_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """What fraction of each month's bill was actually paid."""
    df = df.copy()
    for i, (pay_col, bill_col) in enumerate(zip(PAY_AMT_COLS, BILL_COLS), start=1):
        # Clamp to [0, 1]: paying more than the bill (overpayment) → 1.0
        ratio = df[pay_col] / df[bill_col].replace(0, np.nan)
        df[f'pay_ratio_m{i}'] = ratio.clip(upper=1.0).fillna(1.0)

    pay_ratio_cols = [f'pay_ratio_m{i}' for i in range(1, 7)]
    df['pay_ratio_avg'] = df[pay_ratio_cols].mean(axis=1)
    df['pay_ratio_min'] = df[pay_ratio_cols].min(axis=1)

    # Total paid vs total billed across all 6 months
    total_bill = df[BILL_COLS].sum(axis=1).replace(0, np.nan)
    df['avg_payment_ratio'] = (df[PAY_AMT_COLS].sum(axis=1) / total_bill).clip(upper=1.0).fillna(1.0)
    return df


# ---------------------------------------------------------------------------
# Task 9.3 — Delinquency Indicators
# ---------------------------------------------------------------------------

def add_delinquency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Count, severity, and pattern of payment delays over 6 months."""
    df = df.copy()
    pay = df[PAY_STATUS_COLS]

    df['total_delinquencies'] = (pay > 0).sum(axis=1)
    df['max_delinquency']     = pay.max(axis=1)
    df['ever_severely_late']  = (pay >= 2).any(axis=1).astype(int)

    # Streak: how many consecutive recent months with delay (starting from PAY_0)
    def _streak(row):
        streak = 0
        for col in PAY_STATUS_COLS:
            if row[col] > 0:
                streak += 1
            else:
                break
        return streak

    df['delinquency_streak'] = df.apply(_streak, axis=1)
    return df


# ---------------------------------------------------------------------------
# Task 9.4 — Risk Trend Features
# ---------------------------------------------------------------------------

def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Is debt growing? Are payments declining? Directional signals."""
    df = df.copy()

    # Debt trend: positive = debt growing (bad), negative = paying down (good)
    # BILL_AMT1 = most recent, BILL_AMT6 = oldest
    df['bill_trend']      = df['BILL_AMT1'] - df['BILL_AMT6']
    df['bill_trend_pct']  = df['bill_trend'] / df['BILL_AMT6'].replace(0, np.nan)

    # Payment trend: positive = paying more recently (good)
    df['payment_trend']   = df['PAY_AMT1'] - df['PAY_AMT6']

    # Average bill and payment amounts
    df['avg_bill_amt']    = df[BILL_COLS].mean(axis=1)
    df['avg_pay_amt']     = df[PAY_AMT_COLS].mean(axis=1)

    # Payment gap: how much of the average bill goes unpaid
    df['avg_unpaid_amt']  = (df['avg_bill_amt'] - df['avg_pay_amt']).clip(lower=0)
    return df


# ---------------------------------------------------------------------------
# Task 9.5 Risk Segment (from Day 6 policy)
# ---------------------------------------------------------------------------

def add_risk_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Rule-based Low/Medium/High segment for interpretability and benchmarking."""
    df = df.copy()

    def _segment(row):
        if row['PAY_0'] >= 2:
            return 'High'
        elif row['PAY_0'] == 1 or row['LIMIT_BAL'] < 200_000:
            return 'Medium'
        return 'Low'

    df['risk_segment']    = df.apply(_segment, axis=1)
    df['risk_segment_id'] = df['risk_segment'].map({'Low': 0, 'Medium': 1, 'High': 2})
    return df


# ---------------------------------------------------------------------------
# Main pipeline — call this from notebooks and modeling scripts
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline. Applies all transformations in order.
    Returns a new DataFrame — does not mutate the input.
    """
    n_original = df.shape[1]
    df = add_utilization_features(df)
    df = add_payment_ratio_features(df)
    df = add_delinquency_features(df)
    df = add_trend_features(df)
    df = add_risk_segment(df)
    n_new = df.shape[1] - n_original
    logger.info("Feature engineering complete. Added %d features. Final shape: %s", n_new, df.shape)
    return df


def get_feature_names() -> dict[str, str]:
    """Returns {feature_name: description} for all engineered features."""
    return {
        # Utilization
        'util_ratio_m1':       'BILL_AMT1 / LIMIT_BAL — most recent month utilization',
        'util_ratio_m2':       'BILL_AMT2 / LIMIT_BAL',
        'util_ratio_m3':       'BILL_AMT3 / LIMIT_BAL',
        'util_ratio_m4':       'BILL_AMT4 / LIMIT_BAL',
        'util_ratio_m5':       'BILL_AMT5 / LIMIT_BAL',
        'util_ratio_m6':       'BILL_AMT6 / LIMIT_BAL',
        'util_ratio_avg':      'Mean utilization across 6 months',
        'util_ratio_max':      'Peak utilization across 6 months',
        # Payment ratios
        'pay_ratio_m1':        'PAY_AMT1 / BILL_AMT1, clamped to [0, 1]',
        'pay_ratio_m2':        'PAY_AMT2 / BILL_AMT2, clamped to [0, 1]',
        'pay_ratio_m3':        'PAY_AMT3 / BILL_AMT3, clamped to [0, 1]',
        'pay_ratio_m4':        'PAY_AMT4 / BILL_AMT4, clamped to [0, 1]',
        'pay_ratio_m5':        'PAY_AMT5 / BILL_AMT5, clamped to [0, 1]',
        'pay_ratio_m6':        'PAY_AMT6 / BILL_AMT6, clamped to [0, 1]',
        'pay_ratio_avg':       'Mean pay ratio across 6 months',
        'pay_ratio_min':       'Lowest pay ratio across 6 months (worst month)',
        'avg_payment_ratio':   'Total PAY_AMT / Total BILL_AMT across 6 months',
        # Delinquency
        'total_delinquencies': 'Count of months with PAY_STATUS > 0',
        'max_delinquency':     'Worst payment delay in months (max of PAY_0..PAY_6)',
        'ever_severely_late':  'Binary: 1 if any month had PAY_STATUS >= 2',
        'delinquency_streak':  'Consecutive recent months with delay (from PAY_0 backward)',
        # Trends
        'bill_trend':          'BILL_AMT1 - BILL_AMT6: positive = growing debt',
        'bill_trend_pct':      'bill_trend / BILL_AMT6: relative debt growth rate',
        'payment_trend':       'PAY_AMT1 - PAY_AMT6: positive = paying more recently',
        'avg_bill_amt':        'Mean bill amount across 6 months',
        'avg_pay_amt':         'Mean payment amount across 6 months',
        'avg_unpaid_amt':      'avg_bill_amt - avg_pay_amt (floored at 0)',
        # Risk segment
        'risk_segment':        'Rule-based segment: Low / Medium / High (from Day 6 policy)',
        'risk_segment_id':     'Numeric encoding: Low=0, Medium=1, High=2',
    }
