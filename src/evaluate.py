"""Evaluation utilities — metrics, risk bands, and business cost analysis."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test, *, threshold: float = 0.30) -> dict:
    """
    Full evaluation of a trained classifier.

    Returns a flat dict suitable for JSON serialisation and metadata logging.
    Includes threshold-independent (AUC, Gini) and threshold-dependent metrics.
    """
    y_probs = model.predict_proba(X_test)[:, 1]
    y_preds = (y_probs >= threshold).astype(int)

    auc = roc_auc_score(y_test, y_probs)
    tn, fp, fn, tp = confusion_matrix(y_test, y_preds).ravel()
    n = len(y_test)

    return {
        "auc_roc":           round(float(auc), 4),
        "gini":              round(float(2 * auc - 1), 4),
        "threshold":         threshold,
        "precision":         round(float(precision_score(y_test, y_preds, zero_division=0)), 4),
        "recall":            round(float(recall_score(y_test, y_preds, zero_division=0)), 4),
        "f1":                round(float(f1_score(y_test, y_preds, zero_division=0)), 4),
        "accuracy":          round(float(accuracy_score(y_test, y_preds)), 4),
        "tp":                int(tp),
        "tn":                int(tn),
        "fp":                int(fp),
        "fn":                int(fn),
        "approval_rate":     round(float((tn + fn) / n), 4),
        "default_slip_rate": round(float(fn / (tn + fn)) if (tn + fn) > 0 else 0.0, 4),
        "business_cost_usd": int(fn * 5_000 + fp * 500),
        "n_test":            n,
    }


def risk_band_summary(y_probs: np.ndarray, y_true) -> pd.DataFrame:
    """
    Assign Low / Medium / High risk bands and return per-band default rates.

    Bands: Low < 0.20 <= Medium < 0.50 <= High
    """
    bands = pd.cut(
        y_probs,
        bins=[-np.inf, 0.20, 0.50, np.inf],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )
    df = pd.DataFrame({"band": bands, "actual": np.asarray(y_true)})
    return (
        df.groupby("band", observed=True)
        .agg(count=("actual", "count"), default_rate=("actual", "mean"))
        .assign(pct_of_portfolio=lambda d: d["count"] / len(df))
        .round(4)
    )


def print_metrics(metrics: dict) -> None:
    """Print a formatted metrics summary to stdout."""
    width = 44
    print("=" * width)
    print("  MODEL EVALUATION REPORT")
    print("=" * width)
    print(f"  Threshold     : {metrics['threshold']}")
    print(f"  AUC-ROC       : {metrics['auc_roc']}  (Gini: {metrics['gini']})")
    print(f"  Precision     : {metrics['precision']}")
    print(f"  Recall        : {metrics['recall']}")
    print(f"  F1            : {metrics['f1']}")
    print(f"  Accuracy      : {metrics['accuracy']}")
    print("-" * width)
    print(f"  TP / TN / FP / FN : {metrics['tp']} / {metrics['tn']} / {metrics['fp']} / {metrics['fn']}")
    print(f"  Approval rate     : {metrics['approval_rate']:.1%}")
    print(f"  Default slip rate : {metrics['default_slip_rate']:.1%}")
    print(f"  Business cost     : ${metrics['business_cost_usd']:,.0f}")
    print("=" * width)
