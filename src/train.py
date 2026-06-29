"""
Training pipeline for the Credit Risk Intelligence Platform.

Entry points
------------
run_training_pipeline()   Full pipeline: load → engineer → train → evaluate → save.
_load_processed()         Load the best available processed parquet (no re-download).
"""
import json
import logging
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data_ingestion import load_data
from src.evaluate import evaluate_model, print_metrics, risk_band_summary
from src.features.engineer import engineer_features

logger = logging.getLogger(__name__)

# Columns that must not enter the feature matrix
_DROP_FROM_X = {"default", "risk_segment", "edu_label", "mar_label", "age_group"}

# Processed parquet candidates — preferred first
_PROCESSED_PATHS = [
    Path("data/processed/credit_default_eda_ready.parquet"),
    Path("data/processed/credit_default_validated.parquet"),
    Path("data/processed/credit_default_cleaned.parquet"),
]

# Champion model hyperparameters (matched to notebooks 05–07)
_XGB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    eval_metric="logloss",
    n_jobs=-1,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _load_processed() -> "pd.DataFrame":
    import pandas as pd
    for path in _PROCESSED_PATHS:
        if path.exists():
            logger.info("Loading processed data from %s", path)
            return pd.read_parquet(path)
    raise FileNotFoundError(
        "No processed parquet found in data/processed/. "
        "Pass download=True to run_training_pipeline() to fetch from source."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_training_pipeline(
    *,
    download: bool = False,
    model_dir: str | Path = "models",
    test_size: float = 0.2,
    random_state: int = 42,
    threshold: float = 0.30,
) -> dict:
    """
    Run the full training pipeline from data to saved model artifact.

    Parameters
    ----------
    download    : Re-download and re-clean the UCI source file if True.
                  If False (default), use the best available processed parquet.
    model_dir   : Directory where model artifacts are saved.
    test_size   : Fraction of data held out for evaluation.
    random_state: RNG seed — must match notebooks for reproducibility.
    threshold   : Classification threshold used to compute threshold-dependent metrics.

    Returns
    -------
    metadata dict — same content as the saved metadata.json file.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data ────────────────────────────────────────────────────────────
    logger.info("Step 1/5 — Loading data")
    if download:
        df_raw = load_data()
    else:
        df_raw = _load_processed()

    df_raw = df_raw.drop(
        columns=[c for c in ["edu_label", "mar_label", "age_group"] if c in df_raw.columns]
    )

    # 2. Feature engineering ──────────────────────────────────────────────────
    logger.info("Step 2/5 — Engineering features")
    df = engineer_features(df_raw)

    X = df.drop(columns=[c for c in _DROP_FROM_X if c in df.columns])
    y = df["default"]
    logger.info("Feature matrix: %s  |  Class balance: %.1f%% default", X.shape, y.mean() * 100)

    # 3. Train / test split ───────────────────────────────────────────────────
    logger.info("Step 3/5 — Splitting data (test_size=%.0f%%)", test_size * 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = float(neg / pos)

    # 4. Train model ──────────────────────────────────────────────────────────
    logger.info("Step 4/5 — Training XGBoost champion model")
    model = XGBClassifier(
        **_XGB_PARAMS,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    # 5. Evaluate + save ──────────────────────────────────────────────────────
    logger.info("Step 5/5 — Evaluating and saving artifacts")
    metrics = evaluate_model(model, X_test, y_test, threshold=threshold)
    band_df = risk_band_summary(model.predict_proba(X_test)[:, 1], y_test)

    datestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    versioned_path = model_dir / f"champion_xgb_{datestamp}.joblib"
    latest_path    = model_dir / "champion_model.joblib"

    joblib.dump(model, versioned_path)
    shutil.copy2(versioned_path, latest_path)
    logger.info("Model saved: %s (copied to %s)", versioned_path, latest_path)

    metadata = {
        "model_name":        "credit_risk_xgb",
        "version":           "1.0",
        "training_date":     datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "git_commit":        _git_hash(),
        # Full metrics dict merged in — keeps metadata self-contained
        **metrics,
        "n_train":           len(X_train),
        "n_features":        X.shape[1],
        "feature_names":     X.columns.tolist(),
        "hyperparameters": {
            **_XGB_PARAMS,
            "scale_pos_weight": round(scale_pos_weight, 4),
            "random_state":     random_state,
        },
        "risk_bands": band_df[["count", "pct_of_portfolio", "default_rate"]].to_dict(),
        "model_path":        str(latest_path),
        "versioned_path":    str(versioned_path),
    }

    meta_path = model_dir / "champion_model_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Metadata saved: %s", meta_path)

    return metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    result = run_training_pipeline()
    print_metrics({k: result[k] for k in [
        "threshold", "auc_roc", "gini", "precision", "recall", "f1", "accuracy",
        "tp", "tn", "fp", "fn", "approval_rate", "default_slip_rate", "business_cost_usd", "n_test",
    ] if k in result})
