"""
Credit Risk Intelligence Platform — Pipeline Entry Point

Usage
-----
    python main.py                   # Use cached processed data (fast)
    python main.py --download        # Re-download from UCI source (slow)
    python main.py --threshold 0.35  # Override classification threshold

Outputs
-------
    models/champion_model.joblib           Latest model artifact
    models/champion_xgb_YYYYMMDD.joblib   Versioned copy
    models/champion_model_metadata.json   Training provenance + metrics
"""
import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path when run from any working directory
sys.path.insert(0, str(Path(__file__).parent))

from src.evaluate import print_metrics, risk_band_summary
from src.train import run_training_pipeline


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and save the credit-risk XGBoost champion model.")
    p.add_argument("--download",  action="store_true", help="Re-download raw UCI data before training.")
    p.add_argument("--threshold", type=float, default=0.30, help="Classification threshold (default: 0.30).")
    p.add_argument("--test-size", type=float, default=0.20, help="Held-out fraction (default: 0.20).")
    p.add_argument("--model-dir", type=str,   default="models", help="Directory to save model artifacts.")
    p.add_argument("--verbose",   action="store_true", help="Set log level to DEBUG.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    print()
    print("=" * 52)
    print("  CREDIT RISK INTELLIGENCE PLATFORM v1.0")
    print("  Training Pipeline")
    print("=" * 52)
    print(f"  Threshold  : {args.threshold}")
    print(f"  Test size  : {args.test_size:.0%}")
    print(f"  Model dir  : {args.model_dir}")
    print(f"  Download   : {args.download}")
    print("=" * 52)
    print()

    metadata = run_training_pipeline(
        download=args.download,
        model_dir=args.model_dir,
        test_size=args.test_size,
        threshold=args.threshold,
    )

    # Build a metrics dict that print_metrics understands
    metrics_for_display = {
        "threshold":         metadata["threshold"],
        "auc_roc":           metadata["auc_roc"],
        "gini":              metadata["gini"],
        "precision":         metadata["precision"],
        "recall":            metadata["recall"],
        "f1":                metadata["f1"],
        "accuracy":          metadata.get("accuracy", "—"),
        "tp":                metadata.get("tp", 0),
        "tn":                metadata.get("tn", 0),
        "fp":                metadata.get("fp", 0),
        "fn":                metadata.get("fn", 0),
        "approval_rate":     metadata["approval_rate"],
        "default_slip_rate": metadata.get("default_slip_rate", 0),
        "business_cost_usd": metadata["business_cost_usd"],
        "n_test":            metadata["n_test"],
    }
    print()
    print_metrics(metrics_for_display)

    print()
    print("  Risk Band Breakdown")
    print("  -------------------")
    bands = metadata.get("risk_bands", {})
    for band in ["Low Risk", "Medium Risk", "High Risk"]:
        cnt  = int(bands.get("count",            {}).get(band, 0))
        pct  = float(bands.get("pct_of_portfolio",{}).get(band, 0))
        rate = float(bands.get("default_rate",   {}).get(band, 0))
        print(f"  {band:<13} {cnt:>5,} customers  ({pct:.1%})  default rate: {rate:.1%}")

    print()
    print(f"  Git commit  : {metadata['git_commit']}")
    print(f"  Model saved : {metadata['model_path']}")
    print(f"  Metadata    : models/champion_model_metadata.json")
    print()
    print("  Pipeline complete.")
    print()


if __name__ == "__main__":
    main()
