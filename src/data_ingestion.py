import logging
import requests
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/00350/default%20of%20credit%20card%20clients.xls"
)

RAW_PATH = Path("data/raw/UCI_Credit_Card.xls")
PROCESSED_PATH = Path("data/processed/credit_default_cleaned.parquet")


def download_raw(dest: Path = RAW_PATH) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("Raw file already exists at %s, skipping download.", dest)
        return dest
    logger.info("Downloading UCI dataset...")
    response = requests.get(UCI_URL, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)
    logger.info("Saved to %s", dest)
    return dest


def load_raw(file_path: Path = RAW_PATH) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    # header=1: row 0 is metadata, row 1 has actual column names
    # index_col=0: column ID is the row identifier, not a feature
    df = pd.read_excel(file_path, header=1, index_col=0)
    logger.info("Loaded %d rows, %d columns from %s", *df.shape, file_path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Rename target to a short, consistent name used across all notebooks/modules
    df = df.rename(columns={"default payment next month": "default"})

    # EDUCATION: undocumented values 0, 5, 6 → bin into 4 (others)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

    # MARRIAGE: undocumented value 0 → bin into 3 (others)
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    logger.info("Cleaning complete. Shape: %s", df.shape)
    return df


def load_data(raw_path: Path = RAW_PATH, processed_path: Path = PROCESSED_PATH) -> pd.DataFrame:
    """
    Full pipeline: download if needed → load → clean → save parquet → return DataFrame.
    All downstream code should call this single entry point.
    """
    download_raw(raw_path)
    df = load_raw(raw_path)
    df = clean(df)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_path, index=True)
    logger.info("Saved cleaned data to %s", processed_path)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = load_data()
    print(df.shape)
    print(df.head())
