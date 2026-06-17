"""
Day 2 placeholder: tests for data ingestion.
Will be filled in on Day 3 when src/data/loader.py is built.
"""
import pytest
from pathlib import Path


def test_raw_data_dir_exists():
    assert Path("data/raw").exists()


def test_processed_data_dir_exists():
    assert Path("data/processed").exists()
