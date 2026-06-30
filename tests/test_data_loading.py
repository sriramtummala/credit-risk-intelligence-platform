"""Tests for data ingestion."""
import pytest
from pathlib import Path


def test_raw_data_dir_exists():
    assert Path("data/raw").exists()


def test_processed_data_dir_exists():
    assert Path("data/processed").exists()
