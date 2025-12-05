"""
Basic metrics for H-drift on the politeness dataset.

This module:
- loads data/processed/stanford_politeness_h_drift.parquet
- defines a simple H_drift_index
- prints summary stats
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "stanford_politeness_h_drift.parquet"


def load_politeness_h_drift(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the H-drift feature table for the politeness dataset.
    """
    p = path or PROCESSED_PATH
    if not p.exists():
        raise FileNotFoundError(f"H-drift parquet not found at {p}")
    df = pd.read_parquet(p)
    return df


def add_h_drift_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a simple H_drift_index column.

    For v1, we define:
        H_drift_index = h_total

    Later this can be normalized or combined with other features.
    """
    if "h_total" not in df.columns:
        raise ValueError("Expected column 'h_total' in dataframe.")
    df = df.copy()
    df["H_drift_index"] = df["h_total"].astype(float)
    return df


def summarize(df: pd.DataFrame) -> None:
    """
    Print basic summary statistics for H_drift_index and counts,
    optionally grouped by politeness_label if present.
    """
    print("\n=== Overall H-drift summary ===")
    print(df[["h1_emotion", "h2_relational", "h3_hedging", "h4_anthro", "h5_softeners", "H_drift_index"]].describe())

    if "politeness_label" in df.columns:
        print("\n=== H_drift_index by politeness_label ===")
        print(df.groupby("politeness_label")["H_drift_index"].describe())


def main() -> None:
    df = load_politeness_h_drift()
    df = add_h_drift_index(df)
    summarize(df)


if __name__ == "__main__":
    main()
