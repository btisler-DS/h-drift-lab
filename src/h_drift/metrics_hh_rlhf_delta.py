"""
Within-pair H-drift deltas for Anthropic HH-RLHF.

For each (pair_id), compute:
    H_chosen  = H_drift_index(chosen)
    H_reject  = H_drift_index(rejected)
    ΔH        = H_chosen - H_reject

Then:
- summarize ΔH distribution
- report how often RLHF prefers more vs less humanistic drift
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "hh_rlhf_h_drift.parquet"


def main() -> None:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f"H-drift parquet not found at {PROCESSED_PATH}")

    df = pd.read_parquet(PROCESSED_PATH).copy()

    # Define H_drift_index if not already present
    if "H_drift_index" not in df.columns:
        df["H_drift_index"] = df["h_total"].astype(float)

    # Sanity: we expect exactly two rows per pair_id: chosen + rejected
    # Pivot to wide format: one row per pair_id
    wide = (
        df.pivot_table(
            index="pair_id",
            columns="label",
            values="H_drift_index",
            aggfunc="first",
        )
        .reset_index()
    )

    # Keep only pairs where we have both chosen and rejected
    if not {"chosen", "rejected"}.issubset(wide.columns):
        missing = {"chosen", "rejected"} - set(wide.columns)
        raise ValueError(f"Expected columns 'chosen' and 'rejected' in pivot; missing: {missing}")

    wide = wide.dropna(subset=["chosen", "rejected"])

    wide["delta_H"] = wide["chosen"] - wide["rejected"]

    print("=== ΔH = H_chosen - H_rejected (per pair) ===")
    print(wide["delta_H"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))

    # Direction counts
    eps = 1e-9
    more_humanistic = (wide["delta_H"] > eps).sum()
    less_humanistic = (wide["delta_H"] < -eps).sum()
    equal = ((wide["delta_H"] >= -eps) & (wide["delta_H"] <= eps)).sum()
    total = len(wide)

    print("\n=== Direction of drift preference (counts & proportions) ===")
    print(f"Total pairs: {total}")
    print(f"Chosen more humanistic   (ΔH > 0):  {more_humanistic}  ({more_humanistic / total:.3f})")
    print(f"Chosen less humanistic  (ΔH < 0):  {less_humanistic}  ({less_humanistic / total:.3f})")
    print(f"No change (|ΔH| ≈ 0):              {equal}  ({equal / total:.3f})")

    # Simple mean / skew summary
    mean_delta = wide["delta_H"].mean()
    median_delta = wide["delta_H"].median()
    print("\n=== Central tendency of ΔH ===")
    print(f"Mean ΔH:   {mean_delta:.4f}")
    print(f"Median ΔH: {median_delta:.4f}")


if __name__ == "__main__":
    main()
