"""
Per-feature within-pair H-drift deltas for Anthropic HH-RLHF.

For each (pair_id), compute for each feature f in:
    [h1_emotion, h2_relational, h3_hedging, h4_anthro, h5_softeners, h_total]:

    f_chosen  = f(chosen)
    f_reject  = f(rejected)
    Δf        = f_chosen - f_reject

Then:
- print summary stats for each Δf
- report how often RLHF prefers more vs less of each feature
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "hh_rlhf_h_drift.parquet"


FEATURES: List[str] = [
    "h1_emotion",
    "h2_relational",
    "h3_hedging",
    "h4_anthro",
    "h5_softeners",
    "h_total",  # overall H-load
]


def main() -> None:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f"H-drift parquet not found at {PROCESSED_PATH}")

    df = pd.read_parquet(PROCESSED_PATH).copy()

    # Sanity: expect 'label' with values 'chosen' and 'rejected'
    expected_labels = {"chosen", "rejected"}
    labels_present = set(df["label"].unique())
    if not expected_labels.issubset(labels_present):
        raise ValueError(f"Expected labels {expected_labels}, got {labels_present}")

    # Split into chosen / rejected tables keyed by pair_id
    chosen = df[df["label"] == "chosen"].set_index("pair_id")
    rejected = df[df["label"] == "rejected"].set_index("pair_id")

    # Inner join: only pairs where we have both
    merged = chosen[FEATURES].join(
        rejected[FEATURES],
        how="inner",
        lsuffix="_chosen",
        rsuffix="_rejected",
    )

    eps = 1e-9
    total_pairs = len(merged)
    print(f"Total pairs with both chosen & rejected: {total_pairs}\n")

    for feat in FEATURES:
        c_col = f"{feat}_chosen"
        r_col = f"{feat}_rejected"
        d_col = f"delta_{feat}"

        merged[d_col] = merged[c_col] - merged[r_col]

        deltas = merged[d_col]

        print("=" * 72)
        print(f"Feature: {feat}")
        print("-" * 72)
        print("Summary of Δ (chosen - rejected):")
        print(
            deltas.describe(
                percentiles=[0.1, 0.25, 0.5, 0.75, 0.9],
            )
        )

        more = (deltas > eps).sum()
        less = (deltas < -eps).sum()
        equal = ((deltas >= -eps) & (deltas <= eps)).sum()

        print("\nDirection of preference:")
        print(f"  Chosen more {feat:<12} (Δ > 0): {more:7d}  ({more / total_pairs:0.3f})")
        print(f"  Chosen less {feat:<12} (Δ < 0): {less:7d}  ({less / total_pairs:0.3f})")
        print(f"  No change                (|Δ|≈0): {equal:7d}  ({equal / total_pairs:0.3f})")

        mean_delta = deltas.mean()
        median_delta = deltas.median()
        print("\nCentral tendency:")
        print(f"  Mean Δ{feat}:   {mean_delta:0.4f}")
        print(f"  Median Δ{feat}: {median_delta:0.4f}")
        print()

    print("=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
