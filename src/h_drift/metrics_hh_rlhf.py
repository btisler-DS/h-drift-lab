"""
Metrics for H-drift on the Anthropic HH-RLHF dataset.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "hh_rlhf_h_drift.parquet"


def main() -> None:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f"H-drift parquet not found at {PROCESSED_PATH}")

    df = pd.read_parquet(PROCESSED_PATH)
    df = df.copy()
    df["H_drift_index"] = df["h_total"].astype(float)

    print("=== Overall H-drift summary (HH-RLHF) ===")
    print(
        df[
            [
                "h1_emotion",
                "h2_relational",
                "h3_hedging",
                "h4_anthro",
                "h5_softeners",
                "H_drift_index",
            ]
        ].describe()
    )

    print("\n=== H_drift_index by label (chosen vs rejected) ===")
    print(df.groupby("label")["H_drift_index"].describe())


if __name__ == "__main__":
    main()
