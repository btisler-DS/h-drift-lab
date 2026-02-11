"""
Augment HH-RLHF H-drift data with FEATS counts.

Input:
    data/processed/hh_rlhf_h_drift.parquet
        expected columns (among others):
            - pair_id
            - label  (e.g., 'chosen' / 'rejected')
            - some text column (see detection logic below)
            - h1_emotion, ..., h5_softeners, h_total

Output:
    data/processed/hh_rlhf_h_drift_feats.parquet
        same rows + columns:
            - F_feelings
            - E_expressions
            - A_actions
            - T_thoughts
            - S_sensations

This is deliberately simple and fully inspectable so that
other researchers can swap in different FEATS lexica and
compare results.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from src.h_drift.feats_lexicon import count_feats_tokens

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "hh_rlhf_h_drift.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "hh_rlhf_h_drift_feats.parquet"


def detect_text_column(df: pd.DataFrame) -> str:
    """
    Try to detect which column contains the response text.

    We keep this explicit and small. If none is found, we raise
    with a helpful error listing available columns, so the user
    can adjust this function.
    """
    candidates: List[str] = [
        "response",
        "text",
        "assistant",
        "completion",
        "output",
        "chosen_text",
        "rejected_text",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not detect text column. Looked for {candidates}, "
        f"but columns are: {list(df.columns)}"
    )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input parquet not found at {INPUT_PATH}")

    print(f"Loading H-drift data from: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} rows.")

    text_col = detect_text_column(df)
    print(f"Detected text column: {text_col}")

    # Apply FEATS counter row-wise; expand dict to columns
    print("Computing FEATS counts... (this may take a bit)")
    feats_df = df[text_col].astype(str).apply(count_feats_tokens).apply(pd.Series)

    # Ensure stable column order
    feats_df = feats_df[
        ["F_feelings", "E_expressions", "A_actions", "T_thoughts", "S_sensations"]
    ]

    out_df = pd.concat([df, feats_df], axis=1)

    print("Sample of augmented columns:")
    print(
        out_df[
            [
                "pair_id",
                "label",
                text_col,
                "F_feelings",
                "E_expressions",
                "A_actions",
                "T_thoughts",
                "S_sensations",
            ]
        ]
        .head(5)
        .to_string(index=False)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUTPUT_PATH)
    print(f"\nWrote augmented dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
