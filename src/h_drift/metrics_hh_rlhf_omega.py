"""
metrics_hh_rlhf_omega.py

Attach WWWWHW (Ω) features to the HH-RLHF H-drift dataset using the
*user prompt* text (question) instead of assistant responses.

This script expects the processed HH-RLHF dataset to contain:

  - 'prompt'           : user question / instruction
  - 'response'         : assistant answer text
  - 'H_drift_index'    : precomputed H-drift score (optional but recommended)
  - 'label'            : 'chosen' / 'rejected' (optional but recommended)

It produces:

  data/processed/hh_rlhf_h_drift_omega.parquet

where each row has additional Ω-features:

  omega_who, ..., omega_how
  omega_entropy
  omega_total_markers
  omega_dominant_dim
"""

import pandas as pd
import numpy as np

from src.h_drift.omega import (
    OMEGA_DIMENSIONS,
    extract_omega_vector,
    compute_omega_entropy,
)


INPUT_PATH = "data/processed/hh_rlhf_h_drift.parquet"
OUTPUT_PATH = "data/processed/hh_rlhf_h_drift_omega.parquet"

# We now explicitly intend to analyze *user questions*, not answers.
TEXT_COL = "prompt"        # HH-RLHF user question column
HDRIFT_COL = "H_drift_index"
LABEL_COL = "label"        # 'chosen' / 'rejected'


def compute_omega_row(text: str) -> pd.Series:
    """
    Given a text (prompt), return a Series with:
      omega_who, ..., omega_how,
      omega_entropy,
      omega_total_markers,
      omega_dominant_dim
    """
    if not isinstance(text, str) or not text.strip():
        omega = np.zeros(len(OMEGA_DIMENSIONS), dtype=float)
        H_omega = 0.0
        total = 0
        dominant_dim = "none"
    else:
        omega = extract_omega_vector(text)
        H_omega = compute_omega_entropy(omega)
        total = int(omega.sum())
        if total > 0:
            dominant_dim = OMEGA_DIMENSIONS[int(omega.argmax())]
        else:
            dominant_dim = "none"

    data = {
        f"omega_{dim}": omega[i] for i, dim in enumerate(OMEGA_DIMENSIONS)
    }
    data.update(
        {
            "omega_entropy": float(H_omega),
            "omega_total_markers": int(total),
            "omega_dominant_dim": dominant_dim,
        }
    )
    return pd.Series(data)


def main() -> None:
    print(f"Loading HH-RLHF H-drift data from: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} rows.")
    print("Available columns:", list(df.columns))

    # --- Sanity check: prompt column must exist ---
    if TEXT_COL not in df.columns:
        raise RuntimeError(
            f"Expected prompt column '{TEXT_COL}' not found in data.\n"
            f"Current columns: {list(df.columns)}\n"
            "You need to reload / preprocess HH-RLHF so that the user question "
            "is stored in a 'prompt' column."
        )

    print(f"Computing Ω features from column: '{TEXT_COL}' (user prompts) ...")
    omega_df = df[TEXT_COL].apply(compute_omega_row)
    df_omega = pd.concat([df, omega_df], axis=1)

    print(f"Writing Ω-augmented dataset to: {OUTPUT_PATH}")
    df_omega.to_parquet(OUTPUT_PATH, index=False)

    # --- Basic summaries ---

    print("\n=== Ω dominant dimension counts (by prompt) ===")
    print(df_omega["omega_dominant_dim"].value_counts(dropna=False))

    # Mean H_drift by Ω-dominant dimension (if available)
    if HDRIFT_COL in df_omega.columns:
        print("\n=== Mean H_drift_index by Ω-dominant dimension (prompt-based) ===")
        print(
            df_omega.groupby("omega_dominant_dim")[HDRIFT_COL]
            .mean()
            .sort_values(ascending=False)
        )

        # WHY vs HOW comparison (all rows)
        why_mask = df_omega["omega_dominant_dim"] == "why"
        how_mask = df_omega["omega_dominant_dim"] == "how"

        if why_mask.any() and how_mask.any():
            mean_why = df_omega.loc[why_mask, HDRIFT_COL].mean()
            mean_how = df_omega.loc[how_mask, HDRIFT_COL].mean()
            print("\n=== WHY vs HOW H_drift_index (prompt-based, all rows) ===")
            print(f"WHY-dominant mean H_drift_index: {mean_why:.4f}")
            print(f"HOW-dominant mean H_drift_index: {mean_how:.4f}")
        else:
            print("\n[INFO] Not enough WHY/HOW-dominant prompts for comparison.")

    # Optional: stratify by 'label' if present (chosen vs rejected)
    if LABEL_COL in df_omega.columns and HDRIFT_COL in df_omega.columns:
        print("\n=== Mean H_drift_index by (label, Ω-dominant prompt dimension) ===")
        grouped = (
            df_omega.groupby([LABEL_COL, "omega_dominant_dim"])[HDRIFT_COL]
            .mean()
            .unstack(fill_value=float("nan"))
        )
        print(grouped)


if __name__ == "__main__":
    main()
