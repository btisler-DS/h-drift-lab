"""
Feature extraction for a CSV version of the Stanford Politeness Corpus.

Expected input:
- CSV file at data/raw/stanford_politeness/stanford_politeness.csv
- Contains at least one text column with the request.

This script:
- loads the CSV,
- computes H-class counts for each row,
- writes a text-free feature table to data/processed/stanford_politeness_h_drift.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# --- Project paths -----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_POLITENESS_CSV = (
    PROJECT_ROOT / "data" / "raw" / "stanford_politeness" / "stanford_politeness.csv"
)

from h_drift.lexicon import count_h_tokens  # noqa: E402


# --- Helpers -----------------------------------------------------------------


def detect_text_column(df: pd.DataFrame) -> str:
    """
    Try to guess which column contains the request text.
    """
    candidates = ["text", "Request", "request", "sentence", "utterance"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find a text column in columns: {list(df.columns)}")


def detect_label_column(df: pd.DataFrame) -> Optional[str]:
    """
    Try to guess which column contains a politeness label, if any.
    """
    candidates = ["label", "is_polite", "politeness", "Binary", "y"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


# --- Core logic --------------------------------------------------------------


def build_h_drift_features() -> pd.DataFrame:
    """
    Build an H-drift feature table from the CSV politeness dataset.

    Output columns:
    - row_id (original index)
    - politeness_label (if available)
    - h1_emotion, h2_relational, h3_hedging, h4_anthro, h5_softeners
    - h_total (sum of all H-class counts)
    """
    if not RAW_POLITENESS_CSV.exists():
        raise FileNotFoundError(
            f"Expected CSV at {RAW_POLITENESS_CSV}, but it does not exist.\n"
            "Place your Stanford Politeness CSV there as 'stanford_politeness.csv'."
        )

    df_in = pd.read_csv(RAW_POLITENESS_CSV)
    text_col = detect_text_column(df_in)
    label_col = detect_label_column(df_in)

    rows = []
    for idx, row in df_in.iterrows():
        text = str(row[text_col]) if not pd.isna(row[text_col]) else ""
        counts: Dict[str, int] = count_h_tokens(text)

        out_row = {
            "row_id": idx,
            "h1_emotion": counts["H1_emotion"],
            "h2_relational": counts["H2_relational"],
            "h3_hedging": counts["H3_hedging"],
            "h4_anthro": counts["H4_anthro"],
            "h5_softeners": counts["H5_softeners"],
        }
        out_row["h_total"] = (
            out_row["h1_emotion"]
            + out_row["h2_relational"]
            + out_row["h3_hedging"]
            + out_row["h4_anthro"]
            + out_row["h5_softeners"]
        )

        if label_col is not None:
            out_row["politeness_label"] = row[label_col]
        else:
            out_row["politeness_label"] = None

        rows.append(out_row)

    df_out = pd.DataFrame(rows)
    return df_out


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = build_h_drift_features()
    out_path = PROCESSED_DIR / "stanford_politeness_h_drift.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
