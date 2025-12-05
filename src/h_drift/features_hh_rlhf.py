"""
H-drift feature extraction for the Anthropic HH-RLHF dataset.

This script:
- downloads Anthropic/hh-rlhf via HuggingFace datasets,
- flattens each pair into two rows (chosen, rejected),
- computes H-class counts on the response text,
- writes data/processed/hh_rlhf_h_drift.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import pandas as pd
from datasets import load_dataset

# --- Project paths -----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from h_drift.lexicon import count_h_tokens  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def build_hh_rlhf_features() -> pd.DataFrame:
    """
    Load Anthropic/hh-rlhf and build an H-drift feature table.

    Each original example has:
      - prompt
      - chosen  (preferred assistant reply)
      - rejected (non-preferred assistant reply)

    We flatten to:
      - one row per (pair_id, label in {chosen,rejected})
    with H-class counts computed on the response text.
    """
    ds = load_dataset("Anthropic/hh-rlhf", split="train")

    rows = []
    for pair_id, ex in enumerate(ds):
        prompt = ex.get("prompt", "")

        for label in ("chosen", "rejected"):
            text = ex.get(label, "") or ""
            counts: Dict[str, int] = count_h_tokens(text)

            row = {
                "pair_id": pair_id,
                "prompt": prompt,
                "response": text,
                "label": label,
                "h1_emotion": counts["H1_emotion"],
                "h2_relational": counts["H2_relational"],
                "h3_hedging": counts["H3_hedging"],
                "h4_anthro": counts["H4_anthro"],
                "h5_softeners": counts["H5_softeners"],
            }
            row["h_total"] = (
                row["h1_emotion"]
                + row["h2_relational"]
                + row["h3_hedging"]
                + row["h4_anthro"]
                + row["h5_softeners"]
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = build_hh_rlhf_features()
    out_path = PROCESSED_DIR / "hh_rlhf_h_drift.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
