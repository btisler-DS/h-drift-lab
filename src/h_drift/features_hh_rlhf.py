"""
H-drift feature extraction for the Anthropic HH-RLHF dataset.

This script:
- downloads Anthropic/hh-rlhf via HuggingFace datasets,
- flattens each pair into two rows (chosen, rejected),
- constructs a 'prompt' field by extracting all Human: turns
  from the conversation text,
- computes H-class counts on the assistant response text,
- writes data/processed/hh_rlhf_h_drift.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict
import re

import pandas as pd
from datasets import load_dataset

# --- Project paths -----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from h_drift.lexicon import count_h_tokens  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def extract_human_prompt(example: Dict[str, str]) -> str:
    """
    Extract all 'Human:' segments from the HH-RLHF conversation text.

    HH-RLHF stores conversation as something like:

        "\\n\\nHuman: ...\\n\\nAssistant: ...\\n\\nHuman: ...\\n\\nAssistant: ..."

    We treat the concatenation of all Human turns as the 'prompt'
    (question-side / user-side text) for Ω geometry.

    We use the 'chosen' field preferentially; if that is empty
    we fall back to 'rejected'.
    """
    raw = example.get("chosen", "") or example.get("rejected", "") or ""

    # Find all segments between 'Human:' and the next 'Assistant:' or end of string.
    matches = re.findall(r"Human:(.*?)(?=Assistant:|$)", raw, flags=re.S)
    parts = [m.strip() for m in matches if m.strip()]

    return "\n\n".join(parts)


def build_hh_rlhf_features() -> pd.DataFrame:
    """
    Load Anthropic/hh-rlhf and build an H-drift feature table.

    Each original example has:
      - chosen   (preferred assistant reply, includes Human/Assistant turns)
      - rejected (non-preferred assistant reply, same format)

    We flatten to:
      - one row per (pair_id, label in {chosen, rejected})

    Fields per row:
      - pair_id           : int
      - prompt            : concatenated Human turns (question-side text)
      - response          : assistant reply text (chosen or rejected)
      - label             : 'chosen' or 'rejected'
      - h1_emotion        : H1 count on response
      - h2_relational     : H2 count on response
      - h3_hedging        : H3 count on response
      - h4_anthro         : H4 count on response
      - h5_softeners      : H5 count on response
      - h_total           : sum of H1–H5
    """
    ds = load_dataset("Anthropic/hh-rlhf", split="train")

    rows = []
    for pair_id, ex in enumerate(ds):
        # Build a user-side prompt from all Human: turns in the conversation
        prompt = extract_human_prompt(ex)

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
