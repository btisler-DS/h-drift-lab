from pathlib import Path
import pandas as pd

from src.h_drift.feats_lexicon import count_feats_tokens

IN_PATH = Path("data/processed/ca1_responses.parquet")
OUT_PATH = Path("data/processed/ca1_responses_feats.parquet")


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing CA-1 responses at {IN_PATH}")

    df = pd.read_parquet(IN_PATH)
    print(f"Loaded {len(df)} CA-1 responses from {IN_PATH}")

    feats = df["response_text"].apply(count_feats_tokens).apply(pd.Series)
    feats["FEATS_total"] = (
        feats["F_feelings"]
        + feats["E_expressions"]
        + feats["A_actions"]
        + feats["T_thoughts"]
        + feats["S_sensations"]
    )

    out_df = pd.concat([df, feats], axis=1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT_PATH, index=False)

    print(f"Wrote {len(out_df)} rows with FEATS to {OUT_PATH}")
    print(out_df.head(5))


if __name__ == "__main__":
    main()
