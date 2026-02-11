from pathlib import Path
import pandas as pd

from src.h_drift.feats_lexicon import count_feats_tokens

IN_PATH = Path("data/processed/webgpt_pairs.parquet")
OUT_PATH = Path("data/processed/webgpt_pairs_feats.parquet")


def main():
    df = pd.read_parquet(IN_PATH)
    print(f"Loaded {len(df)} rows from {IN_PATH}")

    # Apply FEATS lexicon to each response
    feats_df = df["response"].apply(count_feats_tokens).apply(pd.Series)

    # Expect columns: F_feelings, E_expressions, A_actions, T_thoughts, S_sensations
    print("FEATS columns:", list(feats_df.columns))

    out_df = pd.concat([df, feats_df], axis=1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote WebGPT FEATS-augmented pairs to: {OUT_PATH}")


if __name__ == "__main__":
    main()
