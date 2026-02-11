import pandas as pd
from pathlib import Path

IN_PATH = Path("data/processed/webgpt_pairs_feats.parquet")


def main():
    df = pd.read_parquet(IN_PATH)
    print(f"Loaded {len(df)} rows from {IN_PATH}")

    # pivot to chosen / rejected per pair
    wide = df.pivot(index="pair_id", columns="label")

    # helper to pull a column safely
    def col(name):
        return wide[(name, "chosen")], wide[(name, "rejected")]

    # build deltas for each FEATS dimension
    deltas = {}
    feats = ["F_feelings", "E_expressions", "A_actions", "T_thoughts", "S_sensations"]

    for feat in feats:
        chosen, rejected = col(feat)
        delta = chosen.fillna(0) - rejected.fillna(0)
        deltas[f"delta_{feat}"] = delta

    # also total FEATS count
    chosen_total = sum(col(f)[0].fillna(0) for f in feats)
    rejected_total = sum(col(f)[1].fillna(0) for f in feats)
    deltas["delta_FEATS_total"] = chosen_total - rejected_total

    delta_df = pd.DataFrame(deltas)
    print(f"Computed deltas for {len(delta_df)} pairs.")

    for feat in feats + ["FEATS_total"]:
        colname = f"delta_{feat}"
        series = delta_df[colname]
        print("\n" + "=" * 72)
        print(f"Feature: {feat}")
        print("-" * 72)
        print("Summary of Δ (chosen - rejected):")
        print(series.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))

        pos = (series > 0).sum()
        neg = (series < 0).sum()
        zero = (series == 0).sum()
        total = len(series)

        print("\nDirection of preference:")
        print(f"  Chosen more {feat:14} (Δ > 0): {pos:7d}  ({pos/total:0.3f})")
        print(f"  Chosen less {feat:14} (Δ < 0): {neg:7d}  ({neg/total:0.3f})")
        print(f"  No change                (|Δ|≈0): {zero:7d}  ({zero/total:0.3f})")

        print("\nCentral tendency:")
        print(f"  Mean Δ{feat}:   {series.mean():0.4f}")
        print(f"  Median Δ{feat}: {series.median():0.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
