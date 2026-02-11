from pathlib import Path
import pandas as pd


RAW_PATH = Path("data/raw/webgpt_reward_model/0000.parquet")
OUT_PATH = Path("data/processed/webgpt_pairs.parquet")


def build_webgpt_pairs():
    # Load the raw WebGPT Parquet shard
    df = pd.read_parquet(RAW_PATH)
    print("Raw columns:", list(df.columns))

    rows = []

    for idx, row in df.iterrows():
        # WebGPT has two answers and two scores per example
        try:
            ans0 = row["answer_0"]
            ans1 = row["answer_1"]
            s0 = float(row["score_0"])
            s1 = float(row["score_1"])
        except KeyError as e:
            raise KeyError(f"Expected WebGPT columns missing: {e}. Got columns: {list(df.columns)}")

        # Skip if any answer is missing
        if pd.isna(ans0) or pd.isna(ans1):
            continue

        # Skip ties (no clear preference)
        if s0 == s1:
            continue

        if s0 > s1:
            chosen_text, rejected_text = ans0, ans1
        else:
            chosen_text, rejected_text = ans1, ans0

        pair_id = int(idx)

        rows.append(
            {
                "pair_id": pair_id,
                "label": "chosen",
                "response": chosen_text,
            }
        )
        rows.append(
            {
                "pair_id": pair_id,
                "label": "rejected",
                "response": rejected_text,
            }
        )

    out_df = pd.DataFrame(rows)
    print(f"Built {len(out_df)} rows from {len(df)} WebGPT items.")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote WebGPT pairs to: {OUT_PATH}")


if __name__ == "__main__":
    build_webgpt_pairs()
