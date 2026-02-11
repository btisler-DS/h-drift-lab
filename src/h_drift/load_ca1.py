from pathlib import Path
from datasets import load_dataset
import pandas as pd


RAW_DIR = Path("data/raw/collective_alignment_1")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Load the comparisons config (one row per prompt)
    ds = load_dataset(
        "openai/collective-alignment-1",
        name="comparisons",
        split="train",
    )

    df = ds.to_pandas()

    out_path = RAW_DIR / "comparisons.parquet"
    df.to_parquet(out_path, index=False)

    print(f"Wrote {len(df)} rows to {out_path} (CA-1 comparisons)")


if __name__ == "__main__":
    main()
