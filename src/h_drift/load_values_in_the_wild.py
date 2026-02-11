from pathlib import Path
from datasets import load_dataset
import pandas as pd


RAW_DIR = Path("data/raw/values_in_the_wild")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Main config name may be "values_frequencies" in the HF card
    ds = load_dataset(
        "Anthropic/values-in-the-wild",
        name="values_frequencies",
        split="train",
    )

    df = ds.to_pandas()

    out_path = RAW_DIR / "values_frequencies.parquet"
    df.to_parquet(out_path, index=False)

    print(f"Wrote {len(df)} rows to {out_path} (Anthropic Values in the Wild)")


if __name__ == "__main__":
    main()
