# src/h_drift/predictive_entropy_drift.py

"""
Predictive relationship between interrogative entropy (omega_entropy = H_i)
and affect drift (h_total) using the HH-RLHF drift dataset enriched with
Omega entropy signals.

This script:
  1. Loads: data/processed/hh_rlhf_h_drift_omega.parquet
  2. Prints basic distribution stats for omega_entropy and h_total
  3. Uses quartiles to define low- and high-entropy groups
  4. Compares mean drift between low and high entropy
  5. Runs an OLS regression: h_total ~ omega_entropy
"""

import pandas as pd
from pathlib import Path
import statsmodels.api as sm


# ------------------------------------------------
# Paths
# ------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"

PARQUET = PROCESSED / "hh_rlhf_h_drift_omega.parquet"


# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    print(f"Loading parquet: {PARQUET}")
    df = pd.read_parquet(PARQUET)

    print("\nColumns found in dataframe:")
    for c in sorted(df.columns):
        print("  -", c)

    # ------------------------------------------------
    # Validate columns
    # ------------------------------------------------
    required = ["omega_entropy", "h_total"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(
            f"\nERROR: Expected columns missing: {missing}\n"
            f"Your parquet must contain omega_entropy and h_total."
        )

    # ------------------------------------------------
    # Basic distribution diagnostics
    # ------------------------------------------------
    print("\n=== Basic Distribution Stats ===")
    print("\nomega_entropy:")
    print(df["omega_entropy"].describe(percentiles=[0.0, 0.25, 0.5, 0.75, 1.0]))

    print("\nh_total:")
    print(df["h_total"].describe())

    # How many exact zeros in omega_entropy?
    zero_count = (df["omega_entropy"].abs() < 1e-9).sum()
    print(f"\nNumber of prompts with omega_entropy ~ 0: {zero_count} "
          f"({zero_count / len(df):.3%} of all prompts)")

    # ------------------------------------------------
    # Quartile split: low vs high entropy
    # ------------------------------------------------
    q25 = df["omega_entropy"].quantile(0.25)
    q75 = df["omega_entropy"].quantile(0.75)

    low = df[df["omega_entropy"] <= q25]
    high = df[df["omega_entropy"] >= q75]

    print("\n=== Quartile Split on omega_entropy ===")
    print(f"25th percentile (Q1): {q25:.6f}")
    print(f"75th percentile (Q3): {q75:.6f}")
    print(f"Low-entropy group:  n={len(low)}   mean drift = {low['h_total'].mean():.6f}")
    print(f"High-entropy group: n={len(high)}  mean drift = {high['h_total'].mean():.6f}")

    drift_diff = high["h_total"].mean() - low["h_total"].mean()
    print(f"Difference (high - low): {drift_diff:.6f}")

    # ------------------------------------------------
    # Regression: h_total ~ omega_entropy
    # ------------------------------------------------
    print("\n=== OLS Regression: h_total ~ omega_entropy ===")
    X = sm.add_constant(df["omega_entropy"])
    y = df["h_total"]
    model = sm.OLS(y, X).fit()
    print(model.summary())

    # ------------------------------------------------
    # Optional: save subset summary for plotting
    # ------------------------------------------------
    out_path = PROCESSED / "entropy_drift_summary.tsv"
    df_out = pd.DataFrame({
        "omega_entropy": df["omega_entropy"],
        "h_total": df["h_total"]
    })
    df_out.to_csv(out_path, sep="\t", index=False)
    print(f"\nWrote summary table: {out_path}")


# ------------------------------------------------
# Entrypoint
# ------------------------------------------------
if __name__ == "__main__":
    main()
