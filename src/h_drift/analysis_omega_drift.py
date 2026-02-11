# src/h_drift/analysis_omega_drift.py
"""
Compare H-drift patterns across interrogative dimensions.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"

def analyze_omega_vs_drift():
    """Compare H-drift across Ω dimensions."""
    
    df = pd.read_parquet(PROCESSED / "hh_rlhf_h_drift_omega.parquet")
    
    # Compute H-drift index as sum of all H-class markers
    h_cols = [col for col in df.columns if col.startswith('h') and '_' in col]
    df['H_drift_total'] = df[h_cols].sum(axis=1)
    
    print(f"\nH-class columns found: {h_cols}\n")
    
    print("\n=== H-drift by Ω dimension ===\n")
    
    # Group by dominant dimension
    grouped = df.groupby('omega_dominant_dim').agg({
        'H_drift_total': ['mean', 'std', 'count'],
        'h3_hedging': 'mean',
        'h4_anthro': 'mean',
        'h5_softeners': 'mean',
        'omega_entropy': 'mean'
    }).round(4)
    
    print(grouped)
    print("\n")
    
    # Compare WHY vs HOW specifically
    why_drift = df[df['omega_dominant_dim'] == 'why']['H_drift_total']
    how_drift = df[df['omega_dominant_dim'] == 'how']['H_drift_total']
    what_drift = df[df['omega_dominant_dim'] == 'what']['H_drift_total']
    
    print("=== WHY vs HOW vs WHAT comparison ===")
    print(f"WHY mean H-drift: {why_drift.mean():.4f} (n={len(why_drift)})")
    print(f"HOW mean H-drift: {how_drift.mean():.4f} (n={len(how_drift)})")
    print(f"WHAT mean H-drift: {what_drift.mean():.4f} (n={len(what_drift)})")
    print()
    
    # Statistical test
    from scipy import stats
    if len(why_drift) > 0 and len(how_drift) > 0:
        why_vs_how = stats.mannwhitneyu(why_drift, how_drift, alternative='two-sided')
        why_vs_what = stats.mannwhitneyu(why_drift, what_drift, alternative='two-sided')
        
        print(f"WHY vs HOW: U={why_vs_how.statistic:.0f}, p={why_vs_how.pvalue:.4f}")
        print(f"WHY vs WHAT: U={why_vs_what.statistic:.0f}, p={why_vs_what.pvalue:.4f}")
        print()
    
    # Omega entropy vs H-drift correlation
    corr = df[['omega_entropy', 'H_drift_total']].corr().iloc[0, 1]
    print(f"Correlation (Ω entropy vs H-drift): {corr:.4f}")
    print()
    
    # High entropy vs low entropy
    high_entropy = df[df['omega_entropy'] > df['omega_entropy'].median()]
    low_entropy = df[df['omega_entropy'] <= df['omega_entropy'].median()]
    
    print("=== Entropy split ===")
    print(f"High Ω-entropy (mixed WH): mean H-drift = {high_entropy['H_drift_total'].mean():.4f}")
    print(f"Low Ω-entropy (focused WH): mean H-drift = {low_entropy['H_drift_total'].mean():.4f}")
    
    # Show individual H-components for WHY vs others
    print("\n=== H-component breakdown (WHY vs HOW vs WHAT) ===")
    for h_col in ['h1_emotion', 'h2_relational', 'h3_hedging', 'h4_anthro', 'h5_softeners']:
        if h_col in df.columns:
            why_val = df[df['omega_dominant_dim'] == 'why'][h_col].mean()
            how_val = df[df['omega_dominant_dim'] == 'how'][h_col].mean()
            what_val = df[df['omega_dominant_dim'] == 'what'][h_col].mean()
            print(f"{h_col:15s} WHY:{why_val:7.4f}  HOW:{how_val:7.4f}  WHAT:{what_val:7.4f}")

if __name__ == "__main__":
    analyze_omega_vs_drift()