"""
omega.py — Minimal Ω-classifier for WWWWHW geometry.

Conventions:
  Ω = [who, what, when, where, why, how]

This is a *rule-based* v1 intended for demonstration and coupling
with H-drift metrics. It is not a production-grade parser.
"""

from __future__ import annotations
from typing import Dict, Any
import re
import numpy as np


# Order matters: this defines the Ω index mapping.
OMEGA_DIMENSIONS = ["who", "what", "when", "where", "why", "how"]

OMEGA_MARKERS: Dict[str, list[str]] = {
    "who":   ["who", "whom", "whose", "who's"],
    "what":  ["what", "what's", "which"],
    "when":  ["when", "when's"],
    "where": ["where", "where's"],
    "why":   ["why", "why's", "how come"],
    "how":   ["how", "how's"],
}


def extract_omega_vector(text: str) -> np.ndarray:
    """
    Extract raw Ω counts from text.

    Returns:
        np.ndarray shape (6,) with counts for
        [who, what, when, where, why, how]
    """
    text_lower = text.lower()
    omega = np.zeros(len(OMEGA_DIMENSIONS), dtype=float)

    for idx, dim in enumerate(OMEGA_DIMENSIONS):
        markers = OMEGA_MARKERS[dim]
        for marker in markers:
            pattern = r"\b" + re.escape(marker) + r"\b"
            omega[idx] += len(re.findall(pattern, text_lower))

    return omega


def omega_to_distribution(omega: np.ndarray) -> np.ndarray:
    """
    Normalize Ω vector to a probability distribution.

    If all counts are zero, returns a uniform distribution
    (this corresponds to 'no explicit interrogative marker').
    """
    total = float(omega.sum())
    if total <= 0.0:
        return np.ones_like(omega, dtype=float) / len(omega)
    return omega / total


def compute_omega_entropy(omega: np.ndarray) -> float:
    """
    Compute Shannon entropy H_Ω over the Ω distribution (base 2).

        H_Ω = - Σ p(i) log2 p(i)

    Uses a safe mask to avoid log(0).
    """
    p = omega_to_distribution(omega)
    mask = p > 0
    if not np.any(mask):
        return 0.0
    p_safe = p[mask]
    return float(-np.sum(p_safe * np.log2(p_safe)))


def summarize_omega(text: str) -> Dict[str, Any]:
    """
    Convenience helper: given raw text, return a dict with

      - omega_vector: list[float] in WWWWHW order
      - omega_entropy: float
      - omega_dominant_dim: str (e.g. 'why')
      - omega_total_markers: int
    """
    omega = extract_omega_vector(text)
    H_omega = compute_omega_entropy(omega)

    total = int(omega.sum())
    if total > 0:
        dominant_idx = int(omega.argmax())
        dominant_dim = OMEGA_DIMENSIONS[dominant_idx]
    else:
        dominant_dim = "none"

    return {
        "omega_vector": omega.tolist(),
        "omega_entropy": H_omega,
        "omega_dominant_dim": dominant_dim,
        "omega_total_markers": total,
    }


# Simple CLI hook for quick manual testing:
if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        text = " ".join(sys.argv[1:])
    else:
        text = "Why do people migrate and how do policies affect it?"

    summary = summarize_omega(text)
    print("Text:", text)
    print("Summary:", summary)
