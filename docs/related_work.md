# Related Work: Elicit Meta-Review (2025)

Elicit reviewed dozens of studies on RLHF and behavioral modification in LLMs.  
Their findings support the necessity of our measurement framework.

## What Elicit Confirms
- RLHF produces **consistent stylistic shifts** (5–7%) toward agreeable or hedged phrasing.  
- Alignment changes **token distributions** but does not provide mechanisms to measure drift.
- Safety research focuses on **surface features**, not latent affect transitions.

## What Was Missing in the Literature
- No method to quantify **paired emotional deltas** between “chosen” and “rejected” responses.
- No field-level entropy metrics for psychological markers.
- No longitudinal comparison across alignment eras.

## What Our System Provides
- FEATS: 9-category affect signature for each response.
- H-Drift: directional entropy shift between paired responses.
- Temporal comparison (2021 → 2023 → 2025) across WebGPT, HH-RLHF, and CA-1.

## Alignment With Our Findings
The Elicit report **does not contradict** any of our measurements.  
It demonstrates the absence of comparable tools and validates our contribution as the **first falsifiable measurement system for affect-drift**.
