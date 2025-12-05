# H-Drift Lab

**Goal**  
Quantify how large language models drift into humanistic / politeness-driven behavior over time, using public datasets and derived, text-free features.

This repository focuses on **H-Drift** – changes in politeness, hedging, empathy-coded language, and anthropomorphic stance – as an early signal of conversational instability and RLHF-induced bias.

## Datasets (external, not bundled)

This project uses only **public, well-established datasets**:

1. **Stanford Politeness Corpus (StackExchange)**  
   - Available via the ConvoKit `stack_politeness` corpus.  
   - Contains ~6.6k requests annotated for politeness.

2. **Anthropic HH-RLHF (Helpful & Harmless)**  
   - Available as `Anthropic/hh-rlhf` on Hugging Face or via Anthropic’s GitHub.  
   - ~160k human preference comparisons between “chosen” and “rejected” responses used for RLHF training.

> **Note:**  
> Raw data are *not* included in this repo.  
> Place them under `data/raw/stanford_politeness/` and `data/raw/hh_rlhf/` after obtaining them from the original sources.

## What this repo computes

For each dataset, we derive **text-free signals** per utterance or response:

- H-class densities (politeness, empathy, hedging, anthropomorphism)
- H-Drift Index over sample or conversation order
- Relationships between politeness markers and:
  - existing politeness annotations (Stanford corpus)
  - human preference labels (HH-RLHF)

Outputs are stored as `.parquet` tables in `data/processed/` and contain **no conversational text**, only numeric and categorical features.

## Structure

- `src/h_drift/lexicon.py` – definition of H-class word lists (H1–H5).
- `src/h_drift/features_politeness.py` – feature extraction for the Stanford Politeness Corpus.
- `src/h_drift/features_hh_rlhf.py` – feature extraction for HH-RLHF comparisons.
- `src/h_drift/metrics.py` – definitions of H-Drift Index and related metrics.
- `notebooks/` – exploratory analysis notebooks.

## License

- Code in this repository is released under the MIT License (see `LICENSE`).
- External datasets are governed by their original licenses; see upstream dataset documentation.
