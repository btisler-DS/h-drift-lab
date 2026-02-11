# H-Drift Lab

**Goal**  
Quantify how large language models drift into humanistic / politeness-driven behavior over time, using public datasets and derived, text-free features.

This repository focuses on **H-Drift** â€“ changes in politeness, hedging, empathy-coded language, and anthropomorphic stance â€“ as an early signal of conversational instability and RLHF-induced bias.

## Datasets (external, not bundled)

This project uses only **public, well-established datasets**:

1. **Stanford Politeness Corpus (StackExchange)**  
   - Available via the ConvoKit `stack_politeness` corpus.  
   - Contains ~6.6k requests annotated for politeness.

2. **Anthropic HH-RLHF (Helpful & Harmless)**  
   - Available as `Anthropic/hh-rlhf` on Hugging Face or via Anthropicâ€™s GitHub.  
   - ~160k human preference comparisons between â€œchosenâ€ and â€œrejectedâ€ responses used for RLHF training.

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

- `src/h_drift/lexicon.py` â€“ definition of H-class word lists (H1â€“H5).
- `src/h_drift/features_politeness.py` â€“ feature extraction for the Stanford Politeness Corpus.
- `src/h_drift/features_hh_rlhf.py` â€“ feature extraction for HH-RLHF comparisons.
- `src/h_drift/metrics.py` â€“ definitions of H-Drift Index and related metrics.
- `notebooks/` â€“ exploratory analysis notebooks.

## License

- Code in this repository is released under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) License (see `LICENSE.txt`).
- External datasets are governed by their original licenses; see upstream dataset documentation.

## New Dataset Integrations (Dec 2025 Update)

### Collective Alignment-1 (OpenAI, 2025)
We now include **CA-1**, a large-scale alignment dataset containing:
- multi-response comparisons (A/B/C/D)
- human annotator rationales
- importance ratings
- subjectivity labels
- acceptability judgments

All responses have been extracted into atomic entries and processed through **FEATS v1.0** (affect markers), producing:


This allows cross-dataset comparison of **affect-drift**, **politeness inflation**, and **epistemic closure** across alignment eras (2021 → 2023 → 2025).

### Elicit Meta-Analysis Integration
We added a summary of the Elicit survey of the RLHF literature.  
Key takeaways:
- No prior work measures **paired affect-drift** between chosen and rejected outputs.
- Reported stylistic shifts post-RLHF average **5–7%**, but no field-level metrics exist.
- Our FEATS/H-drift system fills this methodological gap with the **first falsifiable measurement**.

See: `docs/related_work.md`

### Anthropic Interviewer Protocol (2025)
We analyzed Anthropic’s “AI Interviewer” research protocol to cross-validate our drift categories.
No contradictions were found.  
Their emphasis on *emotional drift, trust, reliability, and conversational appeasement* directly aligns with our quantitative FEATS dimension system.

See: `docs/anthropic_method.md`

## Documentation

For detailed instructions on how to set up and use the H-Drift Lab, see:

- [USER_GUIDE.md](USER_GUIDE.md)
## Theoretical Foundation

This repository applies the measurement framework introduced in:

> Tisler, B. (2025). *A Geometric Instrument for Measuring Interrogative 
> Entropy in Language Systems* (Version v1). Zenodo. 
> https://doi.org/10.5281/zenodo.17811309

That paper establishes Cube Geometry and Interrogative Entropy (Hᵢ) as 
deterministic measurements independent of language model behavior. This 
repository extends that framework to measure how RLHF training introduces 
humanistic drift (H-Drift) in AI responses.
