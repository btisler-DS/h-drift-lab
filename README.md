# H-Drift Lab

**Goal**
Quantify how large language models drift into humanistic / politeness-driven behavior over time, using public datasets and derived, text-free features.

This repository focuses on **H-Drift** -- changes in politeness, hedging, empathy-coded language, and anthropomorphic stance -- as an early signal of conversational instability and RLHF-induced bias.

## Datasets (external, not bundled)

This project uses only **public, well-established datasets**:

1. **Stanford Politeness Corpus (StackExchange)**
   - Available via the ConvoKit `stack_politeness` corpus.
   - Contains ~6.6k requests annotated for politeness.

2. **Anthropic HH-RLHF (Helpful & Harmless)**
   - Available as `Anthropic/hh-rlhf` on Hugging Face.
   - ~160k human preference comparisons between "chosen" and "rejected" responses used for RLHF training.

3. **OpenAI WebGPT Comparisons**
   - Available as `openai/webgpt_comparisons` on Hugging Face.
   - Human preference pairs over web-assisted GPT answers.

4. **OpenAI Collective Alignment-1 (CA-1)**
   - Available as `openai/collective-alignment-1` on Hugging Face.
   - Multi-response comparisons (A/B/C/D) with annotator rationales, importance ratings, subjectivity labels, and acceptability judgments.

5. **Anthropic Values in the Wild**
   - Available as `Anthropic/values-in-the-wild` on Hugging Face.
   - Value-frequency data from real-world conversations.

> **Note:**
> Raw data are *not* included in this repo.
> Run the dataset loaders (`src/h_drift/load_*.py`) to fetch them, or place files manually under `data/raw/`.

## What this repo computes

For each dataset, we derive **text-free signals** per utterance or response:

- H-class densities (politeness, empathy, hedging, anthropomorphism)
- FEATS dimensions (Feelings, Expressions, Actions, Thoughts, Sensations)
- Omega (WWWWHW) interrogative classification (who, what, when, where, why, how)
- H-Drift Index over sample or conversation order
- Within-pair deltas (chosen vs rejected) for H-class and FEATS features
- Predictive entropy drift (interrogative entropy vs affect drift)
- Relationships between politeness markers and:
  - existing politeness annotations (Stanford corpus)
  - human preference labels (HH-RLHF, WebGPT, CA-1)

Outputs are stored as `.parquet` tables in `data/processed/` and contain **no conversational text**, only numeric and categorical features.

## Structure

### Lexicons

- `src/h_drift/lexicon.py` -- H-class word lists (H1--H5: emotion, relational, hedging, anthropomorphism, softeners).
- `src/h_drift/feats_lexicon.py` -- FEATS lexical categories (Feelings, Expressions, Actions, Thoughts, Sensations).
- `src/h_drift/omega.py` -- Rule-based Omega classifier for WWWWHW interrogative geometry.

### Dataset loaders

- `src/h_drift/load_ca1.py` -- Download and cache OpenAI Collective Alignment-1 comparisons.
- `src/h_drift/load_webgpt.py` -- Download and cache OpenAI WebGPT comparisons.
- `src/h_drift/load_values_in_the_wild.py` -- Download and cache Anthropic Values in the Wild.

### Feature extraction

- `src/h_drift/features_politeness.py` -- H-class features for the Stanford Politeness Corpus.
- `src/h_drift/features_hh_rlhf.py` -- H-class features for HH-RLHF (with prompt extraction from Human turns).
- `src/h_drift/features_hh_rlhf_feats.py` -- FEATS augmentation for HH-RLHF.
- `src/h_drift/features_ca1_pairs.py` -- Extract CA-1 responses into atomic (prompt, response, label) rows.
- `src/h_drift/features_ca1_feats.py` -- FEATS augmentation for CA-1 responses.
- `src/h_drift/features_webgpt_pairs.py` -- Extract WebGPT comparisons into paired rows with H-class features.
- `src/h_drift/features_webgpt_feats.py` -- FEATS augmentation for WebGPT pairs.

### Metrics and analysis

- `src/h_drift/metrics.py` -- H-Drift Index and related metrics.
- `src/h_drift/metrics_hh_rlhf.py` -- HH-RLHF summary metrics.
- `src/h_drift/metrics_hh_rlhf_delta.py` -- Within-pair H-drift deltas (chosen vs rejected).
- `src/h_drift/metrics_hh_rlhf_delta_features.py` -- Per-feature within-pair deltas (H1--H5 individually).
- `src/h_drift/metrics_hh_rlhf_omega.py` -- Attach Omega (WWWWHW) features to HH-RLHF using prompt text.
- `src/h_drift/metrics_hh_rlhf_omegaOG.py` -- Omega features with basic summary stats (dominant dimension, WHY vs HOW).
- `src/h_drift/metrics_webgpt_feats_delta.py` -- Within-pair FEATS deltas for WebGPT.
- `src/h_drift/analysis_omega_drift.py` -- Compare H-drift patterns across interrogative dimensions.
- `src/h_drift/predictive_entropy_drift.py` -- OLS regression of affect drift on interrogative entropy.

### Data outputs

- `data/processed/entropy_drift_summary.tsv` -- Predictive entropy drift summary table.

### Documentation

- `docs/Geometric_Instrument_v2_CORRECTED_FINAL.md` -- Geometric Instrument v2 manuscript (markdown).
- `docs/Version_2_manuscript (2).pdf` -- Geometric Instrument v2 manuscript (PDF).
- `docs/related_work.md` -- Elicit meta-analysis of RLHF literature.
- `docs/anthropic_method.md` -- Anthropic Interviewer Protocol cross-validation.
- `USER_GUIDE.md` -- Setup and usage instructions.
- `notebooks/` -- Exploratory analysis notebooks.

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
- Reported stylistic shifts post-RLHF average **5--7%**, but no field-level metrics exist.
- Our FEATS/H-drift system fills this methodological gap with the **first falsifiable measurement**.

See: `docs/related_work.md`

### Anthropic Interviewer Protocol (2025)
We analyzed Anthropic's "AI Interviewer" research protocol to cross-validate our drift categories.
No contradictions were found.
Their emphasis on *emotional drift, trust, reliability, and conversational appeasement* directly aligns with our quantitative FEATS dimension system.

See: `docs/anthropic_method.md`

## Theoretical Foundation

This repository applies the measurement framework introduced in:

> Tisler, B. (2025). *A Geometric Instrument for Measuring Interrogative
> Entropy in Language Systems* (Version v1). Zenodo.
> https://doi.org/10.5281/zenodo.17811309

That paper establishes Cube Geometry and Interrogative Entropy (Hi) as
deterministic measurements independent of language model behavior. This
repository extends that framework to measure how RLHF training introduces
humanistic drift (H-Drift) in AI responses.
