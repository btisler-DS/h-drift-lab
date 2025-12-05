markdown
# H-Drift Lab — User Guide (v1.0)
*A minimal, functional guide for operating the Humanistic Drift Diagnostic Lab.*

---

## 1. Purpose

The H-Drift Lab provides tools to measure **humanistic drift signals** in text:

- Emotion words  
- Relational language  
- Hedging / uncertainty  
- Anthropomorphism  
- Politeness softeners  

These signals accumulate over time and can reflect:

- LLM drift  
- User affect  
- Interaction dynamics  

The lab offers:

- A dataset in simple CSV form  
- A feature extractor (`features_politeness.py`)  
- A drift metric script (`metrics.py`)  

This forms a reproducible baseline for future comparative studies.

---

## 2. Repository Structure

```text
h-drift-lab/
│
├── data/
│   ├── raw/
│   │   └── stanford_politeness/
│   └── processed/
│
├── src/
│   ├── h_drift/
│   │   ├── __init__.py
│   │   ├── lexicon.py
│   │   ├── features_politeness.py
│   │   └── metrics.py
│   └── utils/
│       └── __init__.py
│
├── notebooks/
├── experiments/
│
├── README.md
└── USER_GUIDE.md
````

---

## 3. Installation

Requires Python 3.9+.

Install minimal dependencies:

```bash
pip install pandas pyarrow
```

(You already have these installed if the extractor and metrics scripts run.)

---

## 4. Prepare the Dataset

The extractor expects a CSV at:

```text
data/raw/stanford_politeness/stanford_politeness.csv
```

Minimum requirement:

* A text column (any of): `text`, `Request`, `request`, `sentence`, `utterance`.

Optional (recommended):

* A label column (any of): `label`, `is_polite`, `politeness`, `Binary`, `y`.

Example minimal CSV:

```csv
text,label
Could you please review my code when you have a moment?,polite
Review my code.,impolite
I really appreciate your help with this.,polite
You need to fix this now.,impolite
Thank you for taking the time to look at this.,polite
Why didn't you do what I asked?,impolite
```

Place this file at:

```text
data/raw/stanford_politeness/stanford_politeness.csv
```

---

## 5. Run the Feature Extractor

From the repo root:

```bash
python .\src\h_drift\features_politeness.py
```

What it does:

* Loads the CSV

* Detects the text and label columns

* Computes counts for each utterance:

  * `h1_emotion`
  * `h2_relational`
  * `h3_hedging`
  * `h4_anthro`
  * `h5_softeners`

* Writes the output to:

```text
data/processed/stanford_politeness_h_drift.parquet
```

Each row in the Parquet file corresponds to one input row and includes:

* `row_id`
* H-class counts
* `h_total` (sum of all H-classes)
* `politeness_label` (if present in the CSV)

---

## 6. Run the Metrics Module

From the repo root:

```bash
python .\src\h_drift\metrics.py
```

This will:

* Load `data/processed/stanford_politeness_h_drift.parquet`
* Add `H_drift_index` (currently defined as `h_total`)
* Print:

  * Overall summary statistics for all H-class counts + `H_drift_index`
  * Grouped statistics by `politeness_label` (if present)

This provides a first check that H-drift is meaningfully different between classes (e.g., polite vs impolite).

---

## 7. Interpreting the Signals

Per row:

* Higher **H_drift_index** = more humanistic markers in that utterance.
* Decomposition by H-class:

  * `h1_emotion` – emotional language
  * `h2_relational` – relational / affiliative language
  * `h3_hedging` – uncertainty / hedging
  * `h4_anthro` – anthropomorphic self-reports
  * `h5_softeners` – politeness softeners

Typical uses:

* Compare polite vs impolite language
* Study how humanistic markers accumulate over a conversation
* Use as a feature in downstream drift diagnostics

---

## 8. Troubleshooting

**CSV not found**

* Ensure the file exists at:
  `data/raw/stanford_politeness/stanford_politeness.csv`

**No text column detected**

* Make sure your CSV has one of: `text`, `Request`, `request`, `sentence`, `utterance`.

**Parquet error**

* Install `pyarrow`:

  ```bash
  pip install pyarrow
  ```

**All counts zero**

* Text may be too neutral or very short.
* Check `lexicon.py` to see which phrases are being counted.

---

## 9. Roadmap

Planned extensions:

* Conversation-level drift curves (time series).
* Multi-model comparison of drift signatures.
* Integration with interrogative geometry (WWWWHW cube).
* Early-warning signals for pathological drift patterns.

---

## 10. License

Code is released under the **MIT License** (see `LICENSE.txt`).
Datasets remain under their original licenses.

