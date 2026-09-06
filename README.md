# Heart Attack Prediction: Reproduction & Data Integrity Audit (Paper 3)

This repository contains the complete code, exported artifacts, and technical report for a High Distinction (HD) research task. It focuses on reproducing a state-of-the-art machine learning methodology for early heart attack prediction, auditing its evaluation integrity, and proposing a robust solution to address severe data leakage.

## Project Overview

* **Objective:** Reproduce the methodology of a published ML paper predicting cardiovascular diseases, evaluate its claims, and address its limitations.
* **Original Claim:** The paper achieved near-perfect accuracy (98.53%+) using complex ensemble models like Stacking.
* **Audit Finding:** A data contamination rate of 98.54% was discovered due to 723 duplicate patient records across train/test splits, inflating the baseline accuracy to an artificial 100%.
* **Proposed Solution:** Implemented a strict deduplication pipeline ($N=1025 \rightarrow 302$), re-evaluating the models on a genuinely independent test set.
* **Realistic Results:** On clean data, the Stacking Ensemble's accuracy dropped to 75.41%, while Logistic Regression proved to be the most robust baseline at 80.33%.

## Repository Structure

* `notebooks/`
  * `Paper3_Faithful_Reproduction_and_Deduplication_Study.ipynb`: The main Jupyter Notebook containing the deterministic setup, full data exploration, baseline reproduction, deduplication methodology, and empirical leakage proof.
* `reports/`
  * `research_report.pdf`: The comprehensive technical research report documenting the SOTA review, methodology, results, and critical analysis.
* `paper3_artifacts/`
  * Serialized `.joblib` models (Stacking Classifiers and Z-score Scalers) for both the full cohort and the deduplicated cohort.
  * Performance matrices exported as `.csv` files for reproducibility.

## How to Reproduce

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Deepmind00/Heart-Attack-Prediction-Reproduction-Audit.git](https://github.com/Deepmind00/Heart-Attack-Prediction-Reproduction-Audit.git)
   cd Heart-Attack-Prediction-Reproduction-Audit