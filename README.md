
# Digit Recognizer (Kaggle) — CS4661 Team: Ryan Garcia, Braedon Edison, Chiemela Eziechile - Nwoke, Annie Lin, Kent Stark

**Goal**: Classify 28×28 grayscale handwritten digits (0–9). Submit a CSV with columns `ImageId,Label` to Kaggle.

## Repo layout
```
data/                # train.csv, test.csv, sample_submission.csv
notebooks/
  01_eda_baselines.ipynb   # existing baseline (KNN/LogReg)
  02_cnn_model.ipynb       # compact Keras CNN + submission
  03_error_analysis.ipynb  # confusion matrix & misclassifications
src/
  utils.py
  train_cnn.py
submissions/
  submission_baseline.csv  # baseline CSV (copy of sample for now)
  submission_cnn_v1.csv    # CNN CSV (placeholder until running 02)
models/                     # saved .keras models (created when training)
```
> Note: this package is designed to run both locally and on Kaggle Notebooks.

## Quickstart
1. Put `train.csv`, `test.csv`, and `sample_submission.csv` into `data/`.
2. Run **`notebooks/02_cnn_model.ipynb`** → trains a small CNN and writes `submissions/submission_cnn_v1.csv`.
3. Run **`notebooks/03_error_analysis.ipynb`** → see confusion matrix and misclassified examples.
4. Submit the CSV on Kaggle.


## Tips
- Always normalize pixels to `[0,1]`.
- Keep a tiny validation split (or CV) and track both CV and LB scores.
- Save 12–20 misclassified images and add 2–3 bullet insights to your report.
- Document submissions in the README (model, settings, LB score).



