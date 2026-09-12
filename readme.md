# CS5228 Project: HDB Rental Rate Prediction

Predict monthly rental rates for Singapore HDB flats using historical rental records and auxiliary data.

## Objective

This project frames HDB rental-rate estimation as a regression task. We investigate data quality, engineer relevant location, property, temporal, and economic features, compare predictive models, and interpret their strengths and limitations.

## Repository structure

- `data/`: local datasets only; excluded from version control
- `notebooks/`: exploratory data analysis and prototyping
- `src/`: reusable preprocessing, feature engineering, training, and evaluation code
- `configs/`: experiment and model configurations
- `outputs/`: figures, Kaggle submissions, and experiment metrics
- `report/`: progress-report and final-report materials

## Setup

```bash
pip install -r requirements.txt
```

Place the competition files in `data/`: `train.csv`, `test.csv`,
`example-submission.csv`, and the six supplied CSV files in `data/auxiliary/`.
Run these commands from the repository root:

```bash
python src/data_audit.py
python -m unittest discover -s tests
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

The audit regenerates cleaned CSVs and `outputs/audit.json` (input hashes,
runtime versions, core quality checks, auxiliary coverage and join validation).
The notebook also regenerates cleaned data before plotting, and saves seven
figures in `outputs/figures/`. Raw datasets are never modified.

All 150,000 train rows and 50,000 test rows are retained by default. Identical
records do not prove duplicate collection: unit/contract identifiers are absent.
The optional `clean_data(drop_exact_duplicates=True)` supports a later training
ablation. Test row order is always preserved for submission IDs.

`src/data_cleaning.py` performs deterministic cleaning of the core train/test files.
`src/data_audit.py` calls that cleaner, checks auxiliary data quality and join
cardinality, and writes reproducible evidence to `outputs/audit.json`. Neither
script fits encoders, scalers, imputers or predictive models. See the notebook
for EDA findings. No predictive model or Kaggle submission has been evaluated yet.
