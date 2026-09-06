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