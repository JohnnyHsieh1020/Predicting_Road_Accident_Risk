# Kaggle – Predicting Road Accident Risk

## Overview
This repository contains my experiments for the Kaggle **Predicting Road Accident Risk** competition. The goal is to forecast the `accident_risk` for every record in the provided road-condition dataset so that the submission can be evaluated with the competition metric.

## Leaderboard performance
- Current best submission ranks **919 / 4,083** with a **Root Mean Squared Error of 0.05580**.

## Dataset
- Source: Kaggle competition [Predicting Road Accident Risk](https://www.kaggle.com/competitions/playground-series-s5e10/).
- Shape: 517,754 training rows with 14 columns describing road geometry, control signals, weather, and temporal context plus the `accident_risk` target.
- Target: `accident_risk`.
- Files: `train.csv`, `test.csv`.

## Repository Guide
- `baseline_process.ipynb`: Notebook that loads the raw data, performs feature engineering, model training and hyperparameter tuning via Optuna.
- `submission_baseline_process.csv`: Generated Kaggle submission files from baseline workflow.
- `road_accident_autogluon.ipynb`: AutoGluon training pipeline used to iterate on tabular models and generate predictions.
- `submission_autogluon.csv`: Generated Kaggle submission files from AutoGluon workflow.

## Feature engineering highlights
- Logical flags: `is_highway`, `is_dark`, `is_peak_hour`, and `is_bad_weather` capture road type, lighting, demand, and adverse weather in binary form.
- Ratio feature: `lane_density = num_reported_accidents / (num_lanes + 1e-5)` measures historical accidents per lane while avoiding divide-by-zero.
- Binned feature: `speed_zone` segments `speed_limit` into `low/medium/high/extreme` bands, letting tree models learn step changes instead of assuming linear behavior.
- The engineered columns feed scikit-learn preprocessors and downstream models (LightGBM, AutoGluon). Feature importance analyses (RF, LightGBM + SHAP) typically promote `curvature`, `speed_zone_high`, `lighting_night`, `num_reported_accidents`, `speed_limit`, and the weather flags, so the final training set focuses on 14 high-impact features.

## Baseline process (`baseline_process.ipynb`)
- Feature selection: random forest and LightGBM + SHAP rankings are merged to keep 14 high-importance columns before training.
- Model training: LightGBM regressor tuned via Optuna (20 trials) using 5-fold cross-validation; best run delivers **RMSE 0.05596**.
- Outputs: CSV exports such as `submission_baseline_process.csv` for submission formatting.

## AutoML process (`road_accident_autogluon.ipynb`)
- Model training: AutoGluon Tabular trains an ensemble of LightGBM, XGBoost, CATBoost and neural nets with automated validation, producing **RMSE 0.05580** on the Kaggle leaderboard.
- Outputs: CSV exports such as `submission_autogluon.csv` for submission formatting.
