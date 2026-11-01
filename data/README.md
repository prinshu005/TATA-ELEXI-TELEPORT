# Dataset Directory: NASA C-MAPSS Turbofan Engine Degradation

This directory holds the raw and processed data for the **Self-Healing Machines** predictive maintenance platform.

## Raw Data (`data/raw/`)
The raw files originate from the official NASA C-MAPSS dataset (A. Saxena et al., PHM 2008).
Files include:
- `train_FD001.txt` to `train_FD004.txt`: Run-to-failure multivariate time series for training engines.
- `test_FD001.txt` to `test_FD004.txt`: Truncated multivariate time series for test engines ending prior to system failure.
- `RUL_FD001.txt` to `RUL_FD004.txt`: Ground truth remaining useful life (in cycles) for the test engines at their last observed cycle.

### Schema
Each row consists of 26 space-delimited numerical values:
1. `engine_id` (unit number)
2. `cycle` (operational cycle)
3. `setting_1` (operational setting 1)
4. `setting_2` (operational setting 2)
5. `setting_3` (operational setting 3)
6-26. `sensor_1` to `sensor_21` (21 physical sensor measurements)

## Processed Data (`data/processed/`)
Contains feature-engineered datasets, validation splits, and preprocessed matrices used during model training and inference.
Processed artifacts are generated deterministically by the preprocessing and feature-engineering pipelines.
