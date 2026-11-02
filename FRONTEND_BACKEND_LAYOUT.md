# Frontend and Backend Layout Map

This document classifies existing workspace files without moving or duplicating them.
Application imports and paths intentionally remain unchanged.

## Streamlit Frontend

The existing Streamlit presentation and dashboard code is located at:

- `app.py`
- `public/index.html` — existing static/Netlify portal
- `frontend/components/` — reserved reusable frontend components
- `frontend/pages/` — reserved Streamlit page modules
- `frontend/assets/` — reserved images, icons, and logos
- `frontend/styles/` — reserved frontend styles

No frontend page modules are currently present separately because the existing dashboard
navigation and UI are implemented directly in `app.py`.

## Backend and Machine Learning

The existing backend implementation remains under `src/` until imports and paths are
authorized for relocation:

- `src/data_loader.py` — dataset ingestion
- `src/data_validation.py` — input validation
- `src/preprocessing.py` — preprocessing
- `src/feature_engineering.py` — feature generation
- `src/rul_labels.py` — RUL labels
- `src/anomaly_detection.py` — anomaly detection
- `src/rul_model.py` — RUL regression
- `src/pipeline.py` — training and inference orchestration
- `src/evaluation.py` — model evaluation
- `src/triage_agent.py` — maintenance recommendation logic
- `src/work_order_service.py` — maintenance service
- `src/database.py` — SQLite database operations
- `src/retraining.py` — retraining workflow
- `src/monitoring.py` — monitoring and drift support

## Supporting Files

- `config/config.yaml` — configuration
- `data/raw/` — NASA C-MAPSS source datasets
- `data/processed/` — processed dataset location
- `data/external/` — external datasets or inputs
- `database/work_orders.db` — SQLite runtime database
- `artifacts/models/` — trained and serialized models
- `artifacts/reports/` — evaluation and prediction reports
- `artifacts/figures/` — reserved generated figures
- `artifacts/mlflow/` — reserved organized MLflow artifacts
- `mlruns/` — existing MLflow experiment tracking directory
- `mlflow.db` — existing MLflow database
- `scripts/` — training, evaluation, inference, and demo commands
- `tests/` — existing automated tests
- `docs/` — technical and project documentation
- `notebooks/` — reserved notebook location
- `logs/` — reserved runtime logs
- `.streamlit/` — Streamlit configuration
- `.venv/` — local Python environment
- `.tools/` — tool-managed executables

## Current Physical State

The `frontend/` and `backend/` directories are reserved package scaffolds. Existing files
remain in their original locations so imports, algorithms, datasets, models, database paths,
MLflow tracking, and Streamlit navigation continue to work unchanged.
