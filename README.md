# TATA ELEXI TELEPORT: SELF-HEALING MACHINES
### AI-Powered Predictive Maintenance Intelligence Platform for Connected Fleets

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 25 Passed](https://img.shields.io/badge/tests-25%20passed-brightgreen.svg)]()
[![Dataset: NASA C-MAPSS](https://img.shields.io/badge/dataset-NASA%20C--MAPSS-orange.svg)](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6)

---

## 1. Project Overview
**Self-Healing Machines** is an industrial-grade, end-to-end Predictive Maintenance (PdM) intelligence platform designed for commercial aircraft turbofan fleets. Built upon the official **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset, the system processes high-dimensional physical sensor streams to detect uncharacteristic operational anomalies, forecast Remaining Useful Life (RUL) with high precision, apply deterministic safety triage rules, and autonomously dispatch persistent maintenance work orders to an operational registry.

The entire solution runs efficiently on standard hardware without requiring expensive cloud services or GPUs, providing a reliable, transparent, competition-ready codebase with zero black-box generative AI hallucinations in physical safety loops.

---

## 2. Problem Statement
Commercial airline operators lose over \$30 billion annually due to unscheduled aircraft maintenance, ground delays, flight cancellations, and catastrophic component wear. Traditional maintenance regimes suffer from two fundamental inefficiencies:
1. **Time-Based Maintenance (Reactive/Preventative):** Components are serviced at rigid cycle intervals, discarding healthy parts prematurely and driving up unnecessary depot costs.
2. **Run-to-Failure (Corrective):** Catastrophic component failure risks in-flight emergencies, engine loss, and human safety hazards.

**The Solution:** Continuous predictive health monitoring that tracks cumulative degradation and flags instantaneous sensor anomalies, enabling precision-scheduled depot servicing before mechanical failure occurs.

---

## 3. Main Objectives
- **Data Integrity & Validation:** Parse raw, whitespace-delimited telemetry; validate sensor columns, cycle monotonicity, and non-finite values.
- **Strict Leakage Prevention:** Ensure zero future data leakage and zero cross-engine contamination in rolling statistics and feature transformations.
- **Dual-Track Machine Learning:**
  - *Track A:* Unsupervised **Isolation Forest** for instantaneous anomaly detection.
  - *Track B:* Supervised **HistGradientBoostingRegressor** with piecewise linear RUL capping (125 cycles).
- **Benchmark Evaluation:** Evaluate on held-out test engines using standard metrics (MAE, RMSE, $R^2$) and the official NASA asymmetric prognostic scoring function.
- **Deterministic Triage Agent:** Implement transparent, auditable Python decision rules with priority escalation and defensive fail-safes.
- **Persistent Maintenance Work Orders:** Automate idempotent, duplicate-suppressed work-order creation in SQLite with status transitions (`OPEN` $\to$ `IN_PROGRESS` $\to$ `COMPLETED`).
- **MLOps & Automated Retraining:** Track parameters and artifacts in MLflow; enforce candidate-to-champion promotion criteria (requiring $\ge 2\%$ RMSE improvement).
- **Interactive Industrial Dashboard:** Provide an 8-page Streamlit analytics interface with interactive Plotly visualizations.

---

## 4. System Architecture
```
+-----------------------------------------------------------------------------------+
|                            DATA INGESTION & VALIDATION                            |
|  NASA C-MAPSS FD001 (21 Sensors, 3 Settings) -> Monotonicity & Schema Validation  |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                      DATA PREPROCESSING & FEATURE PIPELINE                        |
|  Variance Filter -> Grouped Rolling Stats (5, 10, 20) -> Cycle Differences/Trends |
+-----------------------------------------------------------------------------------+
                                          |
                   +----------------------+----------------------+
                   |                                             |
                   v                                             v
+-------------------------------------+       +-------------------------------------+
|        TRACK A: ANOMALY SCORER      |       |        TRACK B: RUL REGRESSOR       |
|    Isolation Forest (Unsupervised)   |       |    HistGradientBoosting (Capped 125)|
|  Instantaneous Normal / Warning /   |       |    Remaining Useful Life Point      |
|  Anomalous Classification           |       |    Estimates (Cycles to Failure)    |
+-------------------------------------+       +-------------------------------------+
                   |                                             |
                   +----------------------+----------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                            DETERMINISTIC TRIAGE AGENT                             |
|  Deterministic Rules: RUL <= 15 (CRITICAL), <= 30 (HIGH), <= 60 (MED), > 60 (LOW) |
|  Anomaly Escalation Rule: Severe anomaly elevates priority tier by 1 level        |
|  Defensive Guard: Missing / NaN telemetry automatically triggers CRITICAL ticket  |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                     AUTOMATED PERSISTENT WORK ORDER REGISTRY                      |
|  SQLite DB (WAL Mode) -> Duplicate Prevention -> OPEN -> IN_PROGRESS -> COMPLETED |
+-----------------------------------------------------------------------------------+
                                          |
                   +----------------------+----------------------+
                   |                                             |
                   v                                             v
+-------------------------------------+       +-------------------------------------+
|         STREAMLIT DASHBOARD         |       |            MLOPS & MLFLOW           |
|  8-Page Command Analytics & Plots   |       |  Tracking, Data Drift (PSI), Retrain |
+-------------------------------------+       +-------------------------------------+
```

---

## 5. Technologies Used
- **Language:** Python 3.10+ (Tested on Python 3.14.7)
- **Data Engineering:** `pandas`, `numpy`, `scipy`
- **Machine Learning:** `scikit-learn` (`HistGradientBoostingRegressor`, `IsolationForest`, `Ridge`, `StandardScaler`)
- **Experiment Tracking:** `mlflow`
- **Database:** `sqlite3` (with Write-Ahead Logging `WAL` mode)
- **Interactive Dashboard:** `streamlit`, `plotly`
- **Unit & Integration Testing:** `pytest`
- **Configuration & Artifacts:** `PyYAML`, `joblib`

---

## 6. Official NASA C-MAPSS Benchmark Results

The following metrics represent actual measured performance on the official held-out **NASA C-MAPSS FD001 test set** (100 unseen engines evaluated at their final operating cycle against true RUL):

| Prognostic Metric | Measured Value | Unit | Definition |
| :--- | :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | **13.62** | Cycles | Average absolute prediction error |
| **Root Mean Squared Error (RMSE)** | **18.92** | Cycles | Penalizes large errors quadratically |
| **Coefficient of Determination ($R^2$)** | **0.7928** | Dimensionless | Variance explained by model |
| **NASA PHM'08 Penalty Score** | **1099.86** | Points | Asymmetric exponential penalty function |

### Stratified Lifecycle Performance
| Operating Regime | Engine Count | MAE (cycles) | RMSE (cycles) |
| :--- | :--- | :--- | :--- |
| **Critical Stage (RUL $\le 30$)** | 25 | **7.34** | **12.48** |
| **Mid-Life Stage ($30 < \text{RUL} \le 80$)** | 20 | 18.29 | 25.22 |
| **Healthy Stage ($\text{RUL} > 80$)** | 55 | 14.78 | 18.67 |

---

## 7. Execution Commands

```bash
# 1. Run full automated test suite (25 tests)
pytest tests/ -v

# 2. Run model training pipeline on C-MAPSS FD001
python scripts/train.py

# 3. Evaluate official test set benchmark and print report
python scripts/evaluate.py

# 4. Run inference pipeline and dispatch work orders to SQLite
python scripts/run_pipeline.py

# 5. Execute 8-step live competition demo script
python scripts/demo.py

# 6. Launch interactive Streamlit web dashboard
streamlit run app.py
```
