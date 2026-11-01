# Model Card: Self-Healing Machines Prognostic Models

## 1. Model Details
- **Model Name:** Turbofan RUL Estimator & Anomaly Scorer
- **Model Architectures:**
  - **Prognostic Regressor:** `HistGradientBoostingRegressor` (scikit-learn)
  - **Anomaly Scorer:** `IsolationForest` (scikit-learn)
  - **Baseline Regressor:** `Ridge Regression`
- **Version:** `1.0.0`
- **Training Date:** 2026-11-01
- **Developer:** Self-Healing Machines Engineering Team
- **License:** Open Source (Apache 2.0 / MIT compatible)

---

## 2. Intended Use
- **Primary Objective:** Estimate the Remaining Useful Life (RUL in operational cycles) of turbofan jet engines and flag sensor anomalies prior to functional failure.
- **Target Users:** Fleet reliability engineers, depot maintenance dispatchers, and aerospace systems technicians.
- **Out-of-Scope Uses:** Autonomous real-time flight shutdown authority, safety-critical flight control systems, or uninspected certified aircraft operations.

---

## 3. Training Data
- **Dataset:** NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset (Subset `FD001`).
- **Engines:** 100 run-to-failure engines (16,561 training snapshots across 80 training units, 4,070 validation snapshots across 20 validation units).
- **Sensors:** 21 physical sensor measurements (pressures, temperatures, fan speeds, bypass ratios) + 3 operational settings.
- **Preprocessing:** Constant sensors dropped (`sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`, `setting_3`). StandardScaler fitted on training partition.
- **RUL Capping:** Piecewise linear degradation target capped at 125 cycles.

---

## 4. Evaluation Metrics (Official FD001 Benchmark)

### Test Evaluation (Held-Out Test Engines Evaluated at Last Observed Cycle)
- **Mean Absolute Error (MAE):** `13.62` cycles
- **Root Mean Squared Error (RMSE):** `18.92` cycles
- **Coefficient of Determination ($R^2$):** `0.7928`
- **NASA Asymmetric PHM'08 Penalty Score:** `1099.86` points
  $$\text{NASA Score} = \sum_{i} \left( e^{-d_i / 13} - 1 \text{ for } d_i < 0 \right) + \left( e^{d_i / 10} - 1 \text{ for } d_i \ge 0 \right)$$
  where $d_i = \hat{y}_i - y_i$.

### Stratified Error Analysis
| Lifecycle Regime | Engine Count | MAE (cycles) | RMSE (cycles) |
| :--- | :--- | :--- | :--- |
| **Critical Stage (RUL $\le 30$)** | 25 | **7.34** | **12.48** |
| **Mid-Life Stage ($30 < \text{RUL} \le 80$)** | 20 | 18.29 | 25.22 |
| **Healthy Stage ($\text{RUL} > 80$)** | 55 | 14.78 | 18.67 |

> **Diagnostic Insight:** The model achieves superior precision (MAE 7.34 cycles) in the critical end-of-life stage (RUL $\le 30$), which is the most critical window for physical maintenance intervention.

---

## 5. Ethical & Operational Safeguards
- **Explainability:** Feature importance diagnostic generated and stored with model artifacts.
- **Fail-Safe Mechanism:** In the event of missing input channels or model inference failure, the triage agent automatically flags the asset as `CRITICAL` with reason *"Inference Exception: Telemetry Failure"*—it never defaults to a healthy assumption.
- **Non-Autonomous:** The platform issues advisory work orders; physical inspection by certified technicians is mandatory before return-to-service.
