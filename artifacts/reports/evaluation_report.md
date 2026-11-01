# NASA C-MAPSS FD001 Official Test Benchmark Report
**Context:** Evaluation on held-out test engines at final observed operating cycle  
**Total Evaluated Units:** 100 engines  

## Key Prognostic Metrics
| Metric | Value | Unit | Definition |
| :--- | :--- | :--- | :--- |
| **MAE** | `13.62` | Cycles | Mean Absolute Error |
| **RMSE** | `18.92` | Cycles | Root Mean Squared Error |
| **R^2 Score** | `0.7928` | Dimensionless | Coefficient of Determination |
| **NASA Score** | `1099.86` | Penalty Points | Asymmetric PHM'08 Penalty Function |

## Residual Characteristics
- **Mean Residual (Bias):** `3.20` cycles
- **Residual Std Dev:** `18.64` cycles
- **Over-predictions (Late):** `61` engines (Penalized heavily)
- **Under-predictions (Early):** `39` engines (Conservative)

## Stratified Lifecycle Performance
| Operating Regime | Engine Count | MAE (cycles) | RMSE (cycles) |
| :--- | :--- | :--- | :--- |
| **Critical Stage (RUL <= 30)** | 25 | `7.34` | `12.48` |
| **Mid-Life Stage (30 < RUL <= 80)** | 20 | `18.29` | `25.22` |
| **Healthy Stage (RUL > 80)** | 55 | `14.78` | `18.67` |
