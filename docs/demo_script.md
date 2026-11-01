# Live Competition Demonstration Script

**Project Title:** Self-Healing Machines: AI-Powered Predictive Maintenance for a Connected Fleet  
**Target Audience:** Technical Judges, Industry Evaluators, and Engineering Panel  
**Duration:** 7 - 10 Minutes  

---

## Preparation Checklist
1. Ensure virtual environment is active: `.\.venv\Scripts\activate` (or invoke via `.\.venv\Scripts\python.exe`).
2. Run database initialization and model training: `python scripts/train.py`.
3. Launch Streamlit Dashboard: `streamlit run app.py` (or run headless).
4. Have terminal window open for the CLI demonstration: `python scripts/demo.py`.

---

## Step-by-Step Presentation & Speaker Notes

### 1. Introduction & Problem Statement (1 Minute)
- **Speaker:**  
  *"Good morning judges. Today, our team presents **Self-Healing Machines**, an end-to-end, industrial-grade predictive maintenance platform. Unscheduled aircraft maintenance accounts for over \$30 billion in annual airline operational disruptions and safety hazards. Traditional scheduled maintenance either over-services healthy engines or risks missing rapid in-flight degradation. Our platform continuously analyzes 21 sensor channels from connected turbofan engines to detect anomalies, accurately forecast Remaining Useful Life, and autonomously dispatch auditable maintenance work orders."*

### 2. Live Automated Pipeline Demo (2 Minutes)
- **Action:** Open terminal and run:
  ```powershell
  python scripts/demo.py
  ```
- **Speaker:**  
  *"We are executing our automated demonstration script. In Step 1 and 2, our data ingestion and validation module loads the official NASA C-MAPSS FD001 dataset—100 run-to-failure engines—and verifies cycle monotonicity, schema completeness, and absence of non-finite measurements. Notice that our unit test suite verifies 100% engine-boundary isolation, preventing cross-engine data leakage."*

### 3. Prognostic Accuracy & Asymmetric Scoring (2 Minutes)
- **Speaker:**  
  *"Look at Step 4 of the demo output. On the official held-out NASA test benchmark of 100 unseen engines evaluated at their final operating cycle:*
  - *Mean Absolute Error (MAE) is **13.62 cycles**.*
  - *Root Mean Squared Error (RMSE) is **18.92 cycles**.*
  - *Coefficient of Determination ($R^2$) is **0.7928**.*
  - *NASA PHM'08 Asymmetric Penalty Score is **1099.86 points**.*
  *Most importantly, our stratified error analysis shows that when engines are in the **critical failure zone ($RUL \le 30$ cycles)**, our model's MAE drops to **7.34 cycles**. That means when the engine is within days of critical wear, our prediction error is approximately 7 flight cycles."*

### 4. Deterministic Triage & Work Orders (2 Minutes)
- **Speaker:**  
  *"In Step 5 and 6, our dual-track inference executes:*
  - *Track A: Isolation Forest detects transient out-of-distribution sensor spikes.*
  - *Track B: HistGradientBoosting predicts cumulative degradation.*
  *Notice Engine 20: operating at cycle 184, predicted RUL is 10.3 cycles. Our deterministic triage agent flags this as **CRITICAL**. Because a warning anomaly score (0.67) was also observed, the system records the decision rationale and autonomously writes ticket `WO-ENG020-C184-CRIT` to our persistent SQLite database.*
  *In Step 7, we demonstrate idempotency: subsequent telemetry readings never create duplicate tickets for the same active failure. We transition ticket status from `OPEN` to `IN_PROGRESS` to `COMPLETED` with certified technician audit notes."*

### 5. Automated Retraining Demonstration (2 Minutes)
- **Speaker:**  
  *"In Step 8, we demonstrate our automated MLOps retraining workflow. When a new batch of labeled run-to-failure data arrives, our system trains a candidate model and benchmarks it against the existing production champion. Our retraining policy strictly enforces that a candidate is **only promoted** if its validation RMSE improves over the champion by at least 2.0%. If the candidate is worse, the champion is safely preserved and the rejection is logged in MLflow. Zero silent or unverified model promotions."*

### 6. Interactive Dashboard Walkthrough (1-2 Minutes)
- **Action:** Switch browser to Streamlit dashboard (`http://localhost:8501`).
- **Speaker:**  
  *"Finally, let us explore the **Self-Healing Machines** interactive command dashboard:*
  - *Page 1: Executive Overview gives the fleet manager high-level risk stratification and urgent dispatch alerts.*
  - *Page 2: Fleet Health allows real-time filtering across all 100 engines by priority and anomaly status.*
  - *Page 3: Engine Details lets engineers drill into individual sensor degradation curves over hundreds of cycles.*
  - *Page 4: Model Performance shows our actual predicted vs. true RUL scatter plot and feature importances.*
  - *Page 5: Work Orders allows technicians to update ticket status and export maintenance logs to CSV.*
  - *Page 6: MLOps Monitoring tracks Population Stability Index (PSI) drift metrics to distinguish natural mechanical wear from global sensor calibration shift."*

### 7. Conclusion & Judging Q&A (1 Minute)
- **Speaker:**  
  *"To summarize: Self-Healing Machines is fully modular, transparent, leakage-free, and tested with 25 automated tests. Thank you, and we welcome the judges' questions."*
