# Competition Presentation: Self-Healing Machines
### AI-Powered Predictive Maintenance for a Connected Fleet

**Target Audience:** Technical Judging Panel, Industry Practitioners, and Reliability Engineers  
**Presentation Format:** 15 Slides with Speaker Notes and Visual Recommendations  

---

### Slide 1: Title & Team Introduction
- **Slide Title:** Self-Healing Machines: AI-Powered Predictive Maintenance for a Connected Fleet
- **Subtitle:** Closing the Loop from High-Frequency Telemetry to Automated Maintenance Work Orders
- **Main Points:**
  - Complete, end-to-end predictive maintenance intelligence platform.
  - Built on NASA C-MAPSS Turbofan Engine Degradation dataset.
  - Zero-hallucination deterministic triage coupled with supervised prognostics and unsupervised anomaly detection.
- **Suggested Visual:** Modern dark-industrial title card with a 3D turbofan engine schematic, connected telemetry nodes, and team member names/roles.
- **Speaker Notes:**
  > *"Good morning, esteemed judges. We are team Self-Healing Machines. Today, we are proud to present an industrial-grade, fully working predictive maintenance platform for connected aircraft fleets. We take raw, noisy sensor streams from turbofan engines and transform them into actionable, auditable maintenance work orders."*

---

### Slide 2: Problem Statement
- **Slide Title:** The Aviation Maintenance Dilemma: High Costs and In-Flight Hazards
- **Main Points:**
  - Unscheduled maintenance accounts for massive airline operational disruptions and safety risks.
  - A single unscheduled engine removal can ground an aircraft for days, costing \$100,000+ per event.
  - Late detection of High Pressure Compressor (HPC) wear risks catastrophic in-flight shutdown (IFSD).
- **Suggested Visual:** Infographic illustrating an aircraft grounded at a gate, displaying metrics on unscheduled delay costs and flight safety exposure.
- **Speaker Notes:**
  > *"Commercial aviation operates on razor-thin margins and uncompromising safety standards. Unscheduled engine removals disrupt flight schedules, strand passengers, and introduce severe safety hazards. The core challenge is that modern jet engines operate inside extreme thermal and mechanical environments where degradation starts microscopically long before catastrophic failure occurs."*

---

### Slide 3: Why Traditional Maintenance is Insufficient
- **Slide Title:** Moving Beyond Reactive & Calendar-Based Maintenance
- **Main Points:**
  - **Run-to-Failure (Corrective):** Unacceptable in aerospace; risks structural failure and catastrophic incidents.
  - **Time/Cycle-Based (Preventative):** Rigid servicing schedules (e.g. overhaul every 2,000 hours) discard healthy parts with significant remaining life, wasting millions.
  - **The Missing Link:** Condition-Based Predictive Maintenance that tracks actual physical wear in real time.
- **Suggested Visual:** Comparison matrix contrasting Corrective, Preventative, and Predictive maintenance regimes across Cost, Asset Utilization, and Safety.
- **Speaker Notes:**
  > *"Why can't traditional maintenance solve this? Corrective maintenance is unthinkable in safety-critical flight. Preventative maintenance—servicing components on fixed flight-hour intervals—is blind to actual wear: an engine flown in hot, dusty climates wears much faster than one in mild conditions. Traditional schedules discard healthy components prematurely while missing accelerated wear in severe conditions."*

---

### Slide 4: Proposed Solution
- **Slide Title:** Self-Healing Machines: Closed-Loop Fleet Intelligence
- **Main Points:**
  - Continuous ingestion and validation of 21 physical sensor telemetry streams.
  - Dual-track analytics: Unsupervised Anomaly Detection + Supervised RUL Prognostics.
  - Deterministic triage agent with auditable maintenance action mapping.
  - Persistent SQLite work order registry with duplicate ticket suppression.
  - Production-ready Streamlit command center and MLflow experiment governance.
- **Suggested Visual:** Conceptual diagram showing data flowing from an aircraft turbofan $\to$ AI Inference Pipeline $\to$ Work Order Registry $\to$ Line Maintenance Hangar.
- **Speaker Notes:**
  > *"Our solution is Self-Healing Machines. We do not stop at training a model in a notebook. We built a complete, production-grade closed-loop platform that validates incoming telemetry, scores anomalies with Isolation Forests, predicts Remaining Useful Life with gradient-boosted trees, evaluates risk with deterministic safety rules, and persists auditable work orders directly into a maintenance database."*

---

### Slide 5: System Architecture
- **Slide Title:** End-to-End System Architecture & Leakage Defense
- **Main Points:**
  - 5-stage architecture: Ingestion $\to$ Feature Engineering $\to$ Dual ML Inference $\to$ Deterministic Triage $\to$ Work Order Registry.
  - **Leakage Prevention Pillar 1:** Grouped splitting by engine ID (zero cross-engine data leakage).
  - **Leakage Prevention Pillar 2:** Multi-window rolling features isolated strictly within engine boundaries.
  - **Leakage Prevention Pillar 3:** Strictly causal temporal ordering (past and present observations only).
- **Suggested Visual:** Architectural flow diagram highlighting the boundary isolation walls between engines and the strict training/validation partition.
- **Speaker Notes:**
  > *"Architecture integrity was our highest design priority. In prognostic modeling, data leakage is rampant in amateur pipelines. If rolling statistics from Engine 1 bleed into Engine 2, or if future cycles contaminate current features, reported metrics look artificially perfect. Our platform enforces strict engine-boundary isolation and causal ordering, verified across our automated test suite."*

---

### Slide 6: Data Engineering & Feature Engineering
- **Slide Title:** Sensor Trajectory Processing & Physics-Informed Feature Engineering
- **Main Points:**
  - Automated variance filtering: drops 7 uninformative/constant channels (`setting_3`, `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`).
  - Multi-window rolling statistics: Means and Standard Deviations across 5, 10, and 20 cycles.
  - Cycle-to-cycle first-order differences and short-term trend deviation metrics.
  - Expands 17 active sensor channels into 137 high-signal temporal features.
- **Suggested Visual:** Plot showing noisy raw sensor readings (e.g. HPC outlet temperature) contrasted with the smoothed rolling mean and volatility envelope.
- **Speaker Notes:**
  > *"Raw turbine telemetry is noisy due to thermal turbulence and sensor vibration. We filter out non-informative constant channels and construct multi-window temporal features over 5, 10, and 20 cycle horizons. This captures both long-term degradation trajectories and short-term volatility spikes, generating 137 highly predictive features without human feature hand-crafting."*

---

### Slide 7: Anomaly Detection Approach
- **Slide Title:** Track A: Unsupervised Isolation Forest Anomaly Detection
- **Main Points:**
  - Learns normal operating boundaries on nominal training cycles.
  - Identifies out-of-distribution observations and sensor anomalies.
  - Outputs normalized continuous anomaly score in $[0.0, 1.0]$.
  - Categorizes into `NORMAL`, `WARNING`, and `ANOMALOUS`.
  - **Key Concept:** An anomaly indicates an uncharacteristic event for investigation—it is complementary to, but fundamentally different from, RUL prediction.
- **Suggested Visual:** Isolation Forest score distribution plot with configurable warning ($0.55$) and critical ($0.75$) threshold cutoffs.
- **Speaker Notes:**
  > *"Track A is our unsupervised Isolation Forest. Why do we need anomaly detection if we already predict RUL? Because an engine can suffer a sudden foreign object damage or sensor bias while having a high predicted RUL. The Isolation Forest scores instantaneous deviations from nominal flight baselines, providing early warning signals before wear patterns fully manifest."*

---

### Slide 8: RUL Prediction Approach
- **Slide Title:** Track B: Piecewise Linear RUL Regression
- **Main Points:**
  - **Why RUL Capping at 125 Cycles?** Early engine cycles exhibit healthy baseline variation without physical wear. Uncapped linear targets force models to distinguish between cycle 300 and cycle 200 with identical sensor readings. Capping at 125 cycles concentrates learning on active degradation.
  - Primary Model: `HistGradientBoostingRegressor` (fast, non-linear, robust to feature correlations).
  - Enforces physical non-negativity constraint ($RUL \ge 0.0$).
- **Suggested Visual:** Side-by-side plot showing the uncapped linear degradation target vs. the piecewise capped target curve (flat at 125, then linearly declining).
- **Speaker Notes:**
  > *"For Track B, we model Remaining Useful Life. A critical innovation validated in PHM literature is piecewise linear capping. In early operational life, machinery operates in a healthy steady-state. Capping the target at 125 cycles allows our HistGradientBoosting regressor to ignore benign early variance and focus all model capacity on the steep wear propagation phase."*

---

### Slide 9: Model Evaluation & Actual Benchmark Results
- **Slide Title:** Official NASA C-MAPSS FD001 Benchmark Performance
- **Main Points:**
  - Evaluated on **100 held-out test engines** evaluated at their final operating cycle against official ground truth:
    - **Mean Absolute Error (MAE):** `13.62 cycles`
    - **Root Mean Squared Error (RMSE):** `18.92 cycles`
    - **$R^2$ Score:** `0.7928`
    - **NASA PHM'08 Penalty Score:** `1099.86 points`
  - **Stratified Breakdown:** For engines in the critical failure zone ($RUL \le 30$), MAE drops to **7.34 cycles**!
- **Suggested Visual:** Actual predicted vs. actual RUL scatter plot alongside the stratified performance table.
- **Speaker Notes:**
  > *"Here are our actual measured results on the official held-out NASA C-MAPSS test set of 100 engines. We achieved an overall MAE of 13.62 cycles and an RMSE of 18.92 cycles. Crucially, look at our stratified error analysis: when engines enter the critical stage with less than 30 cycles remaining, our MAE drops to just 7.34 cycles! When precision matters most, the model is accurate down to approximately 7 flight cycles."*

---

### Slide 10: Rule-Based Triage Agent
- **Slide Title:** Deterministic Triage: Zero Black-Box Generative Hallucinations
- **Main Points:**
  - Transparent Python decision logic mapping RUL and anomaly indicators to priority tiers:
    - **CRITICAL:** $RUL \le 15$ cycles
    - **HIGH:** $15 < RUL \le 30$ cycles
    - **MEDIUM:** $30 < RUL \le 60$ cycles
    - **LOW:** $RUL > 60$ cycles
  - **Dynamic Escalation:** A severe anomaly ($\text{score} \ge 0.70$) escalates priority by 1 level (e.g. MEDIUM $\to$ HIGH).
  - **Defensive Guard:** Missing telemetry or inference failure defaults to **CRITICAL** with an alert—never assumed healthy.
- **Suggested Visual:** Decision tree flowchart showing the deterministic triage logic and anomaly escalation paths.
- **Speaker Notes:**
  > *"A critical competition requirement was: do not use an LLM for maintenance decisions. In aerospace engineering, safety workflows must be 100% deterministic and auditable. Our triage agent uses transparent Python rules. Furthermore, we built defensive fail-safes: if telemetry drops or inference fails, the system automatically triggers a CRITICAL data investigation ticket. A missing prediction is never treated as a healthy engine."*

---

### Slide 11: Automatic Maintenance Work Orders
- **Slide Title:** Persistent Work Order Dispatch & Lifecycle Tracking
- **Main Points:**
  - Backed by persistent **SQLite database** with Write-Ahead Logging (`WAL`).
  - **Idempotency Guard:** Prevents duplicate active tickets if subsequent cycles re-trigger the same condition.
  - Lifecycle state machine: `OPEN` $\to$ `IN_PROGRESS` $\to$ `COMPLETED`.
  - Advisory maintenance recommendations (e.g., HPC borescope inspection, clearance calibration).
  - Instant one-click CSV export for integration with enterprise ERP/MRO systems (SAP/Maximo).
- **Suggested Visual:** UI screenshot of the work order registry table showing unique ticket IDs, priorities, technician logs, and export buttons.
- **Speaker Notes:**
  > *"When a triage rule triggers, the work-order service immediately creates an auditable record in SQLite. It enforces idempotency: subsequent flight cycles for that same engine will not flood dispatchers with duplicate tickets. Technicians can view recommendations, assign depot bays, transition tickets from OPEN to IN_PROGRESS to COMPLETED, and export audit trails to CSV."*

---

### Slide 12: MLOps & Automated Retraining Demonstration
- **Slide Title:** Continuous Governance & Controlled Retraining Gate
- **Main Points:**
  - **MLflow Tracking:** Logs parameters, metrics, run artifacts, and model versions.
  - **Sensor Drift Detection:** Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests distinguish expected mechanical wear from global sensor calibration drift.
  - **Retraining Promotion Gate:** Candidate models must achieve $\ge 2.0\%$ validation RMSE improvement over the active champion model to be promoted; otherwise, the champion is safely preserved.
- **Suggested Visual:** Screenshot of MLflow experiment runs and a flowchart of the candidate-versus-champion promotion gate.
- **Speaker Notes:**
  > *"Machine learning models in production degrade if operating regimes change. Our platform includes an automated retraining manager integrated with MLflow. When new run-to-failure data arrives, a candidate model is trained and evaluated against a fixed validation set. Our promotion policy strictly requires at least a 2.0% RMSE improvement before champion replacement. If the candidate underperforms, it is rejected and logged."*

---

### Slide 13: Live Dashboard Demonstration
- **Slide Title:** Command Center: Streamlit Industrial Analytics Interface
- **Main Points:**
  - **Executive Overview:** Real-time fleet health cards, priority distributions, and urgent alert banners.
  - **Fleet Health:** Interactive filtering and search across all 100 engines by priority and anomaly status.
  - **Engine Details:** Time-series drill-down into multi-sensor degradation curves over hundreds of cycles.
  - **Model Performance:** Real-time parity scatter plots, residual distributions, and feature importance rankings.
  - **Work Orders & MLOps:** Ticket management and Population Stability Index drift tracking.
- **Suggested Visual:** Grid collage of the Streamlit dashboard views (Executive Overview, Engine Detail charts, Work Order editor).
- **Speaker Notes:**
  > *"Our interactive Streamlit command center brings the entire system to life. Fleet managers get instant visibility into fleet risk distributions; reliability engineers can inspect individual sensor trajectories; depot technicians can update maintenance statuses; and MLOps engineers can monitor data drift and trigger candidate model retraining with a single click."*

---

### Slide 14: Business Impact & Expected Benefits
- **Slide Title:** Operational & Financial Value for Fleet Operators
- **Main Points:**
  - **Targeted Unscheduled Maintenance Reduction:** Early warning allows maintenance scheduling during planned overnight turns, preventing costly gate delays.
  - **Asset Utilization Optimization:** Safely maximizing engine cycles before overhaul reduces premature depot visits.
  - **Spare Parts Inventory Forecasting:** Advance notice of component wear enables just-in-time turbine blade and stator procurement.
  - **Audit & Compliance Readiness:** Complete digital traceability for aviation regulatory authorities.
- **Suggested Visual:** Value chain diagram contrasting reactive maintenance costs with precision predictive scheduling.
- **Speaker Notes:**
  > *"While our performance metrics are measured on benchmark data, the operational implications are significant. By accurately flagging high-wear engines 15 to 30 cycles in advance, airlines can schedule line maintenance during planned overnight turns. This avoids gate delays, optimizes depot bay scheduling, and enables just-in-time spare parts logistics."*

---

### Slide 15: Limitations, Safety Scope & Future Roadmap
- **Slide Title:** Operational Boundaries & Next-Generation Capabilities
- **Main Points:**
  - **Current Boundaries:** C-MAPSS simulation baseline; point predictions rather than epistemic intervals; decision-support only (certified human sign-off required).
  - **Roadmap Item 1:** Conformal prediction intervals to quantify uncertainty on RUL estimates.
  - **Roadmap Item 2:** Multi-regime scaling to subsets FD002 and FD004 (six flight conditions).
  - **Roadmap Item 3:** Direct API integration with commercial MRO systems (Boeing Toolbox, Airbus Skywise).
- **Suggested Visual:** Roadmap timeline illustrating the progression from FD001 baseline $\to$ multi-condition scaling $\to$ enterprise MRO integration.
- **Speaker Notes:**
  > *"To conclude with engineering honesty: our current implementation operates as an advisory decision-support tool. It does not replace certified maintenance inspectors, nor does it command physical shutdowns. Moving forward, we plan to implement conformal prediction for uncertainty bounds and scale our regime-normalization algorithms to multi-condition subsets FD002 and FD004. Thank you for your time, and we look forward to your questions."*
