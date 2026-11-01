# System Architecture: Self-Healing Machines

## 1. High-Level System Architecture

The **Self-Healing Machines** platform implements an end-to-end, edge-to-depot predictive maintenance architecture for connected industrial assets. The system ingests multivariate telemetry streams, detects uncharacteristic anomalies, predicts Remaining Useful Life (RUL), applies deterministic safety triage rules, and dispatches persistent maintenance work orders to an operational SQLite registry.

```mermaid
flowchart TD
    subgraph INGESTION["1. Telemetry Ingestion & Validation"]
        raw[Raw Sensor Telemetry<br/>26 Channels / Engine] --> val[Data Integrity Validator<br/>Schema, Nulls, Monotonicity]
    end

    subgraph PIPELINE["2. Feature Engineering & Preprocessing"]
        val --> prep[Variance Filter & Scaler<br/>Fitted strictly on Training split]
        prep --> feat[Temporal Feature Engineer<br/>Multi-window Rolling Mean/Std/Trend<br/>Engine-Boundary Isolated]
    end

    subgraph INFERENCE["3. Dual-Track ML Inference"]
        feat --> iforest[Track A: Isolation Forest<br/>Unsupervised Anomaly Scoring]
        feat --> regressor[Track B: HistGradientBoosting<br/>Supervised RUL Regression]
    end

    subgraph TRIAGE["4. Deterministic Triage Agent"]
        iforest --> triage[Rule-Based Triage Policy<br/>Deterministic Thresholds<br/>Defensive Fail-Safe Guards]
        regressor --> triage
    end

    subgraph DISPATCH["5. Persistence & Operations"]
        triage --> wos[Work Order Service<br/>Idempotent / Duplicate Guard]
        wos --> db[(SQLite Database<br/>WAL Mode Persistent Store)]
        db --> dash[Streamlit Dashboard<br/>Industrial UI & Plotly Charts]
        db --> mflow[MLflow Lineage & Monitoring]
    end
```

---

## 2. Data Flow & Leakage Defense Mechanisms

Data leakage is the most critical failure mode in prognostic time-series models. Self-Healing Machines enforces three strict architectural safeguards:

### A. Engine-Grouped Splitting
Observations belonging to an individual physical engine are **never split** across training and validation sets. Splitting is executed using `GroupShuffleSplit` on `engine_id`.

```mermaid
flowchart LR
    Fleet["All Fleet Engines (N=100)"]
    Fleet -->|80 Engines| Train["Training Fold (Strictly Isolated)"]
    Fleet -->|20 Engines| Val["Validation Fold (Strictly Isolated)"]
```

### B. Engine Boundary Protection in Feature Windows
Rolling statistics (windows: 5, 10, 20 cycles) are calculated strictly within each engine's trajectory via `df.groupby('engine_id')`. No rolling calculation is ever allowed to cross the boundary between engine $i$ and engine $i+1$.

### C. Causal Ordering (Strict Past & Present Observations)
Features for cycle $t$ use only measurements from cycles $\tau \le t$. Future sensor readings and eventual failure cycles are never accessible during feature construction or inference.

---

## 3. Dual-Track Machine Learning Architecture

```mermaid
flowchart TD
    subgraph AnomalyDetection["Track A: Instantaneous Deviation Detection"]
        AD_Desc["Isolation Forest<br/>Unsupervised Out-of-Distribution Detector<br/>Normalizes scores to [0.0, 1.0]<br/>Flags unexpected transient faults & spikes"]
    end

    subgraph Prognostics["Track B: Cumulative Degradation Forecasting"]
        PR_Desc["HistGradientBoosting Regressor<br/>Trained on Piecewise Capped RUL (125 cycles)<br/>Predicts remaining operational cycles to failure"]
    end

    subgraph Fusion["Deterministic Fusion in Triage Agent"]
        AD_Desc --> FusionRule["Rule Engine: Combines RUL baseline with anomaly escalation"]
        PR_Desc --> FusionRule
    end
```

### Why Anomaly Detection and RUL Regression are Different Tasks
1. **Anomaly Detection** answers: *"Is the machine operating outside nominal statistical behavior right now?"* It is sensitive to transient events, sensor faults, and novel operating conditions.
2. **RUL Prediction** answers: *"How many operating cycles remain before structural or thermal failure?"* It learns long-term monotonic wear trajectories.
3. An anomaly is an indicator for investigation, **not a guaranteed imminent failure**. Conversely, an engine deep in its degradation trajectory might have smooth, predictable sensor trends that don't trigger anomaly detectors but have a critically low RUL.

---

## 4. Deterministic Triage & Work Order Lifecycle

Maintenance workflows in safety-critical systems must never rely on probabilistic generative AI models (LLMs) that can hallucinate. The platform implements deterministic Python decision rules:

```mermaid
stateDiagram-v2
    [*] --> IngestTelemetry
    IngestTelemetry --> Validated: Validation Passed
    IngestTelemetry --> DataFaultCritical: Validation Failed

    Validated --> EvaluateRUL
    EvaluateRUL --> CriticalRUL: RUL <= 15
    EvaluateRUL --> HighRUL: 15 < RUL <= 30
    EvaluateRUL --> MediumRUL: 30 < RUL <= 60
    EvaluateRUL --> LowRUL: RUL > 60

    LowRUL --> MediumRUL: Severe Anomaly (Score >= 0.70)
    MediumRUL --> HighRUL: Severe Anomaly (Score >= 0.70)
    HighRUL --> CriticalRUL: Severe Anomaly (Score >= 0.70)

    CriticalRUL --> DispatchWorkOrder
    HighRUL --> DispatchWorkOrder
    MediumRUL --> DispatchWorkOrder
    DataFaultCritical --> DispatchWorkOrder

    DispatchWorkOrder --> IdempotencyCheck
    IdempotencyCheck --> SuppressDuplicate: Active Ticket Exists
    IdempotencyCheck --> InsertSQLite: No Active Ticket

    InsertSQLite --> OpenTicket
    OpenTicket --> InProgress: Technician Assigned
    InProgress --> Completed: Maintenance Certified
    Completed --> [*]
```
