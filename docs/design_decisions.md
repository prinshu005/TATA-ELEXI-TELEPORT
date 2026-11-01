# Design Decisions & Engineering Trade-Offs

This document records the foundational architectural decisions, rationale, and trade-offs made during the development of the **Self-Healing Machines** platform.

---

## 1. Piecewise Linear RUL Target (RUL Capping at 125 Cycles)

### Decision
Apply a piece-wise linear degradation curve that caps maximum target RUL to 125 cycles:
$$RUL_{\text{target}}(t) = \min(RUL_{\text{actual}}(t), 125)$$

### Rationale
In turbofan engines, physical wear signatures (e.g. HPC degradation, blade clearances, exhaust gas temperature rises) are negligible during early operational life. An uncapped linear target forces the regressor to distinguish between cycle 300 and cycle 200 where sensors exhibit identical healthy nominal readings. This introduces substantial target noise, hurting generalization. Capping at 125 cycles concentrates model parameters on active damage propagation.

### Trade-Off
The model does not distinguish between an engine with 250 cycles remaining and one with 150 cycles remaining. For fleet dispatchers, this distinction is irrelevant since both engines are healthy and require no maintenance intervention.

---

## 2. Dual-Model Architecture: Anomaly Detection vs. RUL Regression

### Decision
Implement two independent machine learning tracks:
1. **Isolation Forest** for unsupervised instantaneous anomaly scoring.
2. **HistGradientBoosting** for supervised RUL regression.

### Rationale
- **RUL Regressors** predict long-term wear curves based on known historical degradation trends.
- **Anomaly Detectors** flag sudden uncharacteristic events (e.g., bird strike, sensor bias, electrical surge, oil line pressure dip) that might not follow typical gradual wear paths.
- Treating anomaly detection and RUL as the same task creates severe blind spots: an engine can be near failure with completely normal wear patterns, or brand new with an anomalous sensor spike.

---

## 3. Deterministic Python Triage vs. Generative LLMs

### Decision
Enforce transparent, deterministic Python rules for triage and maintenance ticketing instead of using Large Language Models (LLMs).

### Rationale
Maintenance and reliability engineering require **100% auditability, determinism, and zero hallucination risk**. An LLM can introduce non-deterministic variations, fail edge-case rules, or produce ungrounded recommendations. A deterministic Python rule agent:
- Can be mathematically proven and tested with 100% test coverage.
- Dispatches reproducible work-order IDs.
- Provides exact rule citations for technical judges and regulatory bodies (FAA/EASA).

---

## 4. Multi-Window Rolling Features with Engine Boundary Protection

### Decision
Compute rolling means, standard deviations, and short-term trends across windows of 5, 10, and 20 cycles, grouped strictly by `engine_id`.

### Rationale
Turbofan sensor telemetry is noisy. Multi-window rolling statistics capture:
- High-frequency sensor noise reduction (rolling mean).
- Volatility and vibration shifts (rolling standard deviation).
- Rate of change and transient divergence (lag differences and trend).
Grouping strictly by `engine_id` prevents cross-engine contamination at sequence boundaries.

---

## 5. Persistent SQLite Backend with WAL Mode

### Decision
Use SQLite with Write-Ahead Logging (WAL) enabled as the persistence layer for work orders, telemetry snapshots, and audit logs.

### Rationale
- Zero-configuration and completely serverless (no external daemon or cloud subscription required).
- Native ACID compliance and idempotent operation.
- WAL mode allows concurrent readers (Streamlit interactive queries) and background writers (batch inference pipelines) without locking conflicts.
