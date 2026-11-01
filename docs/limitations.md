# Limitations & Operational Safety Considerations

While the **Self-Healing Machines** system achieves state-of-the-art predictive maintenance accuracy on benchmark datasets, real-world deployment requires understanding its inherent assumptions and physical limitations.

---

## 1. NASA C-MAPSS Simulation Environment Assumptions
- **Thermodynamic Simulation:** C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) models physics-based turbofan behavior under controlled environmental assumptions. Real-world aircraft engines experience non-linear operational factors, such as:
  - Atmospheric dust, volcanic ash, and salt-fog ingestion.
  - Variable pilot throttle dynamics and aggressive thrust reverser usage.
  - Sensor drift due to wiring harness thermal fatigue.
- **Single Flight Regime in FD001:** Subset FD001 operates under a single steady sea-level operating condition. Subsets FD002 and FD004 introduce six discrete flight operating conditions (altitude, Mach number, throttle resolver angle) requiring regime-specific normalization.

---

## 2. Point Predictions vs. Epistemic Uncertainty
- The current primary model outputs a point estimate of Remaining Useful Life (cycles).
- While stratified validation demonstrates high confidence near failure ($MAE \approx 7.3$ cycles for $RUL \le 30$), point estimates should never be treated as exact calendar dates.
- In production, confidence intervals (e.g., via quantile regression or conformal prediction) should accompany point estimates to communicate prediction uncertainty.

---

## 3. Absence of Certified Physical Control Authority
- **Advisory Role Only:** The platform is designed strictly as a clinical decision-support system. It automatically generates work orders for qualified human engineers and line technicians.
- Under FAA (Federal Aviation Administration) and EASA airworthiness regulations (e.g. FAR Part 33 and DO-178C), no software system without formal DO-178C Level-A certification may command autonomous engine shutdown or flight-critical throttle modulation.

---

## 4. Retraining on Unlabeled Fleet Streams
- Supervised RUL retraining requires verified run-to-failure cycles (known failure points). Active fleet engines that are serviced or overhauled prior to catastrophic failure do not provide complete ground truth RUL without right-censored survival analysis techniques.
- When new fleet data arrives without failure labels, the system appropriately halts supervised RUL model promotion while continuing unsupervised anomaly baseline updates.
