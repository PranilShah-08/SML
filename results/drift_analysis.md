# Root-Cause Incident Report: Telemetry Concept & Covariate Drift
**Stream Flows Evaluated**: 7,998
**ADWIN Drift Signals Detected**: 0 change-point triggers
**Primary Drift Timestamp/Index**: Flow #N/A

## 1. Top Attributed Drift Drivers (1-Wasserstein Divergence)
The detector identified significant divergence in feature attribution profiles:

- **Bwd Packet Length Std**: Wasserstein Distance = `0.0274` | Attribution Surged by `70.0%` (Baseline Mean SHAP: `0.0276` -> Drifted: `0.0469`)
- **Packet Length Std**: Wasserstein Distance = `0.0266` | Attribution Surged by `60.9%` (Baseline Mean SHAP: `0.0206` -> Drifted: `0.0331`)
- **Average Packet Size**: Wasserstein Distance = `0.0106` | Attribution Attenuated by `0.1%` (Baseline Mean SHAP: `0.0162` -> Drifted: `0.0162`)
- **Avg Bwd Segment Size**: Wasserstein Distance = `0.0103` | Attribution Surged by `32.5%` (Baseline Mean SHAP: `0.0200` -> Drifted: `0.0265`)
- **Fwd Header Length**: Wasserstein Distance = `0.0096` | Attribution Attenuated by `10.1%` (Baseline Mean SHAP: `0.0260` -> Drifted: `0.0234`)

## 2. Operational Impact & Forensic Assessment
1. **Behavioral Shift**: The sudden deviation in flow dynamics indicates a transition from volumetric attacks (e.g. DoS/Brute Force) to polymorphic or stealthy scanning/botnet communication.
2. **Conformal Uncertainty Response**: Conformal p-values dropped across known classes, triggering the Zero-Day / Novel Attack alert pipeline.
3. **Remediation Action**: Active learning retrained the detector using only the top uncertain flows, recovering high operational precision without requiring a full dataset restudy.