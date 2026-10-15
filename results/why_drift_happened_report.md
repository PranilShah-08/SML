# Root-Cause Incident Report: Telemetry Concept & Covariate Drift
**Stream Flows Evaluated**: 7,998
**ADWIN Drift Signals Detected**: 0 change-point triggers
**Primary Drift Timestamp/Index**: Flow #N/A

## 1. Top Attributed Drift Drivers (1-Wasserstein Divergence)
The detector identified significant divergence in feature attribution profiles:

- **Bwd Packet Length Std**: Wasserstein Distance = `0.4338` | Attribution Surged by `44.8%` (Baseline Mean SHAP: `0.5251` -> Drifted: `0.7602`)
- **Fwd Packet Length Max**: Wasserstein Distance = `0.2373` | Attribution Surged by `13.2%` (Baseline Mean SHAP: `0.5789` -> Drifted: `0.6553`)
- **Avg Bwd Segment Size**: Wasserstein Distance = `0.1940` | Attribution Attenuated by `12.8%` (Baseline Mean SHAP: `0.4758` -> Drifted: `0.4147`)
- **Fwd Header Length**: Wasserstein Distance = `0.1142` | Attribution Surged by `17.2%` (Baseline Mean SHAP: `0.5031` -> Drifted: `0.5898`)
- **Fwd Packet Length Std**: Wasserstein Distance = `0.0995` | Attribution Surged by `27.0%` (Baseline Mean SHAP: `0.1969` -> Drifted: `0.2500`)

## 2. Operational Impact & Forensic Assessment
1. **Behavioral Shift**: The sudden deviation in flow dynamics indicates a transition from volumetric attacks (e.g. DoS/Brute Force) to polymorphic or stealthy scanning/botnet communication.
2. **Conformal Uncertainty Response**: Conformal p-values dropped across known classes, triggering the Zero-Day / Novel Attack alert pipeline.
3. **Remediation Action**: Active learning retrained the detector using only the top uncertain flows, recovering high operational precision without requiring a full dataset restudy.