# IntelliCredit: Model Monitoring & Interactive Underwriter Operations Guide

> **Interactive Dashboard & Model Governance Documentation**  
> **Author:** Gowtham Reddy (@gowthamreddy285) - Interactive Dashboard & Full-Stack UI Lead  
> **Scope:** Streamlit Decision Engine, Population Stability Index (PSI) Surveillance, Counterfactual Simulator & Policy Matrix  

---

## 1. Overview & Architecture

The IntelliCredit interactive suite equips loan underwriters, risk officers, and model validators with real-time decisioning tools and automated statistical surveillance:

```
dashboard/
  ├── app.py                   # Production 5-tab loan underwriter decision engine
  ├── monitoring_engine.py     # Population Stability Index (PSI) & KS drift detector
  ├── monitoring_view.py       # Streamlit surveillance visualizer with Basel traffic lights
  ├── scenario_simulator.py    # Counterfactual what-if scenario simulation engine
  ├── policy_matrix.py         # Institutional credit policy guardrails & DTI/LTI caps
  ├── decision_audit.py        # Regulatory audit packet exporter & decision receipts
  └── ui_utils.py              # Design system, risk badges, and INR formatters
```

---

## 2. Population Stability Index (PSI) Surveillance

### 2.1 Mathematical Formulation
The Population Stability Index measures the shift between the baseline population (training distribution $E$) and the incoming production applicants ($A$):

$$\text{PSI} = \sum_{b=1}^{B} (A_b - E_b) \times \ln\left(\frac{A_b}{E_b}\right)$$

Where:
* $B$: Number of quantile bins (calibrated to $B = 10$).
* $E_b$: Proportion of baseline applicants falling into bin $b$.
* $A_b$: Proportion of current scoring applicants falling into bin $b$.
* $\epsilon = 10^{-4}$: Smoothing constant preventing numerical instability.

### 2.2 Supervisory Traffic-Light Standards (Basel & EBA)

| Alert Status | PSI Range | Operational Implication | Recommended Action |
|---|:---:|---|---|
| **Green (Stable)** | $\text{PSI} < 0.10$ | Negligible distribution shift; scorecard is well calibrated. | Normal operations. |
| **Amber (Warning)** | $0.10 \le \text{PSI} \le 0.25$ | Moderate covariate shift detected in incoming applicant pool. | Increase surveillance frequency; audit marketing channels. |
| **Red (Critical)** | $\text{PSI} > 0.25$ | Significant population drift; risk bands may be misaligned. | Mandatory model governance review; trigger model recalibration. |

---

## 3. Underwriter What-If Counterfactual Simulator

Borderline applicants (Band C and D) often get rejected due to rigid single-point evaluations. The What-If Simulator (`dashboard/scenario_simulator.py`) enables loan officers to test mitigating conditions:

* **Tenure Extension:** Spreading principal across an additional 3–12 months to lower monthly EMI and bring Debt-to-Inflow (DTI) below the 70% policy ceiling.
* **Principal Haircut:** Scaling down requested loan amounts to match verified debt service capacity.
* **Utility Payment Discipline:** Simulating applicant compliance improvements (reducing delay days from >20 to 0) to evaluate whether behavioral stabilization transitions the applicant into an approved risk band.

---

## 4. Institutional Credit Policy Guardrails

Implemented in `dashboard/policy_matrix.py`:

1. **Maximum Debt-to-Inflow (DTI):** Outflow must not exceed **70.0%** of monthly income.
2. **Maximum Loan-to-Income (LTI):** Principal requested must not exceed **3.0x** monthly income.
3. **Utility Delay Ceiling:** Average utility delay must not exceed **25 days**.
4. **Savings Buffer Floor:** End-of-month balance must cover at least **15.0%** of monthly outflow.
5. **Economic Risk Ceiling:** Calibrated PD must not exceed Break-Even PD + 15%.

---

## 5. Regulatory Decision Audit Trail

Every underwriter decision is logged via `dashboard/decision_audit.py` to produce a timestamped JSON audit packet capturing:
* Full applicant financial snapshot at decision time.
* Calibrated PD, Risk Band (A–E), and economic break-even rate.
* Triggered warning rules from Association Rule Mining (ARM).
* Loan officer override flag and mandatory justification commentary.

---

## 6. Execution & Verification Runbook

### Run Automated Unit Tests
```bash
python -m unittest tests/test_monitoring_and_ui.py -v
```

### Run Model Monitoring & PSI Drift Detection
```bash
python -m dashboard.monitoring_engine
```

### Run Policy Matrix Evaluation
```bash
python -m dashboard.policy_matrix
```

### Run Underwriter Counterfactual Simulator
```bash
python -m dashboard.scenario_simulator
```

### Launch Interactive Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
