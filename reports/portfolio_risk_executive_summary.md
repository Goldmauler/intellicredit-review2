# IntelliCredit: Executive Portfolio Risk & Credit Performance Report

> **Prepared by:** Portfolio Risk Modeling & Analytics Team  
> **As of:** 2026-11-10 12:41:59  
> **Scope:** Micro-Lending Portfolio Panel & Real-World Macroeconomic Drivers  

---

## 1. Executive Summary & Portfolio Health

| Portfolio Risk Indicator | Baseline Value | Risk Benchmark | Evaluation |
|---|:---:|:---:|:---:|
| **Total Exposure at Default (EAD)** | INR 577,458,900.50 | - | Active Portfolio |
| **Probability-Weighted ECL Provision** | INR 103,552,422.95 | - | Ind AS 109 Compliant |
| **Portfolio ECL Coverage Ratio** | 17.93% | 12.0% - 18.0% | Adequate Reserve |
| **Roll Rate (Current -> 30 DPD)** | 1.87% | < 3.0% | Healthy Flow |
| **Cure Rate (30 DPD -> Current)** | 38.00% | > 30.0% | Strong Self-Cure |
| **Average Early Payment Default (MOB 3)** | 0.76% | < 1.5% | Well Controlled |

---

## 2. IFRS 9 / Ind AS 109 Staged Provisioning Breakdown

Under the 3-stage expected credit loss framework:
* **Stage 1 (Performing):** 27,621 loans | Provision: **INR 15,703,165.29**
* **Stage 2 (SICR / 30-60 DPD):** 907 loans | Provision: **INR 4,016,395.54**
* **Stage 3 (Credit-Impaired / Default):** 1,472 loans | Provision: **INR 76,608,274.47**

### Forward-Looking Multi-Scenario ECL
* **Upside (20% Weight):** INR 81,878,660.00
* **Base Case (50% Weight):** INR 96,327,835.30
* **Downside (30% Weight):** INR 130,042,577.66
* **=> Regulatory Staged ECL:** **INR 103,552,422.95** (17.93% coverage)

---

## 3. Delinquency Markov Transition Matrix (1-Month)

```
to_state    Current  30_DPD  60_DPD  Default  Settled
from_state                                           
Current      0.8833  0.0187  0.0000   0.0000   0.0980
30_DPD       0.3800  0.0000  0.5741   0.0000   0.0459
60_DPD       0.2167  0.0000  0.0000   0.7549   0.0284
Default      0.0000  0.0000  0.0000   1.0000   0.0000
Settled      0.0000  0.0000  0.0000   0.0000   1.0000
```

* **Roll-forward 30 -> 60 DPD:** 57.41%
* **Roll-forward 60 -> Default:** 75.49%
* **Cure 30 -> Current:** 38.00%
* **Cure 60 -> Current:** 21.67%

---

## 4. Regulatory Macroeconomic Supervisory Stress Test

Simulated under RBI prudential stress scenarios using the Wilson logit hazard framework:

| Scenario | Macro Shock (z) | Stressed Default % | Stressed Loss (INR) | Stressed CRAR % | RBI 15% Breached? |
|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | 0.00 | 9.60% | INR 181,480,368.84 | 13.70% | Yes |
| **Moderate Adverse** | 1.25 | 19.31% | INR 364,995,439.58 | 8.84% | Yes |
| **Severe Adverse** | 2.50 | 34.01% | INR 642,736,307.26 | 1.49% | Yes |

---

## 5. Vintage Cohort Maturation & Early Warning Signals

* **Total Origination Vintages Analyzed:** 26 Quarters
* **Average EPD at MOB 3:** 0.76%
* **Average EPD at MOB 6:** 3.54%
* **Average Cumulative Default Rate at MOB 12:** 5.95%

### Recent Cohorts Performance (Last 6 Quarters)
```
                 cohort_size  EPD_MOB3_pct  EPD_MOB6_pct  Mature_MOB12_pct
vintage_quarter                                                           
2023-Q3                 1624          0.55          1.91              3.33
2023-Q4                 1721          0.52          1.69              2.85
2024-Q1                 1709          0.35          1.87              2.57
2024-Q2                 1777          0.17          1.35              1.52
2024-Q3                 1971          0.46          0.96              0.96
2024-Q4                 1340          0.00          0.00              0.00
```

---

## 6. Data Integrity & Leakage Verification

* **Audit Status:** **PASSED [OK]**
* **Checks Executed:** 8
* **Target and Ground-Truth Leakage:** None detected. Model training features remain fully partitioned.
