# Portfolio Risk Modeling & Staged Loss Framework Methodology

> **IntelliCredit Technical Documentation**  
> **Domain:** FinTech Micro-Lending Risk, Alternative Credit Scoring & Portfolio Analytics  
> **Author:** Abhinav Juturu (@abhinav-juturu) - Portfolio Risk Modeler & Analytics Lead  

---

## 1. Scope & Objective

This module provides institutional-grade portfolio risk modeling capabilities covering:
1. **Markov-Chain Transition Matrices & Roll Rates** (`src/transition_matrix.py`)
2. **Ind AS 109 / IFRS 9 Staged Expected Credit Loss (ECL)** (`src/ecl_engine.py`)
3. **Regulatory Macroeconomic Supervisory Stress Testing** (`src/stress_testing.py`)
4. **Cohort Vintage Maturation & Early Payment Default (EPD)** (`src/vintage_analysis.py`)
5. **Data Quality & Leakage Auditing** (`src/data_integrity.py`)

---

## 2. Markov Chain Delinquency Migration

### 2.1 State Space Definition
Let the loan delinquency status at time $t$ be modeled as a discrete-time Markov chain over state space:
$$S = \{\text{Current}, \text{30\_DPD}, \text{60\_DPD}, \text{Default}, \text{Settled}\}$$

Where:
* **Current (0 DPD):** Performing loan with zero overdue payments.
* **30_DPD (1â€“30 DPD):** One missed EMI payment; early delinquency.
* **60_DPD (31â€“60 DPD):** Two consecutive missed EMI payments; elevated default hazard.
* **Default (90+ DPD):** Non-performing asset (NPA); absorbing default state.
* **Settled:** Matured facility with zero outstanding balance; absorbing payoff state.

### 2.2 Transition Probability Matrix
The 1-month empirical transition probability $P_{ij} = \mathbb{P}(S_{t+1} = j \mid S_t = i)$ is estimated from the longitudinal performance panel:
$$P_{ij} = \frac{N_{ij}}{\sum_{k \in S} N_{ik}}$$

Where $N_{ij}$ represents the count of observed loan-month transitions from state $i$ to state $j$.

### 2.3 Roll-Forward and Cure Rates
* **Roll-Rate (Current $\to$ 30 DPD):** $P(\text{30\_DPD} \mid \text{Current})$
* **Roll-Rate (30 $\to$ 60 DPD):** $P(\text{60\_DPD} \mid \text{30\_DPD})$
* **Roll-Rate (60 $\to$ Default):** $P(\text{Default} \mid \text{60\_DPD})$
* **Cure Rate (30 $\to$ Current):** $P(\text{Current} \mid \text{30\_DPD})$
* **Cure Rate (60 $\to$ Current):** $P(\text{Current} \mid \text{60\_DPD})$

Multi-period state projections over horizon $k$ months are computed via the Chapman-Kolmogorov matrix power:
$$P^{(k)} = P^k$$

---

## 3. Ind AS 109 / IFRS 9 Staged Expected Credit Loss (ECL)

### 3.1 3-Stage Classification Architecture
In accordance with Ind AS 109 and RBI guidelines:

| Stage | Classification Criteria | Loss Provision Horizon |
|---|---|---|
| **Stage 1 (Performing)** | 0 DPD and stable credit risk | **12-Month ECL** |
| **Stage 2 (Underperforming / SICR)** | 30â€“60 DPD or PD jump $\ge 2.0\times$ origination PD | **Lifetime ECL** |
| **Stage 3 (Credit-Impaired)** | 90+ DPD (objective evidence of impairment) | **Lifetime ECL ($PD = 1.0$)** |

### 3.2 Expected Credit Loss Formulation
The baseline provision for loan facility $i$ with Exposure at Default $EAD_i$ and Loss Given Default $LGD$ is:

$$\text{ECL}_i = \sum_{t=1}^{T} \frac{\mathbb{P}(\text{Default at } t) \times LGD \times EAD_i(t)}{(1 + EIR)^t}$$

Where:
* $EAD_i$: Outstanding principal balance at measurement date.
* $LGD$: Benchmark loss given default (calibrated at 50% for unsecured microfinance).
* $EIR$: Effective Interest Rate used for present-value discounting.

### 3.3 Forward-Looking Macroeconomic Overlays
Ind AS 109 mandates probability-weighted provisioning across multiple macroeconomic paths:
$$\text{ECL}_{\text{weighted}} = w_{\text{up}} \cdot \text{ECL}_{\text{up}} + w_{\text{base}} \cdot \text{ECL}_{\text{base}} + w_{\text{down}} \cdot \text{ECL}_{\text{down}}$$

Calibrated weights: Upside (20%), Base Case (50%), Downside (30%).

---

## 4. Wilson Macroeconomic Stress Linkage & Capital Adequacy

### 4.1 Wilson (1997) Logit-Link Formulation
To link portfolio delinquency hazards with systemic shocks, individual default probabilities are shifted on the logit scale via the Wilson CreditPortfolioView formulation:
$$\text{logit}(PD_{\text{stressed}}) = \text{logit}(PD_{\text{base}}) + \beta_{\text{stress}} \cdot Z_{\text{macro}}$$

Where $Z_{\text{macro}}$ is a standardized composite index of macroeconomic stress:
* GDP growth contraction
* CPI food/fuel inflation surge
* RBI repo / call-money rate hikes
* INR currency depreciation

### 4.2 Capital Adequacy Ratio (CRAR) Depletion
The impact of stressed credit losses on NBFC regulatory capital is modeled as:
$$\text{CRAR}_{\text{stressed}} = \frac{\text{Tier 1 Capital} - \Delta \text{Loss}_{\text{stressed}}}{\text{Risk-Weighted Assets (RWA)}}$$

Under RBI regulations for NBFC-MFIs, the minimum required CRAR is **15.0%**. Scenarios breaching this threshold trigger capital conservation warnings.

---

## 5. Vintage Maturation & Early Payment Default (EPD)

Loans are segmented into origination quarterly cohorts $V_q$ and monitored across Months on Books ($\text{MOB} \in \{1, 2, \dots, 24\}$).

* **Cumulative Default Rate at MOB $m$:**
  $$CDR_q(m) = \frac{\sum_{k=1}^m \text{Defaults}_q(k)}{\text{Cohort Size}_q}$$
* **EPD-30:** Cumulative default rate at $\text{MOB} = 3$ (detects origination fraud and underwriting leakage).
* **EPD-60:** Cumulative default rate at $\text{MOB} = 6$ (measures early loan seasoning).

---

## 6. References
1. **Wilson, T. C. (1997).** *Portfolio Credit Risk.* Risk, 10(9), 111â€“117.
2. **Basel Committee on Banking Supervision (BCBS).** *Guidance on credit risk and accounting for expected credit losses.*
3. **Reserve Bank of India (RBI).** *Master Direction â€“ Regulatory Framework for Microfinance Loans (2022).*
4. **Ministry of Corporate Affairs (MCA), India.** *Indian Accounting Standard (Ind AS) 109: Financial Instruments.*

