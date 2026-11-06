# Comparison with existing studies (for Final Report Section 6.2)

Compiled 2026-11-06 by SCOUT. Reference numbers follow `REFERENCES_ACM.md`. Only numbers verified in the source (or in a quoted excerpt of it) are shown. Cells marked "n/v" mean the number exists in the paper but could not be verified, so it is left out on purpose.

## Table 1: Default-prediction discrimination (classification)

| # | Study | Data (country, n, default rate) | Method | Reported metric | Why direct comparison may be invalid |
|---|---|---|---|---|---|
| — | **IntelliCredit (this project)** | **Synthetic** India thin-file / gig borrowers, n = 30,000 (6,000 test), default 7.87% | Tuned XGBoost / LightGBM / LR / RF; SHAP | **AUC 0.738 (XGB)**, 0.735 (LGBM), 0.728 (LR), 0.713 (RF); calibrated LGBM 0.741; **KS 0.36**; Gini = 2·AUC − 1 ≈ **0.48**. Ceiling AUC from the true PD = 0.781 | Labels are random draws from a known PD, so the maximum achievable AUC is 0.781. Our numbers measure how much of the *built-in* signal is recovered (~85%), not real-world performance. |
| [1] | Berg et al. 2020, RFS | German e-commerce purchases, ~250k–270k, Oct 2015–Dec 2016 | Logit on digital footprint vs bureau score | AUC: bureau **0.683**, footprint **0.696**, combined **0.736** | Real e-commerce invoice credit with a very different product and default definition. It shows the *scale* of AUC that alternative data achieves. |
| [3] | Óskarsdóttir et al. 2019, ASC | Bank + telco, 22,000 applicants, default 4.49% | LR / DT / RF on socio-demographic (SD) and call-detail (CB) features | RF AUC: SD **0.899**, CB **0.870**, SD+CB **0.923**. LR AUC: SD 0.587, CB 0.535, SD+CB 0.612 | 300+ features from real bank and telco records, credit-card product. The RF/LR gap is unusually large, so treat it as dataset-specific. |
| [4] | Agarwal, Alok, Ghosh & Gupta 2020 (WP) | **Indian** fintech lender applicants (n not verified) | Logit / ML on mobile + social footprint | AUC: credit score only **0.59** (0.586 in Oct 2019 draft); footprint only **0.604** | The closest real Indian analogue. Its AUCs are much lower than ours, which is a reminder that synthetic signal is cleaner than real data. Numbers vary by paper version. |
| [2] | Björkegren & Grissen 2020, WBER | South American telecom borrowers (n not verified) | ML on phone-usage features | Top vs bottom risk quintile default ratio **2.8×**; beats bureau models on thin-file borrowers. AUC n/v | A different metric (quintile ratio). Compare it with our top-decile lift of 2.9×. |
| [51] | Fuster et al. 2022, JF | US mortgages, 9.4M loans 2009–2013; default = 90+ DPD within 3 years | Logit vs random forest | ROC AUC: logit **≈0.748**, RF **≈0.753** (authors' 2020 slides; version-specific) | Secured mortgages, huge n, prime-heavy. The point is that the ML gain over logit is small (+0.005). We see LR 0.728 vs XGB 0.738, a similarly small gap. |
| [7]/[1] | Iyer et al. 2016 (as cited by [1]) | P2P lending | — | **AUC ≈ 0.60 is "desirable" in information-scarce settings; ≥ 0.70 is the goal in information-rich ones** | A rule of thumb, not a result. Use it to interpret our 0.738. |
| [8] | Lessmann et al. 2015, EJOR | 8 real credit datasets | 41 classifiers | Heterogeneous ensembles rank best; several methods significantly beat LR; RF recommended as the benchmark. Per-dataset AUC n/v | Ranking evidence only. Supports our model menu, not our AUC level. |
| [9] | Gunnarsson et al. 2021, EJOR | Several real credit datasets | MLP, DBN vs 2 conventional + 2 ensemble methods | **XGBoost best**; deep nets no better and costlier. AUC n/v | Ranking evidence. Matches our result (XGB best). |
| [10] | Dastile et al. 2020, ASC | Review of 74 studies (2010–2018) | Meta-analysis | Ensembles generally beat single classifiers | Literature-level only. |
| [11] | Yeh & Lien 2009, ESWA | Taiwan credit-card clients, n = 30,000, 23 features | 6 data-mining methods | Only the ANN estimated the *real* PD accurately (calibration focus). Metric values n/v | Same n as ours by coincidence. Different product (credit cards) and country. |

**Suggested sentence for Section 6.2:** "Our XGBoost AUC of 0.738 lies between the bureau-only (0.683) and combined (0.736) models of Berg et al. [1], and above the ~0.60 reported for an Indian fintech's footprint model [4]. Because our labels come from a known synthetic PD (ceiling AUC 0.781), the comparison shows that the pipeline recovers about 85% of the available signal. It does not show that the model would reach 0.74 on real Indian borrowers. That is why we added a real-data track."

## Table 2: Forecasting accuracy and intervals

| # | Study | Data | Method | Reported metric | Why direct comparison may be invalid |
|---|---|---|---|---|---|
| — | **IntelliCredit Method 1** | Simulated monthly portfolio default rate, 72 months (2019–2024), driven by real FRED and World Bank macro data | SARIMAX + macro drivers | 2024 hold-out **MAE 0.054 pp, MAPE 16%**; 40-origin backtest mean MAE 0.124 pp; 80% PI covers 92% of 2024 actuals. Beats seasonal naive (MAE 0.124), SARIMA (0.088), ETS (0.075), gradient boosting (0.065) | The target is simulated from a macro-linked hazard. The forecaster recovering that link validates the pipeline only. |
| [36] | Makridakis et al. 2018, PLOS ONE | 1,045 monthly M3 series | 8 ML vs 8 statistical methods | **ML dominated by statistical methods on both accuracy measures at all horizons** | General-purpose series, no exogenous drivers. Supports the choice of method, not our error level. |
| [37] | Makridakis et al. 2018, IJF (M4) | 100,000 series, 61 methods | Competition | Winner (ETS–RNN hybrid) about **10% better sMAPE** than the combination benchmark. **None of 6 pure ML methods beat the combination; only 1 beat Naïve2** | Same caveat. Note that our seasonal-naive baseline is the M4-style sanity check. |
| [33] | Gibbs & Candès 2021, NeurIPS | Stock volatility; election-night data | Adaptive Conformal Inference | Long-run miscoverage bound: \|avg err − α\| ≤ (max{α₁,1−α₁}+γ)/(γT). Non-adaptive intervals drift from 90%; ACI stays close (no single coverage number reported in the text) | Use this for the FRED delinquency track. Report *our* empirical coverage, because the bound is loose at T ≈ 140. |
| [34] | Xu & Xie 2021, ICML (EnbPI) | Simulations + real data | Ensemble bootstrap PIs | Coverage-gap bounds; no exchangeability needed | Alternative to ACI. Compare interval width at equal coverage. |

## Table 3: Real-data anchors (for fidelity and sanity checks of the synthetic data)

| Quantity | Real value | Source | Synthetic or model value | Comment |
|---|---|---|---|---|
| India bank NPL ratio | 9.98% (2017 peak) → 2.06% (2025) | World Bank FB.AST.NPER.ZS [70] | — | Target for the real NPL-panel model (track a) |
| SCB GNPA | 8.2% (Mar 2020) → 7.3% (Mar 2021) | RBI T&P 2020-21 [63] | Simulated COVID default spike ×2.7 | **Real bank NPLs fell during COVID** because of the moratorium, restructuring and write-offs. The spike is an assumption, not a benchmark. |
| SCB GNPA | 1.8% (Mar 2026) | RBI FSR Jun 2026 [61] | — | Context |
| Microfinance PAR 31–180 | 2.2% (Mar 2024), 6.3% (Mar 2025), 1.6% (Jun 2026) | MFIN Micrometer [67] | Default rate 7.87% | Ours is deliberately higher (thin-file segment). The 2025 MFI stress episode shows real delinquency can triple within a year. |
| US credit-card delinquency / charge-off | 2.85% / 3.82% (Q2 2026) | FRED / Fed Board [71] | — | Target series for conformal forecasting (track b) |
| US consumer-loan delinquency | 2.62% (Q2 2026) | FRED DRCLACBS [71] | — | Second target series |
| Gig worker income | about 40% earn < ₹15,000/month | Economic Survey 2025-26 [65] | Synthetic range ₹14k–48k | A fidelity check: compute the share of synthetic gig borrowers below ₹15k and compare with about 40%. |

## How to use these tables honestly

1. Put **IntelliCredit's row first** and the caveat column next to every external number.
2. **Never** claim we "beat" Berg [1] or Óskarsdóttir [3]: different data, different label, different product.
3. The fair claims are these. Our model ranking (XGB ≥ LGBM > LR > RF) agrees with [8], [9] and [10]. Our AUC level is in the range real alternative-data studies report ([1], [4]). Our forecasting choice agrees with [36] and [37].
4. Real-data results from the new track (World Bank NPL panel, FRED with ACI, HMDA ARM) should get their own rows when they are ready. Those are the numbers that answer "where is the real data?".
