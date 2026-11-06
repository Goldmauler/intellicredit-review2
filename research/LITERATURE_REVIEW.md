# IntelliCredit: Literature Review (Review 2 evidence pack)

Compiled 2026-11-06 by SCOUT. There are 72 sources in total, numbered as in `REFERENCES_ACM.md`, with BibTeX in `references.bib`.

**Verification rule.** Every source below was checked against a primary record: a Crossref DOI record, an arXiv abstract page, a proceedings page, or an official regulator or government page.
- **[V]**: the primary page was opened.
- **[S]**: confirmed only through search excerpts of the primary document or reputable secondary reporting. Re-check [S] numbers in the original PDF before putting them on a slide.
- Numbers I could not verify are written as "not verified" rather than guessed.

**Entry format:** reference | link | dataset | method | **key numeric finding** | how it supports IntelliCredit, and what a reviewer might ask.

---

## a. Alternative and digital-footprint credit scoring for thin-file borrowers

**[1] Berg, Burg, Gombović & Puri (2020), RFS 33(7):2845–2897** [V]
- Link: https://doi.org/10.1093/rfs/hhz099. Open WP: NBER 24551; FDIC CFR WP 2018-04.
- Data: purchases at a German e-commerce company where goods ship before payment, Oct 2015 – Dec 2016. Slides give about 270,000 purchases; the abstract says more than 250,000 observations.
- Method: logistic scorecards on simple digital-footprint variables (data left just by visiting or registering on a website; the FDIC write-up highlights device or smartphone type), compared with the credit-bureau score. The exact variable list was not re-verified.
- **Key numbers:** AUC is 68.3% for the bureau score alone, **69.6% for the digital footprint alone**, and **73.6% for both combined** (+5.3 pp). Footprint discrimination for "unscorable" customers is similar to that for scored customers.
- IntelliCredit link: this is the canonical evidence that non-bureau behavioural data carries default signal comparable to a bureau score. Our XGBoost AUC of 0.738 sits in the same range. *Reviewer Q: "Is 0.74 good?"* Yes: Berg et al., citing [7], treat ~60% AUC as desirable in information-scarce settings and 70%+ as the goal in information-rich ones.

**[2] Björkegren & Grissen (2020), World Bank Economic Review 34(3):618–634** [V]
- Link: https://doi.org/10.1093/wber/lhz006. Open: arXiv:1712.05840.
- Data: call-detail records matched to repayment for credit extended by a South American telecom.
- Method: ML on phone-usage behaviour features, such as usage regularity and balance management.
- **Key numbers:** borrowers in the top risk quintile are **2.8× more likely to default** than those in the bottom quintile. On thin-file borrowers the model beats bureau-based models both in-period and out-of-period, and it performs similarly for people with no financial history. The exact AUC was not verified: one secondary source quotes 0.76 from an early draft.
- IntelliCredit link: this supports scoring the "no-file" segment from behavioural data. Note that India's RBI Digital Lending Directions 2025 forbid lending apps from reading call logs and contacts [60], so the Indian equivalent has to rely on consented UPI, utility and recharge data, as IntelliCredit does.

**[3] Óskarsdóttir, Bravo, Sarraute, Vanthienen & Baesens (2019), Applied Soft Computing 74:26–39** [V]
- Link: https://doi.org/10.1016/j.asoc.2018.10.004. Open: arXiv:2002.09931.
- Data: a bank plus a telco in the same (unnamed) country. The bank has over 2 million customers and the CDRs cover about 90 million phone numbers. The modelling set is **22,000 credit-card applicants, default rate 4.49%**.
- Method: LR, decision trees and random forests on socio-demographic (SD), calling-behaviour (CB) and social-network features.
- **Key numbers (Table 4, AUC):**

  | Features | LR | DT | RF |
  |---|---|---|---|
  | SD only | 0.5869 | 0.7004 | 0.8993 |
  | CB only | 0.5351 | 0.7043 | 0.8700 |
  | SD+CB | 0.6115 | 0.7127 | **0.9227** |

  In profit terms, the best model used calling-behaviour features only.
- IntelliCredit link: alternative data adds measurable AUC over traditional data. *Reviewer Q: "Why is their RF 0.92 and yours 0.74?"* Their label is credit-card default in a bank's book with 300+ features. Our label is drawn from a known PD, which caps achievable AUC at 0.781 (see BENCHMARK_TABLE.md).

**[4] Agarwal, Alok, Ghosh & Gupta (2020 WP; ISB lists JMCB 2022)** [S]
- Link: JBS-hosted PDF (see refs).
- Data: loan applications to a large **Indian fintech lender**, with phone footprint data (apps installed, call logs, social connectivity).
- Method: logit and ML default and approval models.
- **Key numbers:** AUC is **59%** for the credit score alone (58.6% in the Oct 2019 draft) and **60.4%** for the mobile and social footprint alone. The authors estimate about **129 million** people in India have mostly been denied credit for lack of a credit history. The numbers differ between versions.
- IntelliCredit link: this is the closest India-specific evidence that alternative data substitutes for CIBIL on thin files. Indian fintech AUCs are low (~0.6), which makes our 0.74 on synthetic data look optimistic. Say so openly.

**[5] Alok, Ghosh, Kulkarni & Puri (2024), NBER WP 33259** [V]
- Link: https://econpapers.repec.org/RePEc:nbr:nberwo:33259
- Data: India's UPI rollout, using data covering the universe of consumer loans.
- Method: natural-experiment, difference-in-differences style design.
- **Key finding:** UPI expanded formal credit to new-to-credit borrowers. **Fintechs lent across all segments and were the only lenders reaching no-credit-history customers.** There was **no evidence of higher default rates**. Growth was strongest where internet was cheap and bank accounts had recently expanded. The abstract gives no numeric effect size.
- IntelliCredit link: this directly justifies UPI-derived features and the gig and thin-file target segment in India.

**[6] Ghosh, Vallee & Zeng (2025), NBER WP 34148** [V]
- Link: https://econpapers.repec.org/RePEc:nbr:nberwo:34148
- Finding: cashless-payment usage goes with **better FinTech credit access and lower predicted default**. The effect is stronger for more informative payment types and for outflows, and it complements other borrower-quality signals. The abstract does not say which country is studied.
- IntelliCredit link: supports using UPI outflow patterns (bill payments, recharge) as features.

**[7] Iyer, Khwaja, Luttmer & Shue (2016), Management Science 62(6):1554–1577** [V]
- Link: https://doi.org/10.1287/mnsc.2015.2181
- Data: peer-to-peer lending, where lenders screen small borrowers using soft information.
- Role: [1] cites it for the AUC norm of about 60% as desirable in information-scarce settings and 70%+ as the goal in information-rich ones.
- IntelliCredit link: use this to frame what a "good" AUC is in Section 6.2.

---

## b. ML credit-scoring benchmarks (for Section 6.2 comparison)

**[8] Lessmann, Baesens, Seow & Thomas (2015), EJOR 247(1):124–136** [V]
- Data: 8 real-world credit-scoring datasets. Method: benchmark of 41 classifiers on several metrics; a secondary summary lists AUC, partial AUC and H-measure.
- **Key finding:** several classifiers beat logistic regression significantly, and heterogeneous ensembles rank best. The authors recommend random forest as the minimum benchmark for new methods. *Per-dataset AUC values were not verified, because the PDF text could not be extracted.*
- IntelliCredit link: justifies comparing LR, RF, XGB and LightGBM and reporting more than one metric. *Reviewer Q: "Why not stacking?"* We did stack (M4); Lessmann et al. support ensembles.

**[9] Gunnarsson, vanden Broucke, Baesens, Óskarsdóttir & Lemahieu (2021), EJOR 295(1):292–305** [V]
- Method: a multilayer perceptron and a deep belief network compared with two conventional methods and two ensemble methods (including XGBoost) on several real-life credit datasets, with Bayesian and non-parametric tests.
- **Key finding:** **XGBoost performed best overall; deep networks did not beat shallower models** and cost far more to compute. The authors recommend XGBoost when classification performance is the main goal. Numeric AUCs were not verified.
- IntelliCredit link: this backs our choice of XGBoost (0.738) and LightGBM, and our decision not to use deep learning for tabular PD.

**[10] Dastile, Celik & Potsane (2020), Applied Soft Computing 91:106263** [V]
- Systematic review of **74 primary studies (2010–2018)**. **Ensembles generally beat single classifiers.** Deep learning was rare but promising, and explainability is the barrier to adopting it.
- IntelliCredit link: literature-level backing for ensembles plus SHAP.

**[11] Yeh & Lien (2009), Expert Systems with Applications 36(2):2473–2480** [V]
- Data: **30,000 Taiwanese credit-card clients** (the UCI "default of credit card clients" dataset, 23 features). Six data-mining methods compared.
- Finding: only the artificial neural network estimated the *real* probability of default accurately. The study focuses on calibration of PD, not only ranking.
- IntelliCredit link: same n = 30,000. It supports our separate calibration step (calibrated LightGBM for the dashboard), because ranking (AUC) and calibrated PD are different targets.

**[12] Baesens, Setiono, Mues & Vanthienen (2003), Management Science 49(3):312–329** [V]
- Method: extracts rules from neural networks and puts them into decision tables for credit-risk evaluation.
- Relevance: an early high-profile argument that credit decisions need human-readable rules. This supports Method 2 (association rules) and the warning-rule engine.

---

## c. Why synthetic data is legitimate in finance, and its limits

**[13] Assefa et al. (2020), ICAIF '20 (J.P. Morgan AI Research), pp. 1–8** [V]
- Link: https://doi.org/10.1145/3383455.3422554
- Content: describes the need for synthetic financial data because privacy and regulation restrict sharing, lists three research directions (realistic generation, measuring real-versus-synthetic similarity, privacy guarantees), and discusses pitfalls. It is a position paper with no benchmark numbers.
- IntelliCredit link: the main citation for "major banks use synthetic data because real borrower data cannot be shared".

**[14] FCA Synthetic Data Expert Group (Mar 2024), "Using Synthetic Data in Financial Services"** [V]
- The group was set up in 2023. The report covers three uses: data augmentation and bias mitigation, model testing and validation, and data sharing for fraud controls. The report states it is **not regulatory guidance**.
- IntelliCredit link: shows a regulator treating synthetic data as a legitimate tool for model development.

**[15] FCA SDEG (Aug 2025), "Generating and Using Synthetic Data for Models in Financial Services: Governance Considerations"** [S]
- For model validation, it suggests statistical comparison and performance benchmarking, including **Train-Synthetic-Test-Real (TSTR)**. It is not guidance.
- IntelliCredit link: names TSTR, the fidelity check we are adding.

**[16] Esteban, Hyland & Rätsch (2017), arXiv:1706.02633** [V]
- Method: recurrent conditional GANs for medical time series, evaluated by training on synthetic data and testing on real data, and the reverse. This is the TSTR idea, although the abstract does not use the acronym.
- **Key number:** an early-warning model trained on synthetic data generated from **17,000 ICU patients** showed only "minor degradation" on real test data.
- IntelliCredit link: the origin of the TSTR protocol used in our fidelity checks.

**[17] Patki, Wedge & Veeramachaneni (2016), IEEE DSAA, pp. 399–410** [V]
- The Synthetic Data Vault: a system for generating synthetic relational databases. It is the basis of the SDV and SDMetrics libraries.
- IntelliCredit link: gives a citable tool for fidelity metrics (column shapes, pair trends).

**[18] Xu, Skoularidou, Cuesta-Infante & Veeramachaneni (2019), NeurIPS (CTGAN), arXiv:1907.00503** [V]
- Benchmark of **7 simulated and 8 real datasets**. The conditional GAN beat Bayesian-network baselines on most real datasets.
- IntelliCredit link: our generator is rule-based (NumPy and Faker), not GAN-based. *Reviewer Q: "Why not CTGAN?"* There is no real Indian borrower table to learn from. CTGAN needs real seed data, so the honest route is a rule-based generator plus validation against published aggregates.

**[19] Jordon et al. (2022), "Synthetic Data – what, why and how?", Royal Society, arXiv:2205.03257** [V]
- A 57-page explainer. Synthetic data is "a very useful tool", but privacy and utility trade-offs are easy to miss.
- IntelliCredit link: a balanced source for the Limitations section.

**[20] Digital Personal Data Protection Act, 2023 (Act 22 of 2023)** [S]
- Section 2(t): "personal data" means any data about an individual who is identifiable by or in relation to such data.
- Section 17(2)(b): exemption for processing "necessary for research, archiving or statistical purposes" if the data is not used to take any decision specific to a Data Principal and follows prescribed standards.
- IntelliCredit link: our synthetic borrowers are not data about identifiable individuals, and our FRED and World Bank series are aggregates. See REVIEWER_ANSWERS Q7.

**[21] DPDP Rules, 2025 (G.S.R. 846(E), 13 Nov 2025)** [S]
- Phased commencement. Data Protection Board provisions applied from notification. Most obligations, including Rule 16 on the research exemption standards, start **18 months later (about 13–14 May 2027)**.

**[22] Credit Information Companies (Regulation) Act, 2005 (Act 30 of 2005)** [S]
- Registration with RBI is required to run a credit bureau (CIC). The RBI Master Direction lists four CICs: TransUnion CIBIL, Equifax, Experian and CRIF High Mark. Section 15 requires every credit institution to be a member of at least one CIC.
- IntelliCredit link: IntelliCredit is a lender-side scorecard, not a CIC, so it does not need CIC registration (REVIEWER_ANSWERS Q6).

---

## d. Macroeconomic determinants of NPLs and default rates (backs SARIMAX exogenous drivers)

**[23] Beck, Jakubík & Piloiu (2015), Open Economies Review 26(3):525–550; ECB WP 1515 (2013)** [V]
- Data: a **panel of 75 countries** over about a decade. Method: dynamic panel GMM.
- **Key finding:** NPL ratios are significantly driven by **real GDP growth, share prices, the exchange rate and the lending interest rate**. The exchange-rate effect depends on unhedged foreign-currency lending, and the share-price effect is larger where stock markets are big relative to GDP. Coefficient sizes were not verified.
- IntelliCredit link: our SARIMAX drivers (GDP growth, call-money and lending rates, INR/USD, unemployment, CPI) match this literature. It also directly motivates the real-data track: a World Bank NPL panel (FB.AST.NPER.ZS) with the same drivers.

**[24] Nkusu (2011), IMF WP/11/161** [S]
- Data: panel of advanced economies through the 2008 crisis. Method: panel regressions plus panel VAR.
- Finding: adverse macro developments (unemployment, house prices) raise NPLs, and a sharp NPL increase has long-lasting negative effects on the macroeconomy. The feedback runs both ways.
- IntelliCredit link: supports treating unemployment as a main driver. Our model has +1 pp unemployment giving +21% defaults.

**[25] Louzis, Vouldis & Metaxas (2012), J. Banking & Finance 36(4):1012–1027** [V]
- Data: Greek banks' mortgage, business and consumer loan portfolios, compared side by side (per the title). Method: dynamic panel.
- Finding: NPLs are explained by both macroeconomic and bank-specific determinants, and the portfolios are compared separately. Coefficients and the exact variable list were not verified in this pass.
- IntelliCredit link: a published consumer-loan NPL model; a reason to model consumer and micro-loans separately from corporate credit.

**[26] Lokare (2014), RBI WPS (DEPR) 03/2014** [V]
- Data: Indian banks, quarterly 2001–2013.
- Finding: **credit is pro-cyclical in India**, tied to boom-bust credit episodes and to economic and interest-rate cycles. A weak external environment, high inflation and falling asset prices reduced borrowers' debt-servicing capacity. Retail and agriculture defaults rose.
- IntelliCredit link: the Indian central bank's own evidence that inflation and interest rates drive defaults, which is exactly our SARIMAX exogenous set.

**[27] Ranjan & Dhal (2003), RBI Occasional Papers 24(3)** [S]
- Panel of Indian public-sector banks. Terms of credit, bank size and macro shocks all significantly affect NPLs.

---

## e. Time-series forecasting and uncertainty

**[28] Hyndman & Athanasopoulos (2021), FPP3, OTexts** [V]: the standard reference for ARIMA, dynamic regression (regression with ARIMA errors, which is what SARIMAX fits), ETS, time-series cross-validation and prediction intervals. It is our methodological reference for the rolling-origin backtest (40 origins).

**[29] Box, Jenkins, Reinsel & Ljung (2015), 5th ed., Wiley** [S]: Box-Jenkins identification, estimation and diagnostic checking, including Ljung-Box residual tests (our residual p = 0.12 / 0.28).

**[30] Granger (1969), Econometrica 37(3):424–438** [V]: Granger causality. Justifies our Granger tests of macro series against the default rate. *Reviewer Q: "Granger is not causality."* Correct: it measures predictive precedence only.

**[31] Vovk, Gammerman & Shafer (2005), Springer** [V]: the foundational text for conformal prediction. It gives finite-sample, distribution-free coverage under exchangeability.

**[32] Angelopoulos & Bates (2021), arXiv:2107.07511** [V]: a tutorial on split conformal prediction. Its intervals cover the truth with a user-chosen probability (e.g. 90%), with **distribution-free, non-asymptotic guarantees**, and can wrap any model. It includes time-series and distribution-shift extensions.

**[33] Gibbs & Candès (2021), NeurIPS 34 (Adaptive Conformal Inference)** [V]
- Update rule: **α_{t+1} = α_t + γ(α − err_t)**. The experiments use γ = 0.005.
- **Guarantee (Prop. 4.1):** with probability one, |(1/T)Σ err_t − α| ≤ (max{α₁, 1−α₁} + γ)/(γT), for any data-generating process, including adversarial shifts. Long-run miscoverage converges to α.
- Experiments: stock-volatility and election-night data. Non-adaptive conformal intervals drift away from 90% coverage, while ACI stays close.
- IntelliCredit link: **the core of our innovation.** It gives FRED delinquency and charge-off intervals that stay calibrated through regime shifts such as COVID. With T ≈ 140 quarters (about 1991–2026; check the FRED start date), γ = 0.005 and α₁ = 0.1, the bound is (0.9 + 0.005)/(0.005×140) ≈ 1.29, so the guarantee says nothing at our sample length. Use a larger γ (e.g. 0.05 gives ≈ 0.14) and report empirical coverage. *Reviewer Q: "Is the guarantee meaningful at T ≈ 140?"* Answer honestly that the bound is loose. The value of ACI is in the empirical coverage under shift.

**[34] Xu & Xie (2021), ICML (EnbPI), arXiv:2010.09107** [V]: ensemble batch prediction intervals for time series. It **does not require exchangeability** and needs no data splitting or refitting. It was a long oral at ICML 2021 (about 3% of 5,513 submissions). It is an alternative to ACI for our forecast intervals.

**[35] Stankeviciute, Alaa & van der Schaar (2021), NeurIPS 34** [V]: conformal intervals for multi-horizon RNN forecasts. It works with any multi-horizon predictor under mild exchangeability across series. It is relevant if we forecast several horizons (1–8 quarters ahead).

**[36] Makridakis, Spiliotis & Assimakopoulos (2018), PLOS ONE 13(3):e0194889** [V]
- Data: **1,045 monthly M3 series**. **Eight popular ML methods were beaten by eight statistical methods on both accuracy measures at all horizons**, at much higher computational cost.
- IntelliCredit link: the main citation for "why SARIMAX, not LSTM".

**[37] Makridakis, Spiliotis & Assimakopoulos (2018), IJF 34(4):802–808 (M4 results)** [S for the numbers]
- **100,000 series, 61 methods.** The winner (Smyl's hybrid of exponential smoothing and an RNN) was **about 10% more accurate (sMAPE) than the combination benchmark**. Of the six **pure ML methods, none beat the combination benchmark and only one beat Naïve2**.
- IntelliCredit link: pure deep learning does not win on short economic series. We have 72 monthly portfolio observations and about 140 quarterly FRED observations.

---

## f. Association rule mining

**[38] Agrawal & Srikant (1994), VLDB, pp. 487–499** [S]: the Apriori algorithm. Support and confidence are the basis of Method 2.

**[39] Han, Pei & Yin (2000), SIGMOD, pp. 1–12** [V]: FP-growth, which mines frequent patterns without candidate generation using the FP-tree. It explains our Apriori-versus-FP-Growth runtime comparison (same itemsets, faster).

**[40] Webb (2007), Machine Learning 68(1):1–33** [V]: "Discovering Significant Patterns". Large rule searches produce many false discoveries, so rules should be tested on holdout data or with multiple-testing correction.
- IntelliCredit link: **this is the paper behind our Bonferroni-corrected Fisher tests and our 93 rules validated on the held-out test set.** Cite it when a reviewer asks "Aren't some of your rules just chance?"

**[41] Zhang (2011), ICEIS, pp. 129–134** [V]: builds a personal-loan credit index and uses Apriori to extract risk-warning rules for a commercial bank. The author concludes Apriori "plays an important role in identifying risk in personal loans". The abstract gives no numbers.

**[42] Tanantong & Loetwiphut (2023), SEHS 17:23020006** [V]: an ARM framework with feature selection, discretisation and SMOTE for P2P-lending credit risk. Rules flag creditworthiness patterns. The abstract gives no numbers.
- IntelliCredit link for [41] and [42]: prior use of ARM for credit risk exists. Our contribution adds significance testing, out-of-sample lift validation, and interaction with macro stress.

---

## g. Explainability and recourse

**[43] Lundberg & Lee (2017), NIPS 30, pp. 4765–4774** [V]: SHAP gives additive, Shapley-based feature attributions and unifies six earlier attribution methods. It is our TreeSHAP waterfall charts in the dashboard.

**[44] Wachter, Mittelstadt & Russell (2018), Harvard JOLT, arXiv:1711.00399** [V]: counterfactual explanations ("you would have been approved if X were Y") as a way to explain automated decisions without opening the model, framed around the GDPR. This is the conceptual basis of our "path to approval" feature.

**[45] Ustun, Spangher & Liu (2019), FAT\* '19, pp. 10–19** [V]: actionable recourse. It uses integer programming to find the cheapest feasible changes to features the applicant can actually act on, so immutable features such as age are excluded. It audits whether recourse exists at all. In our project, mark gender, age and city tier as immutable.

**[46] Mothilal, Sharma & Tan (2020), FAT\* '20, pp. 607–617 (DiCE)** [V]: diverse counterfactuals that trade off proximity and diversity. The `dice-ml` library is the implementation path.

**[47] Karimi, Barthe, Schölkopf & Valera (2020), arXiv:2010.04050** [V]: a survey of algorithmic recourse. It separates counterfactual *explanations* from causal *recourse* and warns that changing one feature can affect others. Useful as a limitation to state.

**[48] Regulation B, 12 CFR §1002.9** [V]: a US creditor must act within **30 days** of a completed application. An adverse-action notice must give a **"statement of specific reasons"**, and the reasons "must be specific and indicate the principal reason(s)". Saying only that the applicant failed internal standards or missed a qualifying score is not enough.

**[49] CFPB Circular 2022-03** [S]: says these reason requirements apply even to "black-box" models. It was **withdrawn on 12 May 2025** among 67 guidance documents, but the underlying ECOA and Reg B duty in [48] still applies.
- IntelliCredit link for [48] and [49]: SHAP reasons plus counterfactual recourse map directly onto an adverse-action reason notice.

---

## h. Fair lending with HMDA

**[50] Bartlett, Morse, Stanton & Wallace (2022), JFE 143(1):30–56** [V]
- Data: US GSE-securitised and FHA-insured mortgages.
- **Key numbers (published abstract):** risk-equivalent Latinx and Black borrowers pay significantly higher rates, which costs minority borrowers **over $450 million per year**. FinTech rate disparities were similar to those of other lenders for GSE loans, but lower for FHA loans in 2009–2015 and for FHA refinances in 2018–2019. The NBER WP version reported different figures (6–9 bps, ~$750M), so **cite the JFE number**.
- IntelliCredit link: the template for the HMDA fairness module. Compare denial and pricing gaps across `derived_race` and `derived_ethnicity` after controlling for credit variables.

**[51] Fuster, Goldsmith-Pinkham, Ramadorai & Walther (2022), JF 77(1):5–47** [V]
- Data: US mortgages, 9.4 million loans from 2009–2013 according to the authors' 2020 slides. Default is defined as 90+ days delinquent within 3 years.
- **Key finding:** a random forest predicts default better than logit. The slides show ROC AUC of about **0.748 for logit versus 0.753 for RF** [S], version-specific. **Black and Hispanic borrowers are less likely to gain** from ML, and ML widens rate disparities between and within groups.
- IntelliCredit link: a better AUC does not mean fairer. This justifies reporting fairness metrics alongside the AUC table.

**[52] FFIEC/CFPB Public HMDA LAR data fields** [V]
- `action_taken` codes: 1 originated, 2 approved not accepted, **3 denied**, 4 withdrawn, 5 incomplete, 6 purchased, 7 preapproval denied, 8 preapproval approved not accepted.
- `denial_reason-1` codes: 1 DTI, 2 employment history, 3 credit history, 4 collateral, 5 insufficient cash, 6 unverifiable info, 7 incomplete application, 8 mortgage insurance denied, 9 other, 10 not applicable. `denial_reason-2` to `-4` use codes 1–9.
- Other fields: `income` (in $000s), `debt_to_income_ratio` (binned: <20%, 20–<30%, 30–<36%, each whole percent 36–49, 50–60%, >60%), `loan_amount`, `interest_rate`, `combined_loan_to_value_ratio`, `derived_race`, `derived_ethnicity`, `derived_sex`.
- IntelliCredit link: build ARM items such as {DTI 50–60%, low income, credit-history denial reason} → denied. Reproduce Method 2 on **real** underwriting decisions.

---

## i. Consumer-complaint text mining

**[53] Blei, Ng & Jordan (2003), JMLR 3:993–1022** [V]: LDA, the generative topic model behind most complaint-mining work.

**[54] Grootendorst (2022), arXiv:2203.05794** [V]: BERTopic combines transformer embeddings, clustering and class-based TF-IDF. The author reports competitive coherence against classical models. Use it as a modern comparison to LDA.

**[55] Bastani, Namavari & Shaffer (2019), ESWA 127:256–271** [V]: LDA on **CFPB complaint narratives**, tracking topic trends over time to judge whether CFPB regulation is working. The authors chose LDA over LSA and pLSA. This is the direct precedent for our module (d).

**[56] Vaishnav et al. (2024), arXiv:2407.06399** [V]: ML and LDA on the CFPB database (described as more than 4.9 GB) to predict company response time and type, for example "closed with relief". The abstract gives no accuracy numbers.

**[57] CFPB Consumer Complaint Database** [V]
- Updates generally daily. A complaint is published after the company responds or after 15 days, whichever comes first.
- It is **"not a statistical sample of consumers' experiences"**, and complaints about depository institutions with under $10B in assets are not published.
- CSV, ZIP and API are available.
- IntelliCredit link: always state the "not a statistical sample" caveat when presenting complaint topic shares.

**[58] CFPB (2015), narrative publication policy** [S]: narratives are published only with the consumer's **opt-in consent**. Personal information is scrubbed in three steps (automated, a trained reviewer, then QA). The CFPB itself says this minimises, but does not eliminate, re-identification risk.

---

## j. India context: exact figures, publication and date

| # | Figure | Value | Publication / date | Tag |
|---|---|---|---|---|
| [70] | India bank NPLs, % of gross loans (World Bank FB.AST.NPER.ZS) | **2017: 9.98%** (peak); 2018: 9.46%; 2019: 9.23%; 2020: 7.94%; 2021: 6.54%; 2022: 4.81%; 2023: 3.36%; 2024: 2.50%; **2025: 2.06%** | WDI, last updated 2026-10-08 | [V] |
| [61] | SCB GNPA ratio | **1.8% at end-March 2026** ("multi-decadal low"). Baseline stress test: 1.9% by March 2028; 3.8% / 4.1% under medium / severe stress. NBFCs (174 entities): 2.4%. UCBs: 5.2% | RBI FSR June 2026 (released 30 Jun 2026), figures via PSU Watch | [S] |
| [62] | SCB GNPA / NNPA / CRAR | **2.8% / 0.6% / 16.8% at end-March 2024** | RBI FSR June 2024 press release, 27 Jun 2024 | [V] |
| [63] | SCB GNPA | **8.2% (Mar 2020) → 7.3% (Mar 2021) → 6.9% (Sep 2021)**. Write-offs were the main reason for the fall | RBI Trend & Progress 2020-21, 28 Dec 2021 | [S] |
| [64] | Gig workers | **77 lakh (7.7M) in 2020-21** = 2.6% of the non-farm workforce and 1.5% of the total; projected **2.35 crore (23.5M) by 2029-30** | NITI Aayog, 27 Jun 2022 | [S] |
| [65] | Gig workers | 7.7M (FY21) → **12M (FY25)**. About **40% earn below ₹15,000/month**. The Survey notes many gig workers have **"thin-file" credit histories** that restrict access to formal finance | Economic Survey 2025-26, 29 Jan 2026 | [S] |
| [66] | UPI | **25.47 bn transactions, ₹31.53 lakh crore in October 2026**. Daily average 822M transactions, ₹1,01,707 crore. Volume +23% YoY, value +16% YoY | NPCI data, 1 Nov 2026 (via Entrepreneur India) | [S] |
| [67] | Microfinance portfolio quality (MFIN PAR 31–180) | **1.6% at 30 Jun 2026** (2.1% Mar 2026; 5.6% Jun 2025; 6.3% Mar 2025; 2.2% Mar 2024). Portfolio ₹3,28,708 crore, −6.9% YoY | MFIN Micrometer Q1 FY27, 1 Sep 2026 | [S] |
| [67] | Microfinance portfolio size, Mar 2024 | ₹4,33,697 crore; 7.8 crore unique borrowers; 14.9 crore loan accounts | MFIN Micrometer Q4 FY24 | [S] |
| [68] | Account Aggregator | **2.88 bn accounts enabled; 284.6M linked by users**; 179 FIPs and 989 FIUs live | DFS, MoF, as of 31 Mar 2026 | [V] |
| [69] | Unemployment (PLFS, CWS, age 15+) | **5.0% in Sep 2026**; LFPR 55.5% | PLFS monthly bulletin, 15 Oct 2026 | [S] |
| [72] | Unbanked adults worldwide | **1.4 bn (Findex 2021)**; about 1.3 bn in Findex 2025 | World Bank | [S] |
| [59]/[60] | Digital-lending rules | 2022 Guidelines (2 Sep 2022), replaced by the **Digital Lending Directions 2025 (8 May 2025)**: need-based data collection with prior explicit consent; **no access to files, media, contacts, call logs or telephony**; one-time camera, mic and location access allowed for onboarding; **all data stored on servers in India** (para 13(iv)); economic profile (age, occupation, income) must be assessed before lending (para 7(i)); KFS required; cooling-off exit option | RBI | [V] |
| [71] | US real default series (FRED, Fed Board) | Q2 2026: credit-card delinquency **2.85%** (DRCCLACBS), credit-card charge-off **3.82%** annualised net (CORCCACBS), consumer-loan delinquency **2.62%** (DRCLACBS). All quarterly and seasonally adjusted | Fed release, updated 25 Aug 2026 | [V] |

---

## Corrections needed in existing repo documents

I checked these claims in `README.md`, `reports/Review_2_Final_Report.md` and `data/DATASET_VALIDATION_SUMMARY.md`. **Fix them before tomorrow.** A reviewer who checks any one of them will distrust the whole validation section.

1. **"Over 1.4 billion thin-file individuals and gig-economy workers"** (README, validation summary). 1.4 bn is the World Bank Findex 2021 count of **unbanked adults worldwide** [72], not gig workers. India has about 7.7M (2020-21, NITI [64]) to 12M (FY25, Economic Survey [65]) gig workers. Suggested wording: "1.4 billion adults worldwide were unbanked (Findex 2021); India's 12 million gig workers often have thin-file credit histories (Economic Survey 2025-26)."
2. **"CGAP India Gig Economy Financial Diary 2021"** (income CoV 35–60%). **No such publication could be found.** Remove it, or replace it with the Economic Survey 2025-26 figure that about 40% of gig workers earn under ₹15k/month [65].
3. **"MFIN Micrometer: PAR 90+ 1.5–3.5% (FY2023–24)"**. MFIN headlines **PAR 31–180**, not PAR 90+. The verified values are 2.2% (Mar 2024), **6.3% (Mar 2025, a stress peak)** and 1.6% (Jun 2026) [67]. The range quoted was not found.
4. **"RBI Trend & Progress 2020-21 reports a 3×–5× spike in NBFC-MFI delinquencies"**. **Not found.** That report says the SCB GNPA ratio *fell* from 8.2% to 7.3% during the COVID year [63], helped by the moratorium, restructuring and write-offs. Our 2.7× simulated COVID multiplier therefore has no RBI benchmark. Present it as an assumption.
5. **"PLFS 2022-23 gig income ₹12k–50k/month"**. PLFS reports employment and unemployment status; I could not verify a platform-worker income band in PLFS. Use [65] instead.
6. **"NPCI: ~350M active UPI users / ~500M adults = ~70%"** and **"MFIN average ticket ₹42k–68k"** were not verified. Either find the exact NPCI or MFIN table, or drop them.
7. **"Default rate 7.87% perfectly matching industry benchmarks"**. No benchmark above matches 7.87%. Phrase it as "deliberately higher than MFIN PAR 31–180 (1.6–6.3%, 2024–26) because the segment is thin-file and higher-risk".
