# README Defence Q&A

159 questions and answers taken from `README.md` (as of 10 Nov 2026), section by section. Say your answer aloud first, then compare. **Trap** lines mark where an examiner will push and where the README itself admits a weakness. Difficulty: Basic, Probing, Hard. An interactive version with search and a self-check list is the "README Defence Q&A" artifact.

## 1 · Problem statement and why now

*README: "Problem Statement" and "Why this matters now"*

**Q1.1 (Basic) In one sentence, what problem does IntelliCredit solve?**

It lets a lender decide whether to give a small loan to a gig worker who has no credit history, and it does so in a way that can be explained: a chance of default, a money-based Approve / Approve with conditions / Decline decision, the reasons, and what the applicant could change.

**Q1.2 (Basic) Why do gig workers have "thin or no credit files"?**

Credit bureau scores (CIBIL, FICO) are built from past loans and cards. Many gig workers have never borrowed formally and have no payslip, so a lender has nothing to score. The README says they "mostly" have thin or no files; it does not give an exact share.

> **Trap:** Do not quote a percentage for "no credit file". The README does not give one.

**Q1.3 (Basic) Where does "7.7 to 12 million gig workers" come from? Why a range?**

Two sources. NITI Aayog (2022) estimated 7.7 million gig workers in 2020-21. The Economic Survey 2025-26 reports about 12 million in FY25, up 55% from FY21. The range is simply two different years from two sources.

*Evidence: research/REFERENCES_ACM.md [64], [65]*

**Q1.4 (Probing) What does the Economic Survey say about their pay, and how sure are you of it?**

About 40% of gig workers earn under ₹15,000 a month, and gig workers are over 2% of the workforce. We read this in a news report of the Survey (Business Today). We could not locate the Survey chapter itself, so it is a secondary-source figure.

> **Trap:** Say "reported by Business Today". Do not say you read the Survey chapter.

**Q1.5 (Probing) Why is the MFIN news item in a credit-scoring project?**

MFIN's CEO said microfinance is "past the peak" of its stress, with loans 1 to 180 days overdue at 4.7% in January 2026 and liquidity still tight. It shows low-income delinquency can swing sharply within a year. That is exactly what Method 1 (the time-series forecast with economy scenarios) is built to anticipate.

**Q1.6 (Basic) Why is a UPI record relevant?**

UPI hit 25.47 billion transactions worth ₹31.53 lakh crore in October 2026. UPI use is one of the digital-footprint signals our features are built from, so the news shows that footprint exists at scale.

**Q1.7 (Probing) What does the Labour Ministry's social-security board have to do with credit?**

Nothing directly. It is context: the Ministry notified a National Social Security Board for unorganised, gig and platform workers on 30 Oct 2026. It shows policy attention on this workforce. Do not present it as evidence for any method.

> **Trap:** If pushed, say it is background, not a proof.

**Q1.8 (Basic) What are the four questions the project answers, and which method answers each?**

Who is likely to default: interpretable ML (LR, RF, XGBoost, LightGBM, ensembles) plus SHAP. Which combinations of behaviours signal delinquency: association rules. When will portfolio defaults rise and by how much: SARIMAX forecasting with stress scenarios. What should the underwriter do now: the dashboard that combines all three.

**Q1.9 (Probing) The README says traditional scoring "ignores alternative signals, is hard to explain, and ignores macro shocks". How does the project answer each?**

Alternative signals: features from income patterns, bill payment, recharge, BNPL, UPI use and savings. Hard to explain: SHAP reasons, readable warning rules and a "path to approval". Macro shocks: the SARIMAX forecast turns the economy into a multiplier applied to every applicant's default chance.

**Q1.10 (Probing) The README cites "1.4 billion unbanked adults". Does that validate your data?**

No. It is context for the problem statement (World Bank Global Findex 2021), not a validation of the synthetic data.

## 2 · Rubric coverage

*README: "Rubric Coverage" tables for Review 1 and Review 2*

**Q2.1 (Basic) How does the project map to the Review 1 rubric (20 marks)?**

Problem statement and dataset description (2): notebook 01 Part A, the README, the data dictionary. Cleaning and preprocessing (3): notebook 01. EDA (3): notebook 02, 38 charts. Feature engineering (2): notebook 03. Model development (4): notebook 04. Evaluation and interpretation (3): notebook 05. Presentation and individual contribution (3).

**Q2.2 (Basic) How does it map to the Review 2 rubric (20 marks)?**

Method 1 time series: 2 for preparation and implementation, 3 for evaluation. Method 2 association rules: 2 and 3. Combined business insights: 2. Interactive dashboard: 5, mandatory. Presentation and individual contribution: 3. Total 20.

**Q2.3 (Probing) What is "leak-free cross-validation" and why does the README stress it?**

Leakage means information from the test data reaching the model during training. Our audit found SMOTE (oversampling) was applied before cross-validation, so synthetic copies of test rows leaked in: random forest scored CV AUC 0.98 but 0.69 on test. We moved SMOTE inside the cross-validation pipeline, so CV is now consistent with test (0.72).

*Evidence: README "Audit Fixes Applied to Review 1"*

**Q2.4 (Probing) What are WoE and IV, and why were they added?**

Weight of evidence and information value are the standard credit-scoring way to measure how much a feature separates defaulters from non-defaulters. The README's audit table admits the earlier README claimed WoE/IV without implementing them, and we then added the screen.

> **Trap:** Be ready to say the claim was wrong before and was fixed. It is in the audit table.

**Q2.5 (Probing) Why use VIF, mutual information and RFE in feature engineering?**

VIF flags features that overlap and make a scorecard unstable. Mutual information and recursive feature elimination (RFE) rank features by usefulness. Both are fitted on training rows only, with a zero-variance filter, after the audit found selection leakage.

**Q2.6 (Probing) Why is SMOTE applied "train only"?**

SMOTE creates synthetic minority-class rows. If it touches the test set, or happens before cross-validation, the evaluation is inflated. Applying it to training data only (and inside the CV pipeline) keeps the test result honest.

**Q2.7 (Probing) What is the "INR cost-optimal threshold"?**

Instead of flagging default at a 0.5 probability, we choose the cut-off that minimises money lost: a missed default costs about ₹81.9k and a wrongly rejected good borrower about ₹9.7k of lost interest. At that cut-off 74.7% are approved with a 4.4% default rate, against 7.9% if everyone were approved.

**Q2.8 (Probing) Where is the final report the rubric asks for?**

The README's links to `reports/Review_2_Final_Report.md` and the slides file are out of date. On 10 Nov a teammate's commit replaced them with `reports/Final_Report.pdf`. The old Markdown files are still in the git history (for example `git show 33cee16:reports/Review_2_Final_Report.md`).

> **Trap:** Know this before the panel clicks a broken link. The README should be updated to point at Final_Report.pdf.

## 3 · Data collection

*README: "Data Collection" and "Live macroeconomic APIs"*

**Q3.1 (Basic) Why is no Kaggle, UCI or GitHub dataset used?**

The rubric asks for data we collect or generate ourselves. Real borrower files for Indian gig workers are legally closed (Credit Information Companies Act 2005, DPDP Act 2023). So borrower data is generated, and everything legally open (macro and some credit data) is collected from official APIs.

**Q3.2 (Basic) Which data is synthetic and which is real?**

Synthetic: borrower_profiles.csv (30,000), monthly_transactions.csv (360,000), macroeconomic_indicators.csv (36 rows, Review 1 EDA and an offline fallback). Simulated from real macro data: the loan book in data/portfolio. Real: macro_india_monthly.csv (108 months from FRED and World Bank) and the five real credit datasets in the real-data track.

**Q3.3 (Basic) Which APIs supply the macro data, and do they need a key?**

FRED (CPI inflation, call-money rate, 10-year G-sec yield, INR per USD) and the World Bank (unemployment, GDP growth, lending rate). Neither needs a key. Each raw response is cached in data/external/raw and provenance is logged per column in macro_sources.json.

**Q3.4 (Basic) What are the fallback rules if an API is down?**

Live API, then the cached raw file, then a synthetic series, so the pipeline always runs offline. The provenance file records whether each column came live.

**Q3.5 (Probing) The World Bank series are annual. Isn't interpolating them to monthly questionable?**

Yes, it is a limitation, and the README lists it. Unemployment, GDP growth and lending rate are annual and are interpolated to months. It smooths movements inside a year, so we do not claim month-level timing for those three drivers.

*Evidence: README "Limitations"*

**Q3.6 (Probing) Why does CPI stop in March 2025?**

FRED's series ends there. It is carried forward and flagged in the data. This is a stated limitation.

**Q3.7 (Probing) What is the macroeconomic_indicators.csv with 36 rows?**

A synthetic macro series used by the Review 1 EDA and kept as the last-resort offline fallback. The real macro data is data/external/macro_india_monthly.csv.

**Q3.8 (Probing) Why is the loan book called a simulation?**

No real portfolio is available. We simulate repayment of the same 30,000 borrowers from 2019 to 2024, with each borrower's monthly default hazard moving with real macro series. That gives the forecast a monthly default series to learn from.

> **Trap:** The forecaster later "discovers" the macro link we built in. That validates the pipeline, not real borrowers. The README limitations say so.

**Q3.9 (Basic) Name the series IDs.**

FRED: CPALTT01INM659N (CPI), IRSTCI01INM156N (call-money rate), INDIRLTLT01STM (10-year yield), EXINUS (INR per USD). World Bank: SL.UEM.TOTL.ZS (unemployment), NY.GDP.MKTP.KD.ZG (GDP growth), FR.INR.LEND (lending rate).

## 4 · How borrower_profiles.csv is generated, and why

*README: "How borrower_profiles.csv is generated, and why", Steps 1 to 3, and the young and low-income table*

**Q4.1 (Basic) Explain in two sentences how the borrower data is generated.**

`data_generation.py` draws each variable from a stated distribution (seed 42), simulates 12 months of transactions per borrower, summarises them into features, and then draws a default flag from a logistic formula of ten risk signals plus noise. The shapes are our assumptions, checked against real figures where a verified one exists.

**Q4.2 (Basic) Why should the examiners accept synthetic data?**

Real borrower files are closed by law, and banks and regulators use synthetic data for the same reason (JPMorgan AI Research 2020, UK FCA 2024). We do not claim it is real. We use it as a controlled test bench with a known ceiling, we compare it with real figures, and we ran the same pipeline on four real credit datasets.

**Q4.3 (Basic) Why seed 42?**

Any fixed seed makes every number reproducible. A reviewer who re-runs `python data_generation.py` gets identical files. 42 is a convention, not a tuned value.

**Q4.4 (Probing) Why age Normal(32, 8) clipped to 21 to 60?**

Gig work skews young. The one verified survey (NCAER, 924 food-delivery workers) has mean age 29.1 and 84% under 35; ours is 31.8 and 65%. So our borrowers are older than that survey. The 32 is our assumption; a defensible reason is that the platform list also includes freelancers, tutors and content creators, who skew older than delivery riders.

> **Trap:** Admit the mismatch. The README verdict is "older than the one verified survey".

**Q4.5 (Probing) Why is gender 58 / 40 / 2, and is it a model input?**

It is an assumption for a mixed gig population, with no verified anchor. It is not a model input, so the model never scores on gender. We audit performance by gender separately.

**Q4.6 (Probing) How were the location weights and income multipliers chosen?**

Metro 30%, Tier-1 25%, Tier-2 20%, Semi-urban 15%, Rural 10%, with income multipliers 1.4, 1.2, 1.0, 0.8, 0.6. They are assumptions: platform work is urban-heavy and city pay is higher. There is no verified anchor for these numbers.

**Q4.7 (Probing) Education is marked "Matches". On what basis?**

The NCAER survey reports 6% below Class X, 56% Class X to below graduate and 38% graduate. Ours is 5.1%, 55.2% and 39.7%. It is one platform's workers, so it supports the shape, not all gig work.

*Evidence: research/papers/2023_NCAER_PRI_workshop_food_delivery_platform_workers.pdf*

**Q4.8 (Probing) Why is tenure Exponential with mean 18 months, and does it match anything?**

Many gig workers are new and many leave, which an exponential shape captures. NCAER's active workers are 48.7% under one year, 25.7% one to two years and 25.7% over two years; ours is 48.4%, 26.6% and 25.0%. It matches closely.

**Q4.9 (Probing) Why does income range from ₹8,000 to ₹55,000, and does it match reality?**

Uniform ₹8,000 to ₹55,000, times a location and an education multiplier, to span low to middle earners. It does not match: only 15.2% earn under ₹15,000 against about 40% in the Economic Survey, and our median is ₹31,532 against ₹20,744 for NCAER long-shift workers. The README verdict is "too high".

> **Trap:** State it as a known optimistic bias, then give the re-weighting test.

**Q4.10 (Hard) Does the too-high income break your results?**

We re-weighted the data to 40% low earners: AUC stayed 0.747 and predictions stayed calibrated. The default rate fell from 7.9% to 7.0%, which reflects the income and loan-size flaw. So model ranking is not sensitive to it, but absolute losses would be.

*Evidence: README Review 3 section; DATASET_VALIDATION_SUMMARY.md section 9*

**Q4.11 (Probing) Why are bank-account (85%) and dependents distributions given?**

Both are assumptions with no verified anchor. The data shows dependents carry no effect on default, so nothing in the results depends on that choice.

**Q4.12 (Probing) Why is the platform drawn uniformly over eight kinds?**

Deliberately, so the model gets no platform shortcut. In Review 1 a platform risk feature was also dropped after a chi-square test (p = 0.71) found no association.

*Evidence: README audit table*

**Q4.13 (Probing) How is a bill's on-time probability set?**

0.6 + 0.005 × tenure + 0.3 × (this month's income ÷ base income − 1), clipped between 0.3 and 0.98. Steadier and longer-serving workers pay more reliably, and a bad income month makes a late bill more likely. The direction is plausible; the sizes are our judgement.

**Q4.14 (Hard) Why do younger people use more BNPL in your data, and what supports it?**

BNPL probability is 0.3 + 0.02 × (35 − age) − 0.002 × tenure. It is the only way age enters the data. The support is a US regulator slide (CFPB 2022) saying BNPL borrowers skew young. It is US evidence and one source, so call it an assumption.

**Q4.15 (Probing) Why 28-day recharge cycles?**

Prepaid plans ran on 28-day cycles until 30-day validity was required in January 2022. It is a plausibility check, not a validation.

**Q4.16 (Probing) How is the default label made, and why that way?**

Each borrower's default probability is 1 ÷ (1 + e^−z) where z adds ten weighted risk signals and Normal noise; the flag is a random draw from that probability. We use a known formula because no labelled gig-worker repayment data is open to us: it gives a ground truth, so we can say exactly how close a model gets. Logistic is how scorecards turn a weighted sum into a probability.

**Q4.17 (Hard) Where do the coefficients come from?**

We set them by hand. The signs follow the credit-risk literature and are checked against real datasets (five of nine testable drivers match in direction). The sizes are judgement. A model trained on the label re-discovers our formula, so synthetic accuracy tests the pipeline, not the world.

> **Trap:** This is the most important honest sentence in the whole defence.

**Q4.18 (Probing) Why does the noise term matter?**

Normal(0, 0.5) stands for everything unobserved. It is why even a perfect model cannot exceed AUC 0.781: each label is a random draw from a known probability. Our model reaches 0.741, about 85% of what is achievable.

**Q4.19 (Hard) Which terms matter most in the formula?**

By each term's share of the total spread in the generated file: utility on-time share 16.0%, loan amount 16.0%, gig tenure 14.7%, falling income 13.9%, income volatility 10.2%, BNPL debt 9.3%, savings buffer 9.3%, past repayment 5.5%, app logins 4.7%, recharge gap 0.5%.

**Q4.20 (Hard) Which formula terms does real data support?**

On-time share (HELOC −0.28, Taiwan −0.14), tenure (HELOC −0.26 and −0.22), savings (Taiwan −0.14), loan size (German credit +0.09 and +0.21) and BNPL debt (all five real checks positive). Income volatility and falling income cannot be checked: no public dataset records them.

**Q4.21 (Hard) Why does default rise with income in your data?**

Loans are sized at 1.5 to 6 months of income and the formula adds risk per rupee borrowed, so richer borrowers take bigger loans and look riskier. Default is 4.8%, 6.8%, 8.8% and 11.2% from the lowest to the highest income quartile. Real data runs the other way. It is a documented generator flaw.

> **Trap:** Do not claim the synthetic data shows low-income people defaulting more.

**Q4.22 (Hard) Do young and low-income borrowers really default more in the real world?**

There is supporting evidence, with modest effects. The New York Fed says card delinquency is generally highest among the youngest borrowers. The JPMorgan Chase Institute found 70% of 18 to 24 year olds and 74% of the bottom income quintile had income swings above 30%, against 55% nationally. On public credit data age correlates with default at about −0.12 in two of three datasets, and income at −0.07 and −0.17. These are correlates of risk, not causes.

*Evidence: research/papers [73], [74], [75]*

**Q4.23 (Hard) What do young and low-income borrowers look like in YOUR data?**

Age has no direct term. It acts only through BNPL, giving 8.6% default at ages 21 to 25 and 7.3% at 41 to 60. Income is the reverse of real data. So the safe sentence is: income instability and a short track record are the risk drivers here, and youth and low income are documented real-world correlates of them.

**Q4.24 (Probing) Why is the overall default rate 7.87%?**

The intercept was set so it lands near 8%. It is higher than microfinance portfolio-at-risk (2.2% in March 2024, 6.3% in March 2025, 1.6% in June 2026, MFIN) by design, because thin-file gig workers sit outside those books.

**Q4.25 (Probing) What is in monthly_transactions.csv and why 12 months?**

Per borrower and month: income, utility bill and on-time flag, recharge, BNPL, spending, month-end balance, app logins and UPI payments. Twelve months is enough to measure volatility, trend and bill punctuality without making the file huge (360,000 rows).

## 5 · Why these models, and why five

*README: "Step 3: why these models" and the Review 1 rubric row "5 baselines, tuned LR/RF/XGB/LGBM, voting and stacking"*

**Q5.1 (Basic) Why five baseline models?**

One from each family that credit-scoring studies compare: linear (logistic regression), bagged trees (random forest), and three boosted-tree models (XGBoost, LightGBM, scikit-learn gradient boosting). They were not picked for their scores.

**Q5.2 (Probing) Why logistic regression?**

It is the transparent scorecard baseline the industry already uses. Lessmann et al. (2015) compared 41 classifiers on 8 real credit datasets and found several beat it significantly, so it is the baseline, not the answer.

*Evidence: research/papers/2015_Lessmann_benchmarking_credit_scoring.pdf*

**Q5.3 (Probing) Why random forest?**

Lessmann et al. recommend it as the minimum benchmark a credit model should be compared with.

**Q5.4 (Probing) Why XGBoost and LightGBM?**

Gunnarsson et al. (2021) found XGBoost best overall on credit data, and deep networks no better and much costlier. LightGBM is the same family, fast, supports exact TreeSHAP explanations and could be calibrated, so it became the final model.

**Q5.5 (Probing) Why add voting and stacking?**

Dastile et al. (2020), a review of 74 studies, found ensembles generally beat single models. Here they gave no real gain: stacking has the top AUC (0.739) but flags no defaulters at the 0.5 cut-off.

> **Trap:** Do not call stacking the best model. Its precision and recall at 0.5 are both zero.

**Q5.6 (Probing) Which Review 1 model is best?**

By AUC, tuned XGBoost (0.738) is the best single model. By F1, tuned LightGBM (0.256). The four tuned models are within about 0.03 AUC of each other, so the choice was driven by calibration and explainability.

**Q5.7 (Probing) Why is LightGBM the final model if XGBoost has the higher AUC?**

The gap is 0.003 AUC, which is noise. LightGBM is fast, gives exact TreeSHAP reasons and could be calibrated. The dashboard uses a calibrated LightGBM trained without class re-weighting: AUC 0.741, Brier 0.068.

**Q5.8 (Probing) Why not deep learning?**

Gunnarsson et al. found deep networks no better than boosted trees on credit data and far costlier. A lender must also give reasons for a decline, and TreeSHAP gives exact reasons for trees.

**Q5.9 (Hard) What did hyper-parameter tuning achieve?**

It helped XGBoost most (+0.056 AUC, from 0.682 to 0.738) and almost nothing for logistic regression (0.728 before and after). That shows a transparent scorecard was already close to the best.

**Q5.10 (Probing) Why does the README say Review 1 models "over-state PD"?**

Review 1 trained with SMOTE or class weights, which inflates predicted probabilities (mean about 40% against 7.9% observed). They rank people well but cannot be read as a chance of default, so the dashboard uses a separately calibrated model.

## 6 · Real-data validation track

*README: "Real-Data Validation Track" and "Innovations"*

**Q6.1 (Basic) What did the Review 1 panel ask and what did you do?**

The panel asked "why synthetic data, use real data". Because borrower files are closed by law, we collected every legally open real credit-risk dataset from official APIs (IMF, World Bank, FRED, CFPB HMDA, CFPB complaints; 69,662 records) and re-ran both Unit 3 methods on them in notebook 09.

**Q6.2 (Probing) What is each real dataset used for?**

India bank NPL ratio (IMF, 63 quarters): Method 1 on real Indian defaults. World Bank NPL panel (5,642 rows): a 140-country check of the macro-to-default mechanism. US consumer-loan delinquency (FRED, 158 quarters): conformal prediction intervals. CFPB HMDA (48,799 applications): Method 2 on real loans and fairness. CFPB complaints (15,000): borrower pain points.

**Q6.3 (Probing) Explain the India NPL result: MAE 0.48 against 2.04 for naive.**

On 14 unseen quarters of India's bank bad-loan ratio, SARIMAX with RBI regime ramps (chosen by BIC) averaged a 0.48 percentage-point error. ETS scored 0.91, ARIMA 1.03 and a naive last-value forecast 2.04. So the method works on a real Indian default series, and beats simple baselines by a wide margin.

**Q6.4 (Hard) What are "RBI regime ramps"?**

India's NPL path is shaped by policy: the Asset Quality Review (+8% a quarter while active), the resolution and IBC clean-up (−5.4% a quarter) and the COVID standstill (+6.4%, not significant). We add them as ramp variables, and they explain the real path better than macro drivers alone.

*Evidence: models/real/real_data_results.json*

**Q6.5 (Hard) What does "adaptive conformal 80% band covers 81%, Gaussian band 72%" mean?**

A forecast interval promises to cover the truth 80% of the time. On 94 rolling US delinquency forecasts the model's own band covered only 72%, too narrow. Adaptive conformal inference (Gibbs and Candès 2021) widened it as errors showed up and covered 81%. It makes the stated uncertainty honest.

**Q6.6 (Probing) What does the 140-country panel show?**

Changes in unemployment (+0.28), GDP growth (−0.11) and inflation (+0.08) move bank bad-loan ratios in the same direction as in our simulation, all with p below 0.05. So the macro-to-default mechanism is real, not something only our simulation does.

**Q6.7 (Hard) Explain the HMDA result: "82 rules, 95% keep lift ≥ 1.3, rule-count AUC 0.707".**

We mined warning rules on real US mortgage applications from Delaware and Rhode Island and checked them on Vermont, a state the rules never saw. 82 rules were found and 95% still went with denial at least 1.3 times the average. The number of rules an application triggers alone scores AUC 0.707.

**Q6.8 (Probing) Why use mortgage data in a gig-worker project?**

Method 2 (association rules) is general. HMDA is the largest legally open dataset of real lending decisions, so it tests whether mined rules survive on an unseen group. It does not claim mortgages are like gig loans.

**Q6.9 (Probing) Where was the synthetic data wrong, according to the README?**

It has no COVID forbearance (real bank NPLs fell in 2020 while ours spiked), and it under-weights debt burden (real debt-to-income of 50% or more shows lift 2.55 against 1.12 synthetic).

**Q6.10 (Hard) What is the fairness audit?**

On real HMDA decisions, after controlling for the risk flags, applicants who are Black or Asian still had about 1.7 times the odds of denial (1.74 and 1.77), applicants with race unavailable 1.55, and women 1.007 (not significant). It shows why audits matter even when risk flags are in the model.

*Evidence: models/real/real_data_results.json*

**Q6.11 (Basic) Name the five innovations.**

Conformal prediction for credit forecasts, regime-aware NPL forecasting for India, a counterfactual path to approval in the dashboard, a synthetic-versus-real fidelity scorecard, and a fairness audit on real lending decisions.

**Q6.12 (Probing) What is the "path to approval"?**

The cheapest realistic changes that would turn a decline into an approval: a smaller loan, a different tenure, less BNPL, better bill payment, more savings. It never changes the interest rate or immutable attributes such as age, location and education. It follows Wachter 2017, Ustun 2019 and DiCE 2020.

## 7 · Review 3: proof and the new dashboard tabs

*README: "Review 3 — Proof, Data Explorer, Method Tabs and Scenario Lab"*

**Q7.1 (Basic) What question did the Review 3 panel ask?**

"How do we know these numbers are real, and does the synthetic data behave like real data?" The answer is a set of dashboard tabs and a re-runnable validation suite, where every table is read from a file.

**Q7.2 (Probing) What does the four-dataset table show?**

The same model recipe with 5-fold cross-validation scores AUC 0.793 (German credit), 0.786 (Taiwan cards), 0.866 (Give Me Some Credit) and 0.794 (FICO HELOC), against 0.748 for our simulated data. So the pipeline works on real credit data, and the simulated data is not unrealistically easy: it scores below all four.

**Q7.3 (Probing) Five of nine drivers agree with real data. Which, and what about the rest?**

Same direction: late payments, on-time share, loan size, track record, savings. Negligible in the simulation: age, dependents, debt burden. Opposite: income level, because loan size scales with income.

**Q7.4 (Hard) What does "alternative data adds +4.8 AUC points" prove?**

A conventional scorecard (application form plus past loans) reaches AUC 0.701; adding utility, recharge, savings and income-pattern data reaches 0.748. Berg et al. (2020) report +5.3 on real data. But the effect is built into the simulation, so it shows the mechanism and does not prove it.

> **Trap:** The README says this itself. Say it before they do.

**Q7.5 (Hard) Explain "Rs 245 lakh against Rs 200 lakh against Rs 35 lakh".**

On the 6,000 test borrowers, approving by each applicant's own break-even limit earns Rs 245 lakh. The best single risk cut-off for everyone earns Rs 200 lakh, and approving everyone earns Rs 35 lakh. The break-even rule wins because each applicant's limit depends on their rate and loan length.

**Q7.6 (Probing) What is "not proven", according to the README?**

Performance on real Indian gig-worker borrowers. No open data exists. The one real Indian fintech study reports AUC 0.60, so expect lower accuracy on real borrowers than the 0.74 we see.

**Q7.7 (Basic) What does the Proof & comparison tab do?**

It shows the pipeline on four real credit datasets, realism tests of the synthetic data (against published Indian figures, risk-driver directions, default dynamics and re-weighting), comparison with published studies and existing products, and a section on how the data is made step by step.

**Q7.8 (Hard) What does the Insights & Validation tab do?**

It tests the synthetic data against the UCI credit-card dataset. S→S (train and test on synthetic) AUC 0.741; U→U (UCI ceiling) 0.770; S→U (TSTR) 0.517; U→S (TRTS) 0.561. The AUC gap of 0.224 is labelled a strong domain mismatch, and 1 of 10 features is close to real.

**Q7.9 (Probing) What do TSTR and TRTS mean?**

Train on Synthetic, Test on Real (TSTR) measures whether a model trained on synthetic data works on real data. Train on Real, Test on Synthetic (TRTS) is the reverse. A big drop on TSTR means the synthetic data does not carry real-world structure.

**Q7.10 (Hard) TSTR is 0.517, almost a coin flip. Isn't that fatal?**

It is an honest finding of domain mismatch: UCI is Taiwanese credit-card holders mapped through a crosswalk, while ours is gig-worker behaviour. The same recipe gets 0.77 on UCI when trained on UCI, so the method is fine. We report the gap instead of hiding it.

**Q7.11 (Basic) What do the other new tabs do?**

Data explorer: browse and chart all datasets and trace one borrower's 12 months. Review 1 models: the 11 first-round models and how the final model improved. Time series & rules: SARIMAX history and forecast, and the 93 rules, with a full explorer in an expander. Scenario lab: a live what-if on the economy with forecast, expected loss and lending decisions.

**Q7.12 (Basic) How many tabs does the dashboard have?**

Nine: Start here, Data explorer, Score applicant, Scenario lab, Test borrowers, Review 1 models, Time series & rules, Proof & comparison, Insights & Validation.

**Q7.13 (Basic) How do I re-run the validation?**

`python scripts/run_validation_suite.py` (about 7 minutes, downloads four public datasets from OpenML once) and `python scripts/build_fidelity_anchors.py`. The UCI study is `python validation/run_all.py`.

## 8 · Architecture and repository

*README: "Architecture" flowchart and "Repository Structure"*

**Q8.1 (Basic) Walk through the architecture diagram.**

data_generation.py creates 30,000 borrowers and 360,000 transactions, which feed the Review 1 notebooks. FRED and World Bank data go through macro_api.py into the portfolio simulation, which produces the loan book. That feeds time-series forecasting (06) and association rules (07). The underwriting model gives the calibrated PD. All three meet in the combined insights notebook (08) and the Streamlit dashboard.

**Q8.2 (Probing) What does src/underwriting_model.py do?**

It trains and serves the calibrated LightGBM default model and holds the break-even decision rule (margin ÷ (margin + loss given default)) and the risk grades A to E.

**Q8.3 (Probing) What is the difference between validation/ and models/validation/?**

validation/ holds the UCI comparative study (crosswalk, TSTR/TRTS, calibration, fidelity audit; run_all.py). models/validation/ holds the outputs of the Review 3 four-dataset suite and the fidelity anchors that the Proof tab reads.

**Q8.4 (Basic) What is in research/?**

The evidence pack: verified sources, saved open-access papers, benchmark tables, reviewer answers and competitive_landscape.json. Papers are in research/papers, numbered as in REFERENCES_ACM.md.

**Q8.5 (Probing) Why separate src/ modules from notebooks?**

Notebooks explain and show results; src holds reusable, testable code that the notebooks and dashboard both import (macro collector, simulation, model, rules, real-data collectors, recourse). Unit tests cover config, rules, simulation and risk analytics.

## 9 · How to run and reproduce

*README: "How to Run"*

**Q9.1 (Basic) What are the steps to reproduce everything?**

Install requirements, then generate data (`python data_generation.py`), collect macro data (`python -m src.macro_api`), simulate the loan book (`python -m src.portfolio_simulation`), train the PD model (`python -m src.underwriting_model`), run notebooks 01 to 09 in order, optionally run the validation scripts, and start the dashboard with `streamlit run dashboard/app.py` from the repository root.

**Q9.2 (Basic) Why run Streamlit from the repository root?**

So the theme in `.streamlit/config.toml` applies and the relative data paths resolve.

**Q9.3 (Basic) How is reproducibility guaranteed?**

All randomness is seeded: 42 for data and models, 2026 for the loan-book simulation. Re-running reproduces every number in the README.

**Q9.4 (Probing) Why is LightGBM pinned below 4.7?**

LightGBM 4.7.0 crashes on Windows (it hung the Review 1 modelling notebook), so requirements pin `lightgbm<4.7`.

## 10 · Results: Review 1 default prediction

*README: "Results" → "Review 1 — default prediction (test set, 6,000 borrowers)"*

**Q10.1 (Basic) Explain the Review 1 results table.**

On 6,000 unseen borrowers, tuned XGBoost has the best AUC (0.738) and PR-AUC (0.204), tuned LightGBM the best F1 (0.256), logistic regression 0.728 and random forest 0.713. The calibrated LightGBM used in the dashboard has AUC 0.741. The last row is the ceiling: 0.781.

**Q10.2 (Probing) What is the "ceiling" of 0.781 and how do you know it?**

The default flag is a random draw from a known probability (default_probability_true). Scoring that true probability gives AUC 0.781, and no model can beat it. We can therefore say XGBoost captures (0.738 − 0.5) ÷ (0.781 − 0.5) ≈ 85% of the achievable discrimination.

**Q10.3 (Probing) What does KS = 0.36 mean?**

KS is the largest gap between the cumulative score distributions of defaulters and non-defaulters. Retail credit practice treats 0.3 to 0.4 as good separation for an application scorecard, especially without bureau data.

**Q10.4 (Probing) What do "2.9× lift in the top decile" and "riskiest 30% hold 63% of defaults" mean?**

The 10% of applicants ranked riskiest default at 23%, 2.9 times the 7.9% average. The riskiest 30% contain 63% of all defaults, so declining or re-pricing that 30% removes almost two-thirds of losses while serving 70% of applicants.

**Q10.5 (Probing) Explain the cost-optimal threshold result.**

A missed default costs about ₹81.9k and a wrongly rejected good borrower about ₹9.7k of lost interest, so the best cut-off is far below 0.5. At that cut-off 74.7% are approved and the approved default rate is 4.4%, against 7.9% if everyone is approved.

**Q10.6 (Hard) PR-AUC is only about 0.2. Is that poor?**

With 7.9% defaulters, a random model's PR-AUC is about 0.079. At 0.20 the model is roughly 2.5 times better than random, and the ceiling is 0.262. PR-AUC is reported because accuracy misleads when defaults are rare.

**Q10.7 (Hard) Why is LightGBM's recall at 0.5 lower than XGBoost's?**

At a 0.5 cut-off each model trades precision against recall differently. XGBoost flags more defaulters (recall 0.682) with lower precision, LightGBM fewer (0.614) with higher precision, which gives it the better F1. The dashboard does not use 0.5; it uses the break-even rule.

**Q10.8 (Basic) What are the top SHAP drivers?**

Gig tenure, income trend, on-time utility payments, loan size and income volatility.

## 11 · Results: Method 1, portfolio default forecasting

*README: "Review 2 · Method 1" tables and scenario table*

**Q11.1 (Basic) What is Method 1 and what does it forecast?**

A time-series model of the monthly share of active loans newly 90+ days late, in the simulated loan book. It uses SARIMAX with economic drivers and forecasts 2025 under three economies.

**Q11.2 (Probing) Explain the model comparison table.**

Each method forecast 2024, which it had not seen. SARIMAX with macro drivers has the lowest error: 0.054 percentage points (MAPE 16%), against 0.065 for gradient boosting, 0.075 for Holt-Winters, 0.088 for SARIMA and 0.124 for seasonal naive. In a rolling back-test of 40 forecast origins its mean error is 0.124 against 0.176, 0.208 and 0.351.

**Q11.3 (Probing) Why SARIMAX with macro drivers rather than a plain model?**

Defaults respond to unemployment, GDP, inflation and rates about two months later, the time a missed instalment needs to roll to 90 days. Granger tests support the drivers, and adding them cut 2024 error by about 38% against plain SARIMA.

**Q11.4 (Hard) Explain "(1,0,1)(1,0,0,12)".**

One autoregressive term and one moving-average term for month-to-month memory, one seasonal autoregressive term at lag 12, and no differencing (d = 0) because after regressing on the drivers the residuals are stationary (ADF p = 0.0004). Orders were chosen by BIC over a grid of fits on 60 training months, with seven drivers lagged two months.

**Q11.5 (Hard) "Unemployment +1 pp → +21% defaults". How is that read?**

The model works on the log of the default rate, so each coefficient is a percentage effect. A one-point rise in unemployment two months earlier goes with about 21% higher monthly defaults; CPI +15%, call-money rate +10%, a lockdown month ×5.7. Every driver is significant with the expected sign.

**Q11.6 (Hard) Is the lockdown effect of ×5.7 believable when real bank bad loans fell in 2020?**

Not as evidence about real banks. The spike is an assumption of the simulation: real bank NPLs fell because of the moratorium and write-offs. The Severe scenario depends on it, so treat Severe as a stress test, not a forecast.

> **Trap:** The README Limitations and the Proof tab both say so. Say it first.

**Q11.7 (Hard) What do the residual and interval results show?**

Residuals are white noise (Ljung-Box p = 0.12 at lag 6 and 0.28 at lag 12), so the model has left no pattern unexplained. The 80% prediction interval covers 92% of the 2024 actuals.

**Q11.8 (Probing) Explain the 2025 scenario table.**

Baseline: 0.28% average monthly default rate, expected credit loss ₹1.59 crore. Adverse (unemployment +1.5 pp, GDP −3 pp): 0.33%, ₹1.95 crore, +₹36 lakh. Severe (COVID-style): 0.74% with a 2.7% peak, ₹4.32 crore, +₹2.74 crore.

**Q11.9 (Hard) How is expected credit loss computed?**

Expected defaults × exposure at default × loss given default. For the baseline: about 219 defaults × ₹111,589 average exposure × 65% loss given default ≈ ₹1.59 crore. The 65% is an assumption that the Scenario lab lets you change.

**Q11.10 (Hard) What is the risk multiplier in the dashboard and how reliable is it?**

Each scenario's 12-month average forecast divided by the neutral-economy rate (0.4854%): 0.567, 0.689 and 1.526. The dashboard multiplies every applicant's default chance by it ("point-in-time PD = PD × multiplier"). It is a stress-testing overlay: the shocks are assumptions, the series is simulated, one number is applied to everyone, and the result depends on the neutral reference. Normal economy is 1.0 by default.

> **Trap:** Baseline and Adverse are both below 1. Only Severe raises risk above average.

## 12 · Results: Method 2, warning rules

*README: "Review 2 · Method 2" bullets*

**Q12.1 (Basic) What is Method 2 in plain words?**

It finds combinations of behaviours that go with many more defaults than average, such as falling income plus bumpy income plus low savings. Each combination is a rule that an underwriter can read.

**Q12.2 (Basic) Explain support, confidence and lift.**

Support: the share of borrowers who show the whole combination. Confidence: the share of those who defaulted. Lift: confidence divided by the average default rate (7.9%), that is, how many times riskier. A lift of 3 means three times as risky.

**Q12.3 (Probing) How did 2,852 itemsets become 93 rules?**

19 behaviour flags on 24,000 training borrowers give 2,852 frequent itemsets. Rules were kept if lift was at least 1.5, support at least 0.5%, and a one-sided Fisher exact test was significant after Bonferroni correction, then redundant (subset) rules were pruned.

**Q12.4 (Hard) Why a Bonferroni-corrected Fisher test?**

Searching thousands of candidate rules produces many that look good by chance. Testing each one and correcting for how many were tested keeps only patterns unlikely to be luck.

**Q12.5 (Probing) How do you know the rules are not just overfitted?**

They were mined on 24,000 training borrowers and re-checked on 6,000 unseen ones. All 93 kept a test lift of at least 1.3 (minimum 1.35); median lift fell from 2.03 on training to 1.89 on test, the expected selection shrinkage.

**Q12.6 (Basic) What is the top rule?**

Declining income, volatile income and low savings buffer: a 24.7% default rate, 3.1 times the average on unseen borrowers.

**Q12.7 (Probing) Explain "6+ rules default at 24.7% against 4.9% with none".**

Applicants who trigger six or more warning rules default at 24.7%; those who trigger none default at 4.9%. Routing applicants with three or more rules to manual review flags 14% of them and catches 34% of all defaults.

**Q12.8 (Hard) What does "low income and rural location add nothing extra under macro stress" mean?**

When the economy weakens, volatile income adds 4.5 extra default points, but being low-income or rural adds none beyond what it already did. In this data it is instability, not income level or place, that makes a downturn hurt.

**Q12.9 (Hard) Apriori or FP-Growth, and why?**

Both return identical frequent itemsets. Apriori was faster at every support level tested, because with 19 yes/no flags the table is small and dense, so building an FP-tree costs more than it saves. FP-Growth pays off on large sparse baskets.

## 13 · Results: combined view

*README: "Combined"*

**Q13.1 (Hard) What does "rules add information the PD alone misses" mean?**

Inside the lowest-risk PD band, applicants with six or more triggered rules default at 5.6% against 1.8% with none, three times more. So the rules carry information beyond the model's probability.

> **Trap:** It is not universal. In band B the pattern reverses (7.0% with up to two rules against 4.7% with six or more).

**Q13.2 (Probing) Explain "approvals move from 75% to 48%".**

Point-in-time PD is the model PD times the scenario multiplier. Under the same break-even rule, the benign 2025 baseline approves about 75% of applicants and the severe scenario about 48%, because every applicant looks riskier.

**Q13.3 (Probing) What is the final decision rule in the dashboard?**

Decline if the default chance reaches the break-even limit, or the risk grade is E, or 15 or more warning rules fire. Approve with conditions if 3 or more rules fire or the grade is C or D. Otherwise approve. Break-even = margin ÷ (margin + loss given default).

## 14 · The dashboard

*README: "Dashboard" screenshots and "How to Run"*

**Q14.1 (Basic) What does the Score applicant tab do?**

Pick one of three sample customers or enter your own. The decision, the gauge, the default chance, the break-even limit, expected loss and suggested rate update instantly. Below it: a checklist of how the decision was reached, the economy table, SHAP reasons, the warning patterns hit, and a path to approval if declined.

**Q14.2 (Basic) What is the 60-second demo?**

Open Score applicant and pick Customer 3: declined, with reasons. Pick Customer 2: approved with conditions. In the sidebar choose Severe downturn: Customer 2 flips to Decline, because the same person is a worse risk in a bad economy.

**Q14.3 (Probing) Why does the decision sometimes not change when I edit one field?**

Because the applicant must cross a threshold, not just move a little. For example a very high-risk applicant with many warning patterns stays declined even if tenure improves, since 15 or more patterns force a decline. The checklist shows which test set the verdict.

**Q14.4 (Probing) What does the Scenario lab add over a fixed forecast?**

You build the economy yourself with sliders for unemployment, GDP, inflation, rates, rupee and lockdown months. The SARIMAX model is refitted live and the three presets reproduce the saved scenario file exactly. It shows forecast, expected loss and lending decisions.

**Q14.5 (Basic) What does the Test borrowers tab show?**

The 6,000 unseen borrowers with predicted risk against what actually happened, filterable by risk grade, location, gig platform, outcome (repaid or defaulted) and number of warning patterns hit.

## 15 · Audit fixes applied to Review 1

*README: "Audit Fixes Applied to Review 1 (October 2026)" table*

**Q15.1 (Basic) Why does the README list its own mistakes?**

Because a self-audit of the Review 1 pipeline found leakage and bugs that made the results unreliable, and Review 2 builds on those models. Listing each issue, its impact and its fix shows the results were re-checked, and it is stronger than hiding them.

**Q15.2 (Probing) What was wrong with the month labels?**

They were built with a 30-day step (`timedelta(30 * i)`), which drifted and produced repeated months (two "2025-01" rows per borrower). Fixed with `pd.date_range(..., freq="MS")`; the numbers did not change.

**Q15.3 (Probing) What was the ordinal-encoding bug?**

The ordinal maps used labels that were not in the data (for example "High School" and "Urban"), so most rows were encoded as −1. We now use the central `ORDINAL_MAPS` in `src/config.py` and assert every category maps.

**Q15.4 (Hard) What leaked in the M3 target-mean features?**

`platform_risk_premium` and `macro_stress_exposure` were target means computed over all rows, so test labels leaked into features. We split first, used smoothed target encoding on train only, and dropped platform (chi-square p = 0.71).

**Q15.5 (Hard) What was wrong with RFE?**

It was fitted on one random sample's X with another sample's y, so the feature selection was noise. It now uses a single fixed train sample.

**Q15.6 (Hard) What else was wrong in feature selection?**

Mutual information and RFE were fitted on all rows, and a constant `total_utility_bills` was selected. Selection is now train-only with a zero-variance filter, and WoE/IV was genuinely implemented (the README had claimed it before it existed).

**Q15.7 (Hard) What was the SMOTE problem, and what changed?**

Cross-validation and tuning ran on data that had already been oversampled, which inflated scores: random forest CV AUC 0.98 against 0.69 on test. SMOTE now sits inside an imblearn pipeline, and random forest CV is 0.72, consistent with test.

**Q15.8 (Hard) What happened in notebook 05?**

It looked for `test_data.csv` and `*.pkl` files that did not exist, so it silently re-trained different, untuned models. It now loads M3's `test.csv` and M4's `tuned_*.joblib`.

**Q15.9 (Probing) What were the remaining fixes?**

A hard-coded `c:/Users/rgdar/...` path (now relative paths), qualitative "Very High" tables (now computed) and dollar costs (now INR costs from the portfolio), and LightGBM 4.7.0 crashing on Windows (pinned `lightgbm<4.7`).

## 16 · Contributions and governance

*README: "Individual Contributions & Project Governance"*

**Q16.1 (Basic) Who did what, according to the README?**

Darun R G: data architecture, the synthetic generator, feature engineering, ARM data prep, dataset validation summary. goldmauler: lead for Method 1 (time series) and Method 2 (association rules). Abhinav Juturu: portfolio risk modelling, EDA, loan-book simulation. Gowtham Reddy: the interactive dashboard and full-stack UI. Adithya: business intelligence, evaluation, technical report and slides.

**Q16.2 (Probing) What did you personally build? (Goldmauler)**

The README lists the lead on Method 1 and Method 2: stationarity tests, STL, Granger, SARIMAX with live APIs, Apriori versus FP-Growth, the 93 warning rules, and the macro API collector. The git history adds the Review 1 audit fixes, the real-data track (notebook 09), the Review 3 validation suite and dashboard tabs, and the final report draft.

*Evidence: git log --author=Goldmauler*

**Q16.3 (Basic) How were tasks tracked?**

GitHub Issues #2 to #19 (18 tasks, TSK-101 to TSK-405) in four closed sprint milestones, each with a status label and an assignee. Review 1 tracking is in the team repo issues. Setup is in GITHUB_BOARD_SETUP.md.

**Q16.4 (Basic) Where is Review 1 stored?**

In the team repo Darun1017/credit-scoring-engine. This private repo holds Reviews 2 and 3.

**Q16.5 (Basic) How will the 15-minute defence be shared?**

Each member presents their designated technical module. The README ties each to its marks: Method 1 and Method 2 (5 each), the interactive application (5), combined insights (2) and presentation (3).

## 17 · Limitations

*README: "Limitations"*

**Q17.1 (Basic) What is the biggest limitation?**

The borrowers, transactions and loan book are synthetic. The pipeline is validated on real credit data and real macro series, but we cannot show how it performs on real Indian gig-worker borrowers, because no open data exists.

**Q17.2 (Probing) What does the README say about the World Bank and CPI data?**

World Bank indicators are annual, interpolated to monthly. CPI on FRED ends in March 2025 and is carried forward, flagged in the data.

**Q17.3 (Hard) Why does the README say the forecaster "recovering the link" is not evidence?**

The loan book's defaults come from a macro-linked hazard we built (a CreditPortfolioView-style logit link). A forecaster that finds that link shows the pipeline works. It says nothing about real borrowers, which is why notebook 09 repeats the check on real series.

**Q17.4 (Probing) Which synthetic-data gaps did Review 3 expose?**

Higher income looks riskier (loan size scales with income and risk is charged per rupee), debt burden, age and dependents carry almost no effect, interest rates are set independently of risk, and the simulated monthly default series is far noisier than real bank series. They are listed in the Proof tab.

**Q17.5 (Probing) What else does the README admit?**

The SMOTE and class-weighted Review 1 models rank well but over-state PD (mean about 40% against 7.9% observed), so the dashboard uses a separately calibrated model.

## 18 · Questions that cut across the README

*Likely closing and challenge questions*

**Q18.1 (Basic) Give me the 30-second summary.**

Gig workers are a large, low-paid group that lenders cannot score. We built an explainable engine: a calibrated default model, a forecast of how defaults move with the economy, and readable warning rules, joined in a dashboard that gives a money-based decision, the reasons and a path to approval. Real borrower files are closed, so the data is synthetic, and we proved the pipeline on four real credit datasets and real macro data while reporting where the synthetic data fails.

**Q18.2 (Probing) Why should we believe your numbers?**

Every number is recomputed from saved files, all randomness is seeded, the saved results reproduce exactly when re-run, and 30 unit tests pass. The README also lists what does not match: income, the COVID spike, and the generator gaps.

**Q18.3 (Probing) What would you do with real data?**

Retrain the model on a lender's own repayment history, collected with consent (for example through the Account Aggregator framework under the DPDP Act), then recheck calibration and fairness. The pipeline, validation suite and dashboard stay the same. Expect lower accuracy: the one real Indian fintech study reports AUC about 0.60.

**Q18.4 (Probing) Your AUC is 0.74 but a real Indian fintech study got 0.60. Why is yours higher?**

Ours is built that way: the label is a random draw from a formula over the same features the model sees, so the data is easier than real life. The ceiling is 0.781. Real data has hidden factors we cannot simulate. That is why we say real performance should be lower.

**Q18.5 (Hard) Is the model fair?**

Gender is not a model input. We audit by group: AUC is 0.725 for women, 0.754 for men and 0.646 for the small "other" group, so accuracy is not equal. On real HMDA decisions the audit found higher denial odds for Black and Asian applicants after controlling for risk flags. Fairness work on real Indian data is future work.

**Q18.6 (Probing) Why 65% loss given default and a 9% cost of funds?**

They are project assumptions used in the break-even rule and expected-loss calculations, not measured values. The Scenario lab lets you change loss given default.

**Q18.7 (Probing) What is the break-even rule?**

A loan is profitable on average only if the chance of default is below margin ÷ (margin + loss given default), where margin is the interest earned over the cost of funds. Each applicant gets their own limit from their rate and loan length, which is why it beats one cut-off for everyone.

**Q18.8 (Basic) What is the single most honest sentence in the project?**

"The synthetic data is a controlled test bench: it shows the pipeline recovers a known signal and stays calibrated. Whether real gig workers behave this way is a separate question, and we tested the mechanisms we could on real datasets and reported where they failed."

**Q18.9 (Probing) A panel member says the README links are broken. What do you say?**

The final report is now `reports/Final_Report.pdf`. The earlier Markdown report and slides were replaced by a teammate's commit on 10 Nov and are still in the git history. The README links will be updated to point at the PDF.

> **Trap:** Fix the links before the panel sees them.
