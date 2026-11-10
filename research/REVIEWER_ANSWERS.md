# Reviewer answers: short, evidence-backed (Review 2)

Compiled 2026-11-06 by SCOUT. Bracketed numbers point to `REFERENCES_ACM.md`. Each answer gives a 20-second version first, then the evidence. Where the honest answer includes a weakness, it is stated, because reviewers reward candour more than overclaiming.

---

## Q1. "Why synthetic data?"

**Short answer:** Real loan-level data on Indian thin-file borrowers is not publicly available and cannot legally be shared. Synthetic data let us build and test the full pipeline end to end with a known ground truth. We now add a real-data track to test the same methods on real outcomes.

Evidence:
- Banks and regulators use synthetic data precisely because privacy and regulation block sharing real customer data. J.P. Morgan AI Research sets out the case and the pitfalls [13]. The UK FCA's Synthetic Data Expert Group lists model testing and validation, and data augmentation, as legitimate uses [14][15].
- In India, borrower records are personal data under DPDPA 2023 s.2(t) [20]. Credit information can only be held by RBI-registered credit information companies under CICRA 2005 [22]. RBI also requires lending data to be stored on servers in India [60, para 13(iv)]. A student team cannot obtain or redistribute such data.
- A known PD gives a measurable ceiling (AUC 0.781), so we can say exactly how much signal the models recover (about 85%). Real data cannot tell us that.
- Limit (say it yourself): synthetic results validate the method, not real-world performance [19].

## Q2. "Is your synthetic data realistic?"

**Short answer:** It is calibrated to published Indian aggregates, and we now test fidelity formally with marginal, pairwise and TSTR checks. Some earlier "validation" citations were wrong and have been corrected.

Evidence and the checks to show:
- **Fidelity metrics:** per-column distribution similarity and pairwise correlation similarity (SDV/SDMetrics) [17]; **TSTR**, which trains on synthetic data and tests on real data, is the evaluation idea from Esteban et al. [16] and is named by the FCA for synthetic-data model validation [15].
- **Aggregate anchors (verified):** gig workers 7.7M in FY21 rising to 12M in FY25, with about 40% earning under ₹15k/month and many having thin-file credit histories [64][65]. MFI PAR 31–180 was 2.2% (Mar 2024), 6.3% (Mar 2025) and 1.6% (Jun 2026) [67]. Our 7.87% default rate is deliberately above MFI levels because the segment is excluded from mainstream MFIs.
- **Be honest:** our simulated COVID spike (2.3× the 2019 level over 2020, 6× in the peak month) has no RBI benchmark. Reported bank GNPA actually *fell* from 8.2% to 7.3% in FY21 because of the moratorium and write-offs [63]. Present the spike as a stress assumption.
- **AUC realism check:** real alternative-data AUCs range from 0.60 (Indian fintech [4]) to 0.74 (Germany, footprint plus bureau [1]). Our 0.738 is at the top of that range, consistent with clean synthetic signal.

## Q3. "Where is the real data?"

**Short answer:** Four real public datasets, each reproducing one of our methods on real outcomes.

| Track | Real dataset | What it tests | Method reused |
|---|---|---|---|
| (a) | World Bank bank-NPL ratio panel FB.AST.NPER.ZS (India: 9.98% in 2017 to 2.06% in 2025) plus WDI macro drivers [70] | Do the macro drivers in our SARIMAX explain *real* NPLs? | Panel / dynamic regression, following Beck et al. [23] (75 countries: GDP growth, share prices, FX, lending rate) |
| (b) | FRED / Fed Board US credit-card and consumer-loan delinquency and charge-off rates (Q2 2026: 2.85% / 3.82% / 2.62%) [71] | Forecasting real default rates with calibrated intervals | SARIMAX + **adaptive conformal intervals** [33] |
| (c) | CFPB/FFIEC HMDA loan-level applications: `action_taken`, `denial_reason-1..4`, DTI, income, race, ethnicity, sex [52] | Do association rules recover *real* denial patterns? Are there fairness gaps? | Apriori / FP-Growth [38][39] with Webb-style significance [40]; fairness as in [50][51] |
| (d) | CFPB consumer complaint narratives (payday and personal loans) [57][58] | What goes wrong for borrowers after the loan? | LDA [53] / BERTopic [54], following [55] |

Caveats: (b), (c) and (d) are US data, chosen because no Indian equivalent is publicly released at loan level. The macro data in Method 1 was already real (FRED and World Bank) [70][71].

## Q4. "Why SARIMAX and not LSTM?"

**Short answer:** We have 72 monthly points. On short economic series, statistical models beat ML in the largest published comparisons, and SARIMAX gives interpretable macro elasticities that a risk committee can use.

Evidence:
- On 1,045 monthly M3 series, 8 statistical methods beat 8 popular ML methods on both accuracy measures at every horizon, at lower computational cost [36].
- In M4 (100,000 series), none of the 6 pure ML methods beat the simple combination benchmark and only 1 beat Naïve2. The winner was a statistical–RNN hybrid, about 10% better [37].
- SARIMAX is regression with ARIMA errors [28][29]. It yields signed, significant coefficients (unemployment +1 pp → +21% defaults) that match the NPL literature [23][24][26]. An LSTM gives none of this, and it would need far more than 72 observations.
- Uncertainty: conformal intervals [32][33] give coverage guarantees without assuming Gaussian errors. That is our innovation on top of SARIMAX.
- What we did test: gradient boosting with lags plus macro (MAE 0.065) lost to SARIMAX (0.054).

## Q5. "Why association rules for credit?"

**Short answer:** Rules give combinations of behaviours an underwriter can read and act on, and lenders must state specific reasons for adverse decisions. We guard against spurious rules with multiple-testing correction and out-of-sample validation.

Evidence:
- Apriori and FP-Growth [38][39] are standard. ARM has been applied to personal-loan credit-risk warning [41] and to P2P lending [42]. Human-readable rule tables for credit risk have a long record in management science [12].
- Large rule searches produce false discoveries. Webb [40] recommends holdout or corrected significance tests. We use Bonferroni-corrected Fisher tests, and **all 93 rules keep lift ≥ 1.3 on unseen borrowers**.
- Regulatory fit: US Reg B requires a "statement of specific reasons" that "indicate the principal reason(s)" [48]. Rules plus SHAP produce exactly that.
- The added value is measurable: within the lowest-risk PD band, applicants triggering 6+ rules default 3× more (5.6% vs 1.8%).
- Next step: rerun on real HMDA denials [52].

## Q6. "How is this different from CIBIL?"

**Short answer:** CIBIL is a credit bureau that scores people who already have credit history. IntelliCredit is a lender-side decision engine for people the bureau cannot score, and it uses consented alternative data. It complements the bureau score and does not replace it.

Evidence:
- Under CICRA 2005, credit bureaus must be RBI-registered CICs. There are four: TransUnion CIBIL, Equifax, Experian and CRIF High Mark [22]. IntelliCredit does not collect or sell credit information, so it is not a CIC.
- Bureau scores need history. An estimated ~129 million Indians have mostly been denied credit for lack of history [4], and the Economic Survey 2025-26 notes that gig workers' thin-file histories restrict formal credit [65].
- Alternative data complements bureau data: a combined AUC of 73.6% vs 68.3% for the bureau alone [1]. In India, fintechs using UPI-era data were the only lenders reaching no-history borrowers, with no rise in defaults [5].
- IntelliCredit adds three things a bureau score does not: macro-adjusted point-in-time PD (Method 1), behaviour-combination warnings (Method 2), and reasons plus a path to approval (SHAP [43] plus counterfactuals [44][45][46]).

## Q7. "Is it legal and ethical under DPDPA?"

**Short answer:** Our current data falls outside DPDPA, because it is synthetic, aggregate macro data, and US public de-identified releases. A real deployment would rely on explicit consent through the Account Aggregator framework and would follow RBI's digital-lending data rules.

Evidence:
- DPDPA s.2(t): personal data is data about an identifiable individual [20]. Synthetic borrowers and country-level FRED and World Bank series are not about identifiable people.
- HMDA public LAR and CFPB narratives are released by US agencies. Narratives are published only with opt-in consent and after a three-step scrub, and the CFPB itself says this minimises but does not eliminate re-identification risk [58]. We will not attempt re-identification and will report only aggregates.
- Research exemption: s.17(2)(b) exempts processing for research or statistics *if the data is not used to make decisions about a specific person* and prescribed standards are met [20]. The Rules commence in phases, with most obligations about 18 months after 13 Nov 2025 [21]. Do not claim the exemption for a production scorer.
- Ethics: ML can widen disparities even while AUC improves [51], and lending discrimination is measurable in HMDA [50]. So we report fairness metrics and exclude protected or immutable attributes from recourse suggestions [45][47].

## Q8. "How would a real lender deploy this?"

**Short answer:** As a decision-support layer for an RBI-regulated entity. Data comes in through Account Aggregator consent, is stored in India, and is scored with a calibrated PD plus macro overlay. Rule-based flags route applicants to manual review, and every decline comes with specific reasons and a path to approval.

Steps, each tied to a rule or a piece of evidence:
1. **Data in:** bank-statement and UPI cash-flow data via the Account Aggregator framework (2.88 bn accounts enabled, 284.6M linked, 989 FIUs live as of Mar 2026) [68]. The lending app must collect only what it needs, with prior explicit consent, and **must not read files, media, contacts or call logs** [60, para 12(i)]. Call-log models such as [2][3][4] are therefore not deployable in India, which is why IntelliCredit uses cash-flow and utility behaviour.
2. **Storage:** all data on servers in India [60, para 13(iv)].
3. **Assessment:** RBI requires the borrower's economic profile (at least age, occupation and income) to be assessed before lending [60, para 7(i)]. Our model's inputs cover these.
4. **Scoring:** bureau score where available plus the IntelliCredit PD, calibrated and multiplied by the SARIMAX/ACI macro multiplier. Approvals drop from 75% (baseline) to 48% (severe), which turns the stress test into credit policy.
5. **Review routing:** ≥ 3 triggered rules sends the case to manual review (14% of applicants, catching 34% of defaults).
6. **Disclosure:** a Key Fact Statement plus a cooling-off exit option [60]. Give specific reasons for any decline (the Reg B standard [48]) and a counterfactual path to approval [44][46].
7. **Monitoring:** conformal coverage tracking, rule-lift drift checks, and fairness audits, with retraining as the macro regime changes [33][51].
