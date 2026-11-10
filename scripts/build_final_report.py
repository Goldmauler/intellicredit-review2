"""Builds reports/IntelliCredit_Final_Report_Draft.docx in the course's Final Project Documentation Format.

Every number in this script was read from the executed notebooks and saved model outputs:
models/*.csv, models/macro_overlay.json, models/real/real_data_results.json, models/subgroup_performance.csv.
Text in [square brackets] must be completed by the team (names, register numbers, guide, declarations).

    python scripts/build_final_report.py
"""
import json
import os

import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT  # noqa: F401
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "reports", "figures")
OUT = os.path.join(ROOT, "reports", "IntelliCredit_Final_Report_Draft.docx")
REAL = json.load(open(os.path.join(ROOT, "models", "real", "real_data_results.json"), encoding="utf-8"))
TS = pd.read_csv(os.path.join(ROOT, "models", "ts_model_comparison.csv"), index_col=0)
BT = pd.read_csv(os.path.join(ROOT, "models", "ts_backtest_summary.csv"), index_col=0)
SC = pd.read_csv(os.path.join(ROOT, "models", "ts_scenario_impact.csv"), index_col=0)
MC = pd.read_csv(os.path.join(ROOT, "models", "model_comparison.csv"), index_col=0)
SG = pd.read_csv(os.path.join(ROOT, "models", "subgroup_performance.csv"))
IND = pd.read_csv(os.path.join(ROOT, "models", "real", "india_npl_model_comparison.csv"), index_col=0)

doc = Document()
sec = doc.sections[0]
for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, side, Cm(2.54))

st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(12)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
st.paragraph_format.line_spacing = 1.5
st.paragraph_format.space_after = Pt(4)
for name, size in (("Heading 1", 14), ("Heading 2", 13), ("Heading 3", 12), ("Title", 20)):
    h = doc.styles[name]
    h.font.name = "Times New Roman"
    h.font.size = Pt(size)
    h.font.bold = True
    h.font.color.rgb = RGBColor(0, 0, 0)
    h.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    rf = h.element.rPr.rFonts
    rf.set(qn("w:ascii"), "Times New Roman")
    rf.set(qn("w:hAnsi"), "Times New Roman")
    for att in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        rf.attrib.pop(qn(att), None)

FIGNO = [0]
TABNO = [0]
FIGS = []   # figure index for Section 7.2


def H(text, level=1):
    doc.add_heading(text, level=level)


def P(text, bold_lead=None, italic=False, align=None):
    p = doc.add_paragraph()
    if bold_lead:
        p.add_run(bold_lead).bold = True
    r = p.add_run(text)
    r.italic = italic
    if align:
        p.alignment = align
    p.paragraph_format.alignment = align if align else WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def B(items, style="List Bullet"):
    for it in items:
        p = doc.add_paragraph(style=style)
        if isinstance(it, tuple):
            p.add_run(it[0]).bold = True
            p.add_run(it[1])
        else:
            p.add_run(it)


def T(header, rows, caption, widths=None, size=10):
    TABNO[0] += 1
    cap = doc.add_paragraph()
    cap.add_run(f"Table {TABNO[0]}. ").bold = True
    cap.add_run(caption)
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(str(h))
        r.bold = True
        r.font.size = Pt(size)
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:fill"), "E7ECEA")
        c._tc.get_or_add_tcPr().append(shd)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.size = Pt(size)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                p.paragraph_format.line_spacing = 1.0
                p.paragraph_format.space_after = Pt(0)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph()
    return TABNO[0]


def F(path, caption, source, width=15.5):
    FIGNO[0] += 1
    doc.add_picture(path, width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.add_run(f"Figure {FIGNO[0]}. ").bold = True
    cap.add_run(caption + " ")
    s = cap.add_run(f"Source: {source}.")
    s.italic = True
    for r in cap.runs:
        r.font.size = Pt(10)
    FIGS.append((FIGNO[0], caption, source))
    return FIGNO[0]


def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def toc():
    p = doc.add_paragraph()
    r = p.add_run()
    for tag, txt in (("begin", None), (None, 'TOC \\o "1-3" \\h \\z \\u'), ("separate", None), (None, None), ("end", None)):
        if tag:
            e = OxmlElement("w:fldChar")
            e.set(qn("w:fldCharType"), tag)
            r._r.append(e)
        elif txt:
            e = OxmlElement("w:instrText")
            e.set(qn("xml:space"), "preserve")
            e.text = txt
            r._r.append(e)
        else:
            e = OxmlElement("w:t")
            e.text = "Right-click and choose Update Field to build the table of contents."
            r._r.append(e)


f2 = lambda x: f"{x:.2f}"  # noqa: E731
f3 = lambda x: f"{x:.3f}"  # noqa: E731

# =========================================================================================== COVER
t = doc.add_paragraph(style="Title")
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
t.add_run("IntelliCredit: FinTech Micro-Lending Risk and Alternative Credit Scoring Engine")
for line in ["Capstone Project, Business Analytics, Units 1, 2 and 3: Final Project Documentation",
             "[Programme / Section]   ·   Team [No.]",
             "",
             "Darun R G (CB.SC.U4CSE23215)",
             "Vimalharihar S K [Register No.]",
             "Abhinav Juturu [Register No.]",
             "Gowtham Reddy [Register No.]",
             "Adithya [Register No.]",
             "",
             "Guide: [Name, Designation]",
             "Academic year 2026–27   ·   Submitted [date]",
             "Repository: https://github.com/Goldmauler/intellicredit-review2"]:
    p = doc.add_paragraph(line)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
page_break()

H("Declaration and Acknowledgement", 1)
P("We declare that this report and the accompanying code are our own work, carried out for the Business Analytics capstone. "
  "Sources of data, methods, software and prior studies are cited in the text and listed in the References. The use of "
  "generative-AI tools is disclosed in Section 8.3. [Signatures of all members.]")
P("We thank [guide] for the Review 1 feedback that led us to collect and test real credit data, and the IMF, World Bank, "
  "Federal Reserve Bank of St. Louis and the US Consumer Financial Protection Bureau for open access to their data.")

H("Abstract", 1)
P("Gig and platform workers in India, estimated at 7.7 million in 2020–21 and about 12 million by FY25 [16, 17], mostly have "
  "thin or no credit-bureau files, so lenders either exclude them or lend without evidence. IntelliCredit is an analytics "
  "engine that scores such borrowers from cash-flow behaviour, forecasts when portfolio defaults will rise, and explains "
  "every decision. Because individual credit records in India are confidential, borrower data (30,000 borrowers, 360,000 "
  "monthly transactions) were generated with documented, seeded logic, while all macroeconomic drivers were collected live "
  "from the FRED and World Bank APIs. In response to Review 1 feedback, five further real datasets (69,662 records from the "
  "IMF, World Bank, FRED and CFPB) were collected and used to re-test both advanced methods. A calibrated LightGBM model "
  "reaches a test AUC of 0.741. A SARIMAX model with lagged macro drivers forecasts the monthly portfolio default rate with "
  "a 12-month hold-out MAE of 0.0545 percentage points, ahead of six alternatives; on India's real bank NPL ratio a "
  "regime-aware SARIMAX attains 0.48 points against 2.04 for a naive forecast. Association rule mining yields 93 validated "
  "warning rules, and on 48,799 real loan applications 95% of mined denial rules hold in a state not used for mining. A "
  "Streamlit dashboard combines the three layers with macro-scenario approval, counterfactual paths to approval and "
  "conformal prediction intervals.")

page_break()
H("Executive Summary", 1)
P("Lenders to gig workers face two linked problems: they cannot tell which applicants will repay without a credit history, "
  "and they do not know when the economy will push defaults up. IntelliCredit addresses both.", )
B([("Who will repay. ", "Cash-flow signals (income stability, utility bill punctuality, months on the platform, loan size "
    "relative to income) rank applicants with an AUC of 0.741, close to the 0.781 ceiling set by the data. The approval rule "
    "approves an applicant only when the predicted default probability is below the break-even level for the loan's rate and tenure."),
   ("When risk rises. ", "The monthly default rate is forecast from real Indian unemployment, GDP growth, inflation, interest-rate "
    "and exchange-rate data. Under a severe 2025 downturn expected credit loss rises from ₹1.59 crore to ₹4.32 crore, and the "
    "same rule automatically cuts approvals from 75% to 48%."),
   ("Why an applicant is risky. ", "93 tested warning patterns, such as volatile and falling income with thin savings, flag "
    "14% of applicants who account for 34% of defaults, including risky applicants inside the model's safest band."),
   ("Evidence from real data. ", "The forecasting and rule-mining methods were re-run on real Indian bank NPLs and on 48,799 real "
    "US loan applications and held up. The real data also showed where the simulation was wrong: COVID defaults in India were "
    "deferred by regulatory forbearance rather than spiking, and real lenders weight debt burden more heavily.")])
P("Recommended actions: a two-tier decision (automatic approval only for low-risk applicants with fewer than three warning "
  "patterns), loan-size caps for new or declining-income workers, a monthly macro-adjustment of default probabilities, "
  "selective tightening on behaviour rather than on demographics in downturns, and provisioning at the baseline plus an "
  "adverse overlay of about ₹36 lakh.", "Recommended actions. ")
page_break()
H("Contents", 1)
toc()
P("Abbreviations: AUC area under the ROC curve; ACI adaptive conformal inference; ARM association rule mining; BIC Bayesian "
  "information criterion; DPD days past due; DTI debt-to-income; ECL expected credit loss; EAD exposure at default; HMDA "
  "Home Mortgage Disclosure Act; LGD loss given default; MAE mean absolute error; MASE mean absolute scaled error; NPL "
  "non-performing loan; PD probability of default; PiT point-in-time; SARIMAX seasonal ARIMA with exogenous regressors; "
  "SHAP Shapley additive explanations; TTC through-the-cycle.")
page_break()

# =========================================================================================== 1
H("1. Introduction, Business Objectives and Literature Review", 1)
H("1.1 Context and scope", 2)
P("Digital lenders and microfinance institutions serve borrowers whom bureau scores cannot rank. About 1.4 billion adults "
  "worldwide were unbanked in 2021 [21], and India's gig workforce is growing quickly; the Economic Survey 2025-26 reports "
  "that about 40% of gig workers earn under ₹15,000 a month [17]. The stakeholders are the lender's underwriting team "
  "(approve or decline, loan size), its risk and finance teams (provisions and capital), and borrowers, who need fair and "
  "explained decisions. Scope: unsecured micro-loans of ₹5,000–₹3,00,000 over 3–24 months to individual gig workers; "
  "group lending, collections strategy and pricing optimisation are out of scope.")
T(["Analytical question", "KPI (unit)", "Baseline", "Target / success criterion", "Result"],
  [["Who will default?", "Test AUC", "Logistic regression 0.728", "≥ 0.73 and calibrated", "0.741, mean PD 7.9% vs observed 7.9%"],
   ["How many bad loans does the policy let in?", "Default rate of approved loans (%)", "7.87% (approve all)", "< 6% in a neutral economy", "5.09% at 60.3% approval"],
   ["When will portfolio defaults rise?", "12-month MAE (pp)", "Naive 0.088", "Beat every benchmark", "0.0545 (MASE 0.079)"],
   ["Which behaviour combinations signal default?", "Share of rules valid on unseen borrowers", "—", "≥ 80% keep lift ≥ 1.3", "100% of 93 rules"],
   ["Do the methods hold on real data?", "India NPL MAE (pp); real rules valid out-of-state", "Naive 2.04", "Beat naive; ≥ 80% valid", "0.48; 95% of 82 rules"]],
  "Business questions and KPIs.", widths=[3.6, 3.0, 2.6, 3.2, 3.6], size=9)
P("Figure 1 summarises the end-to-end workflow: Unit 1 (cleaning, EDA) and Unit 2 (features, models) produce the PD; Unit 3 "
  "Method 1 produces the macro multiplier and loss scenarios, Method 2 the warning rules; all three feed the dashboard, and "
  "the real-data track re-tests both Unit 3 methods.")
F(os.path.join(FIG, "final", "F00_workflow.png"), "End-to-end workflow connecting Units 1–3, the real-data track and the dashboard.",
  "authors' diagram (scripts/build_final_report.py)", width=16)
H("1.2 Related analytics work and research gap", 2)
P("Berg et al. [1] show on about 250,000 German e-commerce purchases that a simple digital footprint predicts default as well "
  "as a bureau score (AUC 0.696 vs 0.683) and that the two combined reach 0.736. Óskarsdóttir et al. [2] add mobile-phone call "
  "records for 22,000 applicants and raise a random forest's AUC from 0.899 to 0.923. Agarwal et al. [3] report similar gains "
  "from mobile footprints at an Indian fintech. Lessmann et al. [4] benchmark 41 classifiers on credit data and find that "
  "ensembles such as random forests and gradient boosting rank above logistic regression. For portfolio-level risk, Beck, "
  "Jakubík and Piloiu [5] and Nkusu [6] link NPL ratios to GDP growth, unemployment, interest rates and exchange rates across "
  "countries. Association rules [7, 8] are widely used for basket analysis but rarely as validated, significance-tested credit "
  "warnings, where Webb [9] warns that large rule searches produce false discoveries.")
P("The gap: (a) call-record scoring as in [2, 3] cannot be deployed in India, because RBI's Digital Lending Directions forbid "
  "lending apps from accessing contacts and call logs [15]; (b) studies rarely connect borrower-level scoring to a macro "
  "forecast that changes the approval rule month by month; (c) published rule-mining work seldom validates rules on unseen "
  "data; and (d) model-based forecast intervals assume the model is correct. IntelliCredit uses only consented cash-flow "
  "signals, links the PD to a macro forecast, validates every rule out-of-sample, and adds conformal intervals [10, 11] and "
  "actionable counterfactual explanations [12, 13].")

# =========================================================================================== 2
H("2. Dataset Collection, Validation and Preparation", 1)
H("2.1 Provenance, coverage and challenges", 2)
T(["Dataset", "Source and method", "Records × variables", "Period", "Licence / use"],
  [["Borrower profiles", "Synthetic: data_generation.py (NumPy, Faker en_IN), seed 42", "30,000 × 39", "snapshot", "own work"],
   ["Monthly transactions", "Synthetic, same generator", "360,000 × 14", "12 months", "own work"],
   ["India macro indicators", "FRED API (CPI, call-money, 10Y G-sec, INR/USD); World Bank API (unemployment, GDP growth, lending rate)", "108 months × 7 + 2 event flags", "2017–2025", "public, cited"],
   ["Loan book and performance", "Simulation of the 30,000 borrowers driven by the real macro series (src/portfolio_simulation.py, seed 2026)", "30,000 loans; 250,191 loan-months", "2019–2024", "own work"],
   ["India bank NPL ratio", "IMF Financial Soundness Indicators SDMX API", "63 quarters × 2", "2010-Q1 → 2026-Q2", "CC BY 4.0"],
   ["NPL + macro panel", "World Bank WDI API", "5,642 country-years × 10", "2000–2025", "CC BY 4.0"],
   ["US consumer credit", "FRED (Federal Reserve delinquency and charge-off release)", "158 quarters × 6", "1987–2026", "public domain"],
   ["HMDA loan applications", "CFPB HMDA Data Browser API (DE, RI, VT, 2023)", "48,799 × 31", "2023", "US open data"],
   ["Loan complaints", "CFPB Consumer Complaint Database API", "15,000 × 10", "2020–2023", "US open data"]],
  "Data sources. No Kaggle, UCI or GitHub dataset is used. Provenance, endpoints and fetch times are logged in "
  "data/external/macro_sources.json and data/real/real_sources.json.", widths=[3.0, 5.2, 3.2, 2.4, 2.2], size=9)
P("The target is loan_default_status (1 = 90+ days past due), with a 7.87% default rate. The level of observation is one "
  "borrower (Units 1–2, Method 2) or one month of the portfolio (Method 1). The data dictionary is in data/data_dictionary.txt "
  "and data/data_dictionary_review2.md; a 100-row sample is in data/sample_final_dataset.csv.", "Target and dictionary. ")
P("Collection challenges and fixes: World Bank series are annual, so they were interpolated to months and flagged; the "
  "lending rate stops in 2022 and the carried-forward months are excluded from history; the CFPB endpoints reject custom "
  "user-agent strings, so the library default is used; the CFPB search API pages with a search-after key, not an offset, and "
  "no longer returns narrative text, so text mining was not possible; three early IMF quarters are missing and were "
  "interpolated in logs. Every collector caches the raw response, so the pipeline also runs offline.")
P("Synthetic generation logic: borrower attributes are generated by documented, seeded rules (for example a base income drawn "
  "uniformly between ₹8,000 and ₹55,000 and scaled by location and education, with monthly noise that is larger for newer gig "
  "workers, and monthly bill punctuality drawn as on-time or late events whose probability rises with tenure) with correlations "
  "between income, volatility, savings and BNPL use; the default label "
  "is drawn from a logistic function of these signals, which fixes a theoretical AUC ceiling of 0.781 (the AUC of the true "
  "default probability). To guard against circular findings, every mechanism the synthetic data relies on was re-tested on "
  "real data in notebook 09 (Section 5.3). Two held and two did not (Table 8).")
H("2.2 Data quality, privacy and ethics", 2)
P("Quality checks (notebooks 01, 06, 09): no duplicate borrowers; missing values handled by KNN imputation (numeric) and mode "
  "(categorical); IQR outlier review with winsorisation of the most skewed fields; class imbalance 7.87% handled with SMOTE "
  "on training folds only; 0 missing months and 0 missing values in the 72-month modelling window. In the real data, 33 HMDA "
  "rows are identical on the 31 kept fields (kept, as they are distinct applications), and 16.5% of panel cells are missing "
  "because many countries do not report NPLs every year.")
P("Privacy and ethics: individual credit information in India may be shared only within the credit-information system "
  "(CICRA 2005 [14]), and personal data requires purpose-specific consent under the DPDP Act 2023 [19]. The synthetic "
  "borrowers are not real people; the real datasets are aggregates (IMF, World Bank, FRED) or records already de-identified by "
  "the publisher (HMDA has no names or exact ages). No personal data is stored. A deployment would require consent through "
  "the Account Aggregator framework, data minimisation, storage in India and no access to phone contacts or call logs [15]. "
  "Known biases: the synthetic population reflects our assumptions; HMDA covers US mortgages and three states; the IMF series "
  "is system-wide, not micro-loans.")
H("2.3 Transformations and usable dataset", 2)
P("General cleaning (notebook 01) is separated from model-specific preprocessing. Ordinal encoding for location tier and "
  "education, one-hot encoding for platform, log transforms of skewed amounts, and 12-month aggregation of transactions into "
  "borrower features (for example income volatility = standard deviation ÷ mean). The borrower split is stratified 80/20 "
  "(24,000 / 6,000, seed 42); scalers, WoE bins, SMOTE and rule thresholds are fitted on training rows only. Time-series models "
  "use chronological splits only. Field mapping: Unit 1 uses all raw fields; Unit 2 uses 22 application features "
  "(src/underwriting_model.py); Method 1 uses the monthly default rate and seven macro series; Method 2 uses 19 binary flags "
  "derived from Unit 2 features.")

# =========================================================================================== 3
H("3. Exploratory Data Analysis and Visualization (Unit 1)", 1)
P("Notebook 02 produced 38 charts and formal tests (chi-square for categorical fields, Mann–Whitney U for continuous ones). "
  "The findings below drove the feature and rule design.")
F(os.path.join(FIG, "final", "F02_volatility_by_default.png"), "Income volatility index by default status: defaulters have visibly more volatile income.",
  "notebooks/02_M2_exploratory_data_analysis.ipynb, Matplotlib/Seaborn", width=11)
F(os.path.join(FIG, "final", "F03_utility_by_default.png"), "Utility reliability score by default status: on-time bill payment separates the classes.",
  "notebooks/02_M2_exploratory_data_analysis.ipynb", width=11)
P("Income volatility, utility punctuality, savings buffer and gig tenure differ significantly between defaulters and "
  "non-defaulters; demographics (gender, location) differ little. This motivated volatility and reliability ratios as engineered "
  "features (Section 4) and the hypothesis that combinations of cash-flow stress signals, not single flags, mark the riskiest "
  "borrowers (tested in Method 2). Default rates by income quartile are flatter than by volatility, so low income alone is not "
  "treated as a risk flag in downturn policy.")

# =========================================================================================== 4
H("4. Feature Engineering and Predictive Modelling (Unit 2)", 1)
P("Fourteen behavioural features were engineered (income volatility index, income trend slope, utility reliability, savings "
  "buffer ratio, debt-to-inflow ratio and others), then screened with VIF, mutual information, recursive feature elimination "
  "and Weight of Evidence / Information Value on training rows; PCA and t-SNE were used for exploration only. Five baselines "
  "and four tuned models plus voting and stacking ensembles were trained with 5-fold stratified cross-validation, with SMOTE "
  "inside each fold to avoid leakage.")
rows = []
for name in ["Logistic Regression", "Logistic Regression (Tuned)", "Random Forest (Tuned)", "XGBoost (Tuned)", "LightGBM (Tuned)",
             "Voting Classifier (Tuned)", "Stacking Classifier (Tuned)"]:
    r = MC.loc[name]
    rows.append([name, f3(r["AUC-ROC"]), f3(r["Precision"]), f3(r["Recall"]), f3(r["F1-Score"])])
rows.append(["Calibrated LightGBM (dashboard model)", "0.741", "—", "—", "—"])
T(["Model (6,000 test borrowers)", "AUC", "Precision", "Recall", "F1"], rows,
  "Model comparison on the same held-out test set (models/model_comparison.csv; threshold 0.5 for precision/recall).",
  widths=[6.0, 2.0, 2.4, 2.2, 2.0], size=9)
P("Leakage audit (README, Review 1 fixes): preprocessing, SMOTE and WoE were refitted inside training folds; post-outcome "
  "fields were removed; there is one row per borrower, so no entity overlap. The stacking ensemble has the highest AUC but its "
  "maximum score is 0.33, so it flags nobody at 0.5; this is why the decision rule uses an economic threshold rather than 0.5. "
  "The dashboard uses LightGBM trained without class weights so that its probabilities are calibrated (mean PD 7.9% vs "
  "observed 7.9% on test).")
F(os.path.join(FIG, "final", "F07_shap_summary.png"), "SHAP summary for the tuned gradient-boosting model: short gig tenure, falling income, late utility payments and large loans raise predicted risk.",
  "notebooks/05_M5_evaluation_interpretation.ipynb, SHAP TreeExplainer [18]", width=11)
sg = SG.set_index(["attr", "group"])
T(["Subgroup", "n", "Default rate", "AUC", "Approval rate", "Default of approved"],
  [[f"{a}: {g}", int(r.n), f"{r.default:.1%}", f3(r.auc), f"{r.approval:.1%}", f"{r.def_approved:.1%}"] for (a, g), r in sg.iterrows()],
  "Subgroup performance of the dashboard model under the neutral break-even policy (models/subgroup_performance.csv).",
  widths=[4.4, 1.4, 2.2, 1.6, 2.4, 2.8], size=9)
P("Female and male applicants are approved at 59.8% and 60.8% (ratio 0.98, above the four-fifths guideline) with the same "
  "default rate among approved loans; the AUC is lower for women (0.725 vs 0.754), so discrimination is less sharp for that "
  "group, and the 'Other' group (n = 129) is too small to judge. Rural applicants are approved more often because their "
  "default rate is lower in the data. Gender is not a model input. These checks are on synthetic data; the real-data fairness "
  "audit is in Section 5.3.")

# =========================================================================================== 5
H("5. Advanced Analytics (Unit 3): Two Selected Methods", 1)
H("5.1 Method 1 — Time-series forecasting of the portfolio default rate", 2)
P("Business question: when will portfolio defaults rise, by how much, and what will it cost? Input: 72 monthly observations "
  "(2019-01 to 2024-12) of the default rate (new 90+ DPD loans ÷ active loans) from the macro-driven loan book, with seven real "
  "macro series. Time series is chosen because provisions and capital depend on the path of defaults over time, which a "
  "cross-sectional PD cannot give.")
P("Preparation and implementation (notebooks/06_R2_M1_time_series_forecasting.ipynb, src/macro_api.py, "
  "src/portfolio_simulation.py): the target is modelled in logs because its variance grows with its level; drivers enter with a "
  "two-month lag because a loan reaches 90 DPD two instalments after its first miss; train 2019–2023 (60 months), test 2024 "
  "(12 months). STL gives trend strength 0.46 and seasonal strength 0.10. ADF (p = 0.082) and KPSS (p = 0.01) indicate "
  "non-stationary levels. Granger tests [20] (minimum p over lags 1–4) are significant for unemployment (0.0003), GDP growth "
  "(0.004), the call-money rate (0.010) and rupee depreciation (0.046), borderline for CPI (0.051). Seven forecasters were "
  "compared; the SARIMAX driver set was chosen by BIC on training data: SARIMAX(1,0,1)×(1,0,0)₁₂ with the five lagged drivers "
  "and two COVID dummies (BIC 37.0).")
f4 = lambda x: f"{x:.4f}"  # noqa: E731
rows = [[m, f4(r["MAE (pp)"]), f4(r["RMSE (pp)"]), f"{r['MAPE %']:.1f}", f3(r["MASE"]),
         f3(BT.loc[m, "mean MAE"]) if m in BT.index else "—"] for m, r in TS.iterrows()]
T(["Model", "MAE (pp)", "RMSE (pp)", "MAPE %", "MASE", "Backtest MAE (40 origins)"], rows,
  "Method 1 evaluation: 12-month hold-out (2024) and rolling-origin backtest (models/ts_model_comparison.csv, ts_backtest_summary.csv).",
  widths=[4.6, 2.0, 2.2, 1.8, 1.6, 3.4], size=9)
F(os.path.join(FIG, "final", "F11_holdout_forecasts.png"), "2024 hold-out: 12-month-ahead forecasts against the actual default rate.",
  "notebooks/06_R2_M1_time_series_forecasting.ipynb, statsmodels", width=15)
P("Interpretation: a 1-point rise in unemployment two months earlier raises the default rate by 20.8%, CPI inflation by 15.3%, "
  "the call-money rate by 10.4%; GDP growth lowers it by 1.9% per point; lockdown months multiply it by 5.65. Residuals pass "
  "Ljung–Box (p = 0.12 at lag 6, 0.28 at lag 12) and Jarque–Bera (p = 0.31). The 80% interval covered 11 of 12 test months. "
  "The 30+ DPD rate nowcasts defaults two months ahead (R² 0.76, 2024 MAE 0.038 pp). For 2025, expected credit loss is "
  f"₹{SC.loc['Baseline', 'expected credit loss (INR lakh)']:.1f} lakh (baseline), ₹{SC.loc['Adverse', 'expected credit loss (INR lakh)']:.1f} lakh (adverse) "
  f"and ₹{SC.loc['Severe', 'expected credit loss (INR lakh)']:.1f} lakh (severe). Limitations: the forecast is conditional on the macro path; in the "
  "2021 second-wave months of the backtest SARIMAX (MAE 0.381) was worse than ETS (0.276); bias is −0.023 pp, a slight "
  "under-forecast; annual World Bank series are interpolated.")
H("5.2 Method 2 — Association rule mining of delinquency warning patterns", 2)
P("Business question: which combinations of behaviours occur together in borrowers who default far more often than chance? "
  "Input: the 24,000 training borrowers, each turned into a basket of 19 binary flags (average 5.22 per borrower) whose "
  "quantile thresholds were computed on training rows only; validation on the 6,000 test borrowers. Rules were chosen because "
  "underwriters need readable, multi-factor red flags that a single probability does not give.")
P("Implementation (notebooks/07_R2_M2_association_rule_mining.ipynb, src/risk_rules.py): Apriori [7] and FP-Growth [8] at "
  "minimum support 0.5% found the same 2,852 frequent itemsets (Apriori was faster on these 20 dense items, about 0.6 s vs "
  "1.0–1.1 s). From 21,684 rules, 242 have consequent {DEFAULT}; 170 pass lift ≥ 1.5, support ≥ 0.5% and a "
  "Bonferroni-corrected one-sided Fisher exact test; 93 remain after a productive-rule filter that requires each rule to beat "
  "all of its sub-rules by 5%.")
T(["Rule ⇒ DEFAULT", "Train lift", "Test coverage", "Test default rate", "Test lift"],
  [["Declining income + volatile income + low savings", "2.57", "3.85%", "24.7%", "3.14"],
   ["Volatile income + low savings + new gig worker", "2.47", "3.75%", "24.4%", "3.11"],
   ["Declining income + large loan vs income + low savings", "2.30", "2.75%", "24.2%", "3.08"],
   ["Declining income + low savings + new gig worker", "2.79", "4.10%", "23.6%", "3.00"]],
  "Top validated warning rules on unseen borrowers (base default rate 7.87%).", widths=[7.4, 1.8, 2.2, 2.4, 1.6], size=9)
F(os.path.join(FIG, "final", "F15_train_test_lift.png"), "Train vs test lift of the 93 rules: all keep lift ≥ 1.3 on unseen borrowers (correlation 0.61, median lift 2.03 → 1.89).",
  "notebooks/07_R2_M2_association_rule_mining.ipynb, mlxtend", width=15)
P("Interpretation: single flags are weak (best lift 1.65), but three-flag combinations of income instability and thin savings "
  "reach a test default rate near 25%. Applicants triggering three or more rules are 14% of test applicants and 34% of "
  "defaults. Under macro stress, income volatility adds 4.45 percentage points of extra default risk while low income adds "
  "none. Limitations: the default rate by rule count rises overall but not monotonically (17.4%, 13.8%, 11.3% for 3, 4 and 5 "
  "rules); the rule-count AUC is 0.645 when rules are ranked on training lift (0.657 when ranked on test lift).")
H("5.3 Re-testing both methods on real data (notebook 09)", 2)
ic = REAL["india_model_comparison"]
T(["Model (India NPL ratio, 14 unseen quarters)", "MAE (pp)", "MAPE %", "MASE"],
  [[m, f3(ic[m]["MAE (pp)"]), f"{ic[m]['MAPE %']:.1f}", f3(ic[m]["MASE"])] for m in ic],
  "Method 1 on India's real bank NPL ratio (IMF FSI), trained on 2010-Q1 to 2022-Q4.", widths=[7.4, 2.4, 2.4, 2.4], size=9)
P("BIC selected a SARIMAX with three regulatory regime ramps (RBI Asset Quality Review recognition +8.0% per quarter, "
  "IBC clean-up −5.4% per quarter, COVID standstill). Macro drivers did not improve BIC once regimes were included: in India, "
  "recognition rules moved reported NPLs more than the business cycle did.")
F(os.path.join(FIG, "real", "R2_india_npl_forecast.png"), "Forecasting India's real bank NPL ratio for 14 unseen quarters.",
  "notebooks/09_R3_real_data_validation.ipynb; IMF FSI", width=15)
uc = REAL["us_interval_coverage"]
P(f"Conformal intervals: on 94 rolling one-step forecasts of real US consumer-loan delinquency, the adaptive conformal 80% band "
  f"[11] covered {uc['80%']['adaptive conformal (ACI)']:.0%} and the 90% band {uc['90%']['adaptive conformal (ACI)']:.0%}, against "
  f"{uc['80%']['Gaussian model interval']:.0%} and {uc['90%']['Gaussian model interval']:.0%} for the model's Gaussian bands.")
P("Macro mechanism: a two-way fixed-effects panel of 140 countries (2,021 country-years, standard errors clustered by country) "
  "gives ΔNPL coefficients of +0.280 for the change in unemployment (p = 0.015), −0.111 for GDP growth (p = 0.048) and +0.080 for "
  "inflation (p = 0.002), the same signs as the simulation.")
hr = REAL["hmda_rules"]
P(f"Method 2 on real loans: on 39,451 HMDA applications from Delaware and Rhode Island, Apriori produced {hr['final']} productive "
  f"DENIED rules; {hr['share_hold_out_of_state']:.0%} keep lift ≥ 1.3 on 9,348 Vermont applications never used for mining, and the "
  f"rule-count score has AUC {REAL['hmda_rule_count_auc_VT']:.3f} there. Debt-to-income is the most cited real denial reason (43% of "
  "denials). After the risk flags, Black and Asian applicants still have about 1.7 times the odds of denial (p < 0.001); public "
  "HMDA has no credit scores, so this is a signal for review rather than proof of discrimination [22].")
T(["Claim", "Synthetic", "Real", "Verdict"],
  [[r["claim"], r["synthetic project data"], r["real data"], r["verdict"]] for r in REAL["fidelity_scorecard"]],
  "Synthetic-vs-real fidelity scorecard.", widths=[5.0, 3.4, 4.6, 3.0], size=9)

# =========================================================================================== 6
H("6. Comparative Analysis and Business Impact", 1)
H("6.1 Comparison of the two Unit 3 methods", 2)
T(["", "Method 1: time series", "Method 2: association rules"],
  [["Question", "When and how much will portfolio defaults move?", "Which behaviour combinations mark risky borrowers?"],
   ["Input", "72 monthly rates + 7 real macro series", "24,000 borrowers × 19 binary flags"],
   ["Output", "12-month forecast, scenarios in ₹, PD multiplier", "93 readable warning rules, rule-count score"],
   ["Validation", "Chronological hold-out, 40-origin backtest, residual tests, interval coverage", "Unseen-borrower lift, Bonferroni tests, productive-rule filter"],
   ["Strength", "Links defaults to the economy; supports provisioning", "Explains decisions; finds risk inside low-PD bands"],
   ["Limitation", "Conditional on macro path; few observations", "Rules give reasons, not ranking (AUC 0.645–0.657)"]],
  "Method comparison. Their metrics are not comparable numerically and are not ranked against each other.",
  widths=[2.6, 6.6, 6.6], size=9)
P("Synthesis: the methods are complementary. Within PD band A, applicants with six or more rules default at 5.6% against "
  "1.8% with none, and the same ordering holds in bands C and D, so rules add information the PD lacks there. Band B is the "
  "exception (7.0% with up to two rules, 4.7% with six or more) and band E is not monotonic, so the rules are a review aid "
  "rather than a replacement for the PD. The time-series multiplier makes the same break-even rule approve "
  "75.1% of applicants in the baseline 2025 economy and 47.8% in a severe one, while the approved default rate falls from "
  "6.10% to 3.98%. The macro-conditioned rules join the two: in downturns the dangerous flags are behavioural.")
H("6.2 Comparison with existing studies", 2)
P("Our test AUC (0.738 tuned XGBoost; 0.741 calibrated LightGBM) is slightly above the range reported by Berg et al. [1] (0.683–0.736) and "
  "below Óskarsdóttir et al. [2] (up to 0.923). Direct comparison is not valid: their labels are real repayments, ours are "
  "generated with a known ceiling of 0.781; their features include bureau scores or call records that Indian rules forbid for "
  "app lenders [15]. Our ranking of ensembles above logistic regression agrees with Lessmann et al. [4]. On the macro side, the "
  "signs we find on real data match the cross-country evidence of [5, 6]. Our distinct contributions are the link from a macro "
  "forecast to an approval rule, out-of-sample validated rules, and conformal intervals. Remaining gaps: no real micro-loan "
  "repayment labels and no live deployment. Full benchmark tables are in research/BENCHMARK_TABLE.md.")
H("6.3 KPI-based business impact and scenarios", 2)
T(["Scenario (2025)", "PD multiplier", "Approval", "Default of approved", "Expected credit loss", "vs baseline"],
  [["Neutral economy", "1.00", "60.3%", "5.09%", "—", "—"],
   ["Baseline", "0.57", "75.1%", "6.10%", "₹158.9 lakh", "—"],
   ["Adverse (unemployment +1.5 pp, GDP −3 pp)", "0.69", "70.7%", "5.73%", "₹194.6 lakh", "+₹35.6 lakh"],
   ["Severe (COVID-style)", "1.53", "47.8%", "3.98%", "₹432.5 lakh", "+₹273.6 lakh"]],
  "Scenario impact. Approvals are measured on the 6,000 test applicants; credit losses are model forecasts.",
  widths=[4.8, 2.0, 1.8, 2.4, 2.6, 2.4], size=9)
P("Assumptions: LGD 65%, cost of funds 9%, average exposure at default ₹111,589, book growth 1.38% per month. Measured "
  "outcomes (approval and default rates on the test set) are separated from forecasts (2025 losses). Sensitivity: the spread "
  "between baseline and severe losses (₹273.6 lakh) is the capital the lender should be able to absorb; the result depends on "
  "the macro path, the LGD assumption and the simulation's COVID response, which real Indian data suggest overstates "
  "short-run defaults when regulators grant forbearance. Priorities by impact and feasibility: (1) two-tier decision with "
  "rules, (2) monthly macro multiplier, (3) loan-size caps, (4) selective downturn tightening, (5) provisioning overlay.")

# =========================================================================================== 7
H("7. Interactive Application, Figure Index and Reproducibility", 1)
H("7.1 Dashboard", 2)
P("Platform: Streamlit (dashboard/app.py), run locally with streamlit run dashboard/app.py; a standalone Plotly report is in "
  "reports/interactive_dashboard.html. Six tabs: Underwriter decision engine (live form, PD, point-in-time PD, break-even, "
  "TreeSHAP reasons, triggered rules, counterfactual path to approval), Dataset explorer (filters by band, location, platform, "
  "outcome and rules hit, with dynamic KPIs and charts), Portfolio outlook (scenario forecasts and ₹ losses), Warning rules "
  "(lift and coverage sliders), Real-data reality check (India NPL forecasts, conformal band, HMDA rule explorer, fidelity "
  "scorecard) and Model card. Computed live: features, PD, SHAP, decision, rules and recourse; precomputed: model, rules, "
  "scenario forecasts and notebook-09 results.")
P("Tests: Streamlit AppTest renders every tab without exceptions, including the severe scenario with the high-risk preset, "
  "which exercises the recourse search. Known limitations: the recourse search uses a fixed grid (1,134 combinations); the "
  "dashboard is not deployed publicly. [Insert screenshots of the main state and a filtered state from reports/screenshots/.]")
H("7.2 Figure index and reproduction map", 2)
T(["Figure", "Caption", "Reproduce with"], [[str(n), c, s] for n, c, s in FIGS],
  "Figure index.", widths=[1.4, 8.6, 6.0], size=8)
H("7.3 Code and environment", 2)
P("Repository: https://github.com/Goldmauler/intellicredit-review2 (README with project tree, requirements.txt, Python 3.12). "
  "Reproduce: python data_generation.py; python -m src.macro_api; python -m src.portfolio_simulation; python -m src.real_data; "
  "execute notebooks 01–09 in order; python -m src.underwriting_model; streamlit run dashboard/app.py; python "
  "scripts/build_final_report.py. Folders: data/ (raw, processed, external, portfolio, real), notebooks/, src/, models/, "
  "dashboard/, reports/, research/. No secrets are stored; all APIs used are keyless. Seeds: 42 (data, models), 2026 (loan book).")

# =========================================================================================== 8
H("8. End-to-End Validation, Traceability and Contributions", 1)
H("8.1 Completion and validity", 2)
T(["Objective", "Evidence", "Status"],
  [["Score thin-file borrowers", "notebooks 03–05; Table 3; AUC 0.741, calibrated", "met"],
   ["Forecast portfolio defaults", "notebook 06; Table 5; MAE 0.0545 pp", "met"],
   ["Find validated warning patterns", "notebook 07; Table 6; 93 rules, 100% valid", "met"],
   ["Combine into decisions", "notebook 08; Table 10; dashboard", "met"],
   ["Respond to the 'use real data' feedback", "notebook 09; Tables 7–8; 5 real datasets", "met"],
   ["Real micro-loan repayment labels", "not legally obtainable", "not met (future work)"]],
  "Objective-to-evidence traceability matrix.", widths=[5.0, 8.0, 3.0], size=9)
P("Threats to validity: borrower labels are synthetic, so absolute performance is bounded by our generator; the macro "
  "mechanism was simulated and is only externally supported by the real panel; the real datasets are not micro-loans; the "
  "short monthly series limits model complexity. Reproducibility: all code is seeded and notebooks re-execute with identical "
  "model outputs (checked on 6 Nov 2026).")
H("8.2 Project tracking and individual work", 2)
P("Board: GitHub issues on the team repository (19 tasks: 10 Review 1, 9 Review 2) with stages Backlog, To Do, In Progress, "
  "Review/Testing and Completed; task history is in reports/project_management_and_contributions.md. [Insert board screenshot.]")
T(["Member", "Main contribution", "Evidence"],
  [["Darun R G", "Data generator, Review 1 pipeline, dataset validation, project board", "data_generation.py, notebooks 01–05, DATASET_VALIDATION_SUMMARY.md"],
   ["Vimalharihar S K (Goldmauler)", "Method 1 and Method 2, macro APIs, real-data track, recourse", "notebooks 06, 07, 09; src/macro_api.py, real_data.py, recourse.py"],
   ["Abhinav Juturu", "EDA, loan-book simulation, Method 1 data preparation", "notebook 02, src/portfolio_simulation.py"],
   ["Gowtham Reddy", "Cleaning, Streamlit dashboard", "notebook 01, dashboard/app.py"],
   ["Adithya", "Evaluation, combined insights, report and slides", "notebooks 05, 08, reports/"]],
  "Individual contributions (as recorded in the README; confirm before submission). Individual marks may vary with participation and explanation.",
  widths=[3.6, 6.4, 6.0], size=9)
H("8.3 AI use and originality declarations", 2)
P("[To be completed and signed by the team.] Generative-AI tools were used: Claude (Anthropic), for drafting code for the "
  "real-data collectors, notebook 09, the recourse module, the dashboard tab and this report draft, for checking documents "
  "against notebook outputs, and for collecting and verifying literature. All generated code was executed and its outputs "
  "were checked against the saved results; numbers in this report were read from those outputs. [List any other tools and "
  "the parts each member wrote or verified.] Turnitin similarity: [__ %] (limit 15%).")

# =========================================================================================== 9
H("9. Conclusion, Recommendations and Roadmap", 1)
P("All five objectives in Table 1 that could be met legally were met (Table 12). The engine ranks thin-file borrowers close to "
  "the data's ceiling, forecasts portfolio defaults better than six alternatives, produces rules that hold on unseen borrowers, "
  "and both advanced methods carried over to real data.")
T(["Recommendation", "Expected KPI effect", "Owner", "Feasibility"],
  [["Two-tier decision: auto-approve bands A–B with < 3 rules", "Catches 34% of defaults in 14% of applicants for review", "Credit policy", "High"],
   ["Monthly macro multiplier from the forecast", "Approval adjusts 75% → 48% between baseline and severe", "Risk analytics", "High"],
   ["Loan-to-income cap for tenure < 6 months or falling income", "Fewer early defaults among new workers", "Product", "High"],
   ["Tighten on behaviour, not demographics, in downturns", "Protects inclusion of low-income applicants", "Credit policy", "Medium"],
   ["Provision baseline ECL + adverse overlay (~₹36 lakh)", "Capital ready for a moderate downturn", "Finance", "High"],
   ["Track regulatory regimes in any India NPL forecast", "India NPL MAE 0.48 vs 0.91 pp without regimes", "Risk analytics", "High"]],
  "Prioritised recommendations.", widths=[5.6, 5.0, 2.6, 2.4], size=9)
P("Roadmap. Immediate: merge the real-data track and present it. Medium term: add a forbearance regime to the loan-book "
  "simulation and raise the weight of debt burden in the generator; report conformal intervals on the portfolio forecast. "
  "Longer term: a consented gig-worker survey pilot to compare income volatility and BNPL use with the generator, and testing "
  "the rules and model on a partner lender's repayment data. Success checks: rule lift stays ≥ 1.3 on new data and PD "
  "calibration error stays below one percentage point.")

# =========================================================================================== REFERENCES
H("References", 1)
REFS = [
    "Tobias Berg, Valentin Burg, Ana Gombović, and Manju Puri. 2020. On the Rise of FinTechs: Credit Scoring Using Digital Footprints. The Review of Financial Studies 33, 7, 2845–2897. https://doi.org/10.1093/rfs/hhz099",
    "María Óskarsdóttir, Cristián Bravo, Carlos Sarraute, Jan Vanthienen, and Bart Baesens. 2019. The value of big data for credit scoring: Enhancing financial inclusion using mobile phone data and social network analytics. Applied Soft Computing 74, 26–39. https://doi.org/10.1016/j.asoc.2018.10.004",
    "Sumit Agarwal, Shashwat Alok, Pulak Ghosh, and Sudip Gupta. 2020. Financial Inclusion and Alternate Credit Scoring for the Millennials: Role of Big Data and Machine Learning in Fintech. Working paper.",
    "Stefan Lessmann, Bart Baesens, Hsin-Vonn Seow, and Lyn C. Thomas. 2015. Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. European Journal of Operational Research 247, 1, 124–136. https://doi.org/10.1016/j.ejor.2015.05.030",
    "Roland Beck, Petr Jakubík, and Anamaria Piloiu. 2015. Key Determinants of Non-performing Loans: New Evidence from a Global Sample. Open Economies Review 26, 3, 525–550. https://doi.org/10.1007/s11079-015-9358-8",
    "Mwanza Nkusu. 2011. Nonperforming Loans and Macrofinancial Vulnerabilities in Advanced Economies. IMF Working Paper WP/11/161.",
    "Rakesh Agrawal and Ramakrishnan Srikant. 1994. Fast Algorithms for Mining Association Rules in Large Databases. In Proceedings of the 20th International Conference on Very Large Data Bases (VLDB '94), 487–499.",
    "Jiawei Han, Jian Pei, and Yiwen Yin. 2000. Mining frequent patterns without candidate generation. In Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data, 1–12. https://doi.org/10.1145/342009.335372",
    "Geoffrey I. Webb. 2007. Discovering Significant Patterns. Machine Learning 68, 1, 1–33. https://doi.org/10.1007/s10994-007-5006-x",
    "Anastasios N. Angelopoulos and Stephen Bates. 2021. A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. arXiv:2107.07511.",
    "Isaac Gibbs and Emmanuel Candès. 2021. Adaptive Conformal Inference Under Distribution Shift. In Advances in Neural Information Processing Systems 34 (NeurIPS 2021).",
    "Sandra Wachter, Brent Mittelstadt, and Chris Russell. 2018. Counterfactual Explanations without Opening the Black Box: Automated Decisions and the GDPR. Harvard Journal of Law & Technology 31, 2.",
    "Berk Ustun, Alexander Spangher, and Yang Liu. 2019. Actionable Recourse in Linear Classification. In Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT* '19), 10–19. https://doi.org/10.1145/3287560.3287566",
    "Government of India. 2005. The Credit Information Companies (Regulation) Act, 2005 (Act No. 30 of 2005).",
    "Reserve Bank of India. 2025. Reserve Bank of India (Digital Lending) Directions, 2025. RBI/2025-26/36.",
    "NITI Aayog. 2022. India's Booming Gig and Platform Economy: Perspectives and Recommendations on the Future of Work. New Delhi.",
    "Ministry of Finance, Government of India. 2026. Economic Survey 2025-26.",
    "Scott M. Lundberg and Su-In Lee. 2017. A Unified Approach to Interpreting Model Predictions. In Advances in Neural Information Processing Systems 30 (NIPS 2017), 4765–4774.",
    "Government of India. 2023. The Digital Personal Data Protection Act, 2023 (Act No. 22 of 2023).",
    "C. W. J. Granger. 1969. Investigating Causal Relations by Econometric Models and Cross-spectral Methods. Econometrica 37, 3, 424–438. https://doi.org/10.2307/1912791",
    "World Bank. 2022. The Global Findex Database 2021. Washington, DC.",
    "Robert Bartlett, Adair Morse, Richard Stanton, and Nancy Wallace. 2022. Consumer-lending discrimination in the FinTech Era. Journal of Financial Economics 143, 1, 30–56. https://doi.org/10.1016/j.jfineco.2021.05.047",
    "International Monetary Fund. 2026. Financial Soundness Indicators (FSIC), India, series AQ12_CFSI_PT. https://api.imf.org/external/sdmx/2.1/data/IMF.STA,FSIC/IND..",
    "World Bank. 2026. World Development Indicators API, series FB.AST.NPER.ZS, NY.GDP.MKTP.KD.ZG, SL.UEM.TOTL.ZS, FP.CPI.TOTL.ZG, FR.INR.LEND. https://api.worldbank.org/v2/",
    "Federal Reserve Bank of St. Louis. 2026. FRED: series CPALTT01INM659N, IRSTCI01INM156N, INDIRLTLT01STM, EXINUS, DRCLACBS, UNRATE, FEDFUNDS. https://fred.stlouisfed.org",
    "Consumer Financial Protection Bureau. 2026. HMDA Data Browser API, 2023 data for DE, RI and VT. https://ffiec.cfpb.gov/data-browser/",
    "Consumer Financial Protection Bureau. 2026. Consumer Complaint Database API. https://www.consumerfinance.gov/data-research/consumer-complaints/",
    "Skipper Seabold and Josef Perktold. 2010. statsmodels: Econometric and Statistical Modeling with Python. In Proceedings of the 9th Python in Science Conference.",
    "Sebastian Raschka. 2018. MLxtend: Providing machine learning and data science utilities and extensions to Python's scientific computing stack. Journal of Open Source Software 3, 24, 638.",
]
for i, r in enumerate(REFS, 1):
    p = doc.add_paragraph(f"[{i}] {r}")
    p.paragraph_format.left_indent = Cm(0.8)
    p.paragraph_format.first_line_indent = Cm(-0.8)
    p.paragraph_format.line_spacing = 1.15
    for run in p.runs:
        run.font.size = Pt(10.5)

H("Appendices", 1)
P("Appendix A: data dictionaries (data/data_dictionary.txt, data/data_dictionary_review2.md), generation settings "
  "(data_generation.py, seed 42) and provenance logs (data/external/macro_sources.json, data/real/real_sources.json).")
P("Appendix B: additional figures and tuning tables (notebooks 02–07; reports/figures/).")
P("Appendix C: installation and dashboard guide (README, How to Run).")
P("Appendix D: board evidence, contribution records, AI declaration and similarity report [to be attached].")

doc.save(OUT)
print("wrote", OUT, f"({FIGNO[0]} figures, {TABNO[0]} tables)")
