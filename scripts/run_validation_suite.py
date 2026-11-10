"""
IntelliCredit - Validation suite (Review 3)
===========================================
Reproducible evidence that (a) the pipeline works on REAL credit data, (b) the synthetic data behaves
like real credit data where that can be tested, and (c) where it does not.

Everything is recomputed from files; nothing is typed in by hand. Outputs go to models/validation/.

    python scripts/run_validation_suite.py            # full run (~10 min)
    python scripts/run_validation_suite.py --quick    # fewer repeats, for a smoke test

Real datasets (downloaded once from OpenML into data/real/benchmarks/, which is git-ignored):
    german    UCI Statlog German Credit              1,000 loans            30% bad
    taiwan    UCI Default of Credit Card Clients    30,000 card clients     22% default
    gmsc      Kaggle Give Me Some Credit           150,000 borrowers         6.7% distress
    heloc     FICO HELOC (xML Challenge)            10,459 home-equity lines 52% bad

Blocks:
  1  real-data benchmark   same 4 model types, 5-fold CV, on every dataset including the synthetic one
  2  model-ranking test    is the ordering of model types on synthetic data the same as on real data?
  3  effect-direction test does each risk driver push default the same way in real and synthetic data?
  4  SHAP top drivers      what does the model rely on, per dataset?
  5  time-series dynamics  persistence / volatility of the simulated default rate vs real NPL series
  6  income re-weighting   does the model still work if the borrower mix matches the real income spread?
  7  feature-group ablation + approvals at equal loss (conventional scorecard vs alternative data)
"""
import json
import os
import re
import sys
import time
import warnings
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.config import BORROWER_PROFILES_PATH, RANDOM_STATE, TEST_SIZE  # noqa: E402
from src.underwriting_model import break_even_pd, build_features  # noqa: E402

BM = os.path.join(ROOT, "data", "real", "benchmarks")
OUT = os.path.join(ROOT, "models", "validation")
os.makedirs(OUT, exist_ok=True)
QUICK = "--quick" in sys.argv
SEED = 42
FLAT = 0.03          # |Spearman rho| below this is called a negligible effect (std. error of rho at n=30,000 is about 0.006)


# ------------------------------------------------------------------------------------------------ loading
def _download():
    """Fetch the public datasets from OpenML once (cached under data/real/benchmarks)."""
    from sklearn.datasets import fetch_openml
    os.makedirs(BM, exist_ok=True)
    for key, did in {"german": 31, "taiwan": 42477, "gmsc": 46929, "heloc": 46932}.items():
        path = os.path.join(BM, f"{key}.csv.gz")
        if not os.path.exists(path):
            print(f"  downloading {key} (OpenML {did}) ...", flush=True)
            fetch_openml(data_id=did, as_frame=True, parser="auto").frame.to_csv(path, index=False, compression="gzip")


TAIWAN_COLS = (["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
               + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)])


def load_datasets():
    """name -> dict(X model-ready, raw for effect tests, y, meta)."""
    _download()
    ds = {}

    d = pd.read_csv(os.path.join(BM, "german.csv.gz"))
    y = (d["class"] == "bad").astype(int)
    raw = d.drop(columns="class")
    ds["german"] = dict(raw=raw, X=pd.get_dummies(raw, dtype=float), y=y, label="German credit (UCI Statlog)",
                        country="Germany", product="Instalment loans", year="1994", source="UCI / OpenML 31",
                        target="class = bad")

    d = pd.read_csv(os.path.join(BM, "taiwan.csv.gz"))
    d.columns = TAIWAN_COLS + ["y"]
    y = d.pop("y").astype(int)
    raw = d.copy()
    raw["UTILISATION"] = raw["BILL_AMT1"] / raw["LIMIT_BAL"].clip(lower=1)
    raw["PAID_SHARE"] = (raw[[f"PAY_AMT{i}" for i in range(1, 7)]].sum(axis=1)
                         / raw[[f"BILL_AMT{i}" for i in range(1, 7)]].clip(lower=0).sum(axis=1).clip(lower=1)).clip(0, 3)
    ds["taiwan"] = dict(raw=raw, X=d, y=y, label="Taiwan credit cards (UCI)", country="Taiwan", product="Credit cards",
                        year="2005", source="UCI / OpenML 42477", target="default next month")

    d = pd.read_csv(os.path.join(BM, "gmsc.csv.gz"))
    y = (d.pop("FinancialDistressNextTwoYears") == "Yes").astype(int)
    ds["gmsc"] = dict(raw=d.copy(), X=d, y=y, label="Give Me Some Credit (Kaggle)", country="United States",
                      product="Consumer credit", year="2011", source="Kaggle / OpenML 46929",
                      target="90+ days late within 2 years")

    d = pd.read_csv(os.path.join(BM, "heloc.csv.gz"))
    y = (d.pop("RiskPerformance") == "Bad").astype(int)
    d = d.replace({-9: np.nan, -8: np.nan, -7: np.nan})
    ds["heloc"] = dict(raw=d.copy(), X=d, y=y, label="FICO HELOC (xML Challenge)", country="United States",
                       product="Home-equity lines", year="2018", source="FICO / OpenML 46932",
                       target="90+ days late within 2 years")

    b = pd.read_csv(BORROWER_PROFILES_PATH)
    X = build_features(b)
    ds["intellicredit"] = dict(raw=pd.concat([b, X.add_prefix("f_")], axis=1), X=X, y=b["loan_default_status"].astype(int),
                               label="IntelliCredit (synthetic)", country="India (simulated)", product="Gig-worker micro-loans",
                               year="2018-2024", source="data_generation.py", target="90+ days past due")
    for d in ds.values():          # XGBoost / LightGBM reject [ ] < in feature names (German dummies contain them)
        d["X"].columns = [re.sub(r"[^0-9a-zA-Z_]+", "_", str(c)).strip("_") for c in d["X"].columns]
    return ds


# ------------------------------------------------------------------------------------------------ models
class Winsor(BaseEstimator, TransformerMixin):
    """Clip each column to its 1st-99th training percentile (what a scorecard analyst does before binning)."""
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_, self.hi_ = np.nanpercentile(X, 1, axis=0), np.nanpercentile(X, 99, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def make_models(n):
    mcs = int(min(80, max(10, n * 0.004)))        # IntelliCredit uses 80 on ~24k rows (0.33%); scaled for tiny sets
    leaf = int(min(50, max(5, n * 0.002)))
    return {
        "Logistic scorecard": make_pipeline(SimpleImputer(strategy="median"), Winsor(), StandardScaler(),
                                            LogisticRegression(max_iter=3000, C=1.0)),
        "Random forest": make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestClassifier(n_estimators=200, min_samples_leaf=leaf, max_features="sqrt",
                                                              n_jobs=-1, random_state=SEED)),
        "XGBoost": xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                                     min_child_weight=max(1, mcs // 8), eval_metric="logloss", n_jobs=-1, random_state=SEED,
                                     tree_method="hist"),
        "LightGBM (IntelliCredit recipe)": lgb.LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=15,
                                                              min_child_samples=mcs, subsample=0.8, subsample_freq=1,
                                                              colsample_bytree=0.8, reg_lambda=2.0, random_state=SEED,
                                                              verbose=-1, n_jobs=-1),
    }


def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def ece(y, p, bins=10):
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "q": q.values})
    g = df.groupby("q").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((g["n"] / g["n"].sum() * (g["y"] - g["p"]).abs()).sum())


def top_lifts(y, p):
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)}).sort_values("p", ascending=False).reset_index(drop=True)
    base = df["y"].mean()
    n = len(df)
    top10 = df["y"].iloc[: n // 10].mean() / base
    top_q, bot_q = df["y"].iloc[: n // 5].mean(), df["y"].iloc[-(n // 5):].mean()
    return float(top10), float(top_q / max(bot_q, 1e-9))


# ------------------------------------------------------------------------------------------------ block 1
def block1_benchmark(ds):
    print("[1] real-data benchmark", flush=True)
    rows, calib, meta = [], [], []
    for key, d in ds.items():
        X, y = d["X"], d["y"]
        n = len(y)
        repeats = 1 if QUICK else (5 if n <= 2000 else 2 if n <= 12000 else 1)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=SEED)
        folds = {m: [] for m in make_models(n)}
        oof = {m: np.zeros(n) for m in folds}
        t0 = time.time()
        for fi, (tr, te) in enumerate(cv.split(X, y)):
            for name, model in make_models(n).items():
                model.fit(X.iloc[tr], y.iloc[tr])
                p = model.predict_proba(X.iloc[te])[:, 1]
                yt = y.iloc[te]
                folds[name].append(dict(auc=roc_auc_score(yt, p), ks=ks_stat(yt, p), brier=brier_score_loss(yt, p),
                                        prauc=average_precision_score(yt, p), ece=ece(yt, p)))
                if fi < 5:                                     # first repeat only -> out-of-fold predictions
                    oof[name][te] = p
        for name, fl in folds.items():
            f = pd.DataFrame(fl)
            top10, q_ratio = top_lifts(y, oof[name])
            rows.append(dict(dataset=key, label=d["label"], model=name, n=n, folds=len(f),
                             auc=f.auc.mean(), auc_sd=f.auc.std(), ks=f.ks.mean(), brier=f.brier.mean(), prauc=f.prauc.mean(),
                             ece=f.ece.mean(), gini=2 * f.auc.mean() - 1, top_decile_lift=top10, quintile_ratio=q_ratio))
        lg = oof["LightGBM (IntelliCredit recipe)"]
        c = pd.DataFrame({"y": y.values, "p": lg})
        c["decile"] = pd.qcut(c["p"].rank(method="first"), 10, labels=False)
        for dec, g in c.groupby("decile"):
            calib.append(dict(dataset=key, decile=int(dec) + 1, predicted=g.p.mean(), observed=g.y.mean()))
        meta.append(dict(dataset=key, label=d["label"], country=d["country"], product=d["product"], year=d["year"], n=n,
                         features=X.shape[1], default_rate=float(y.mean()), target=d["target"], source=d["source"]))
        print(f"    {key:14s} n={n:>7,}  {time.time() - t0:5.0f}s  "
              + "  ".join(f"{r['model'].split(' ')[0]}={r['auc']:.3f}" for r in rows[-4:]), flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "realbench_results.csv"), index=False)
    pd.DataFrame(calib).to_csv(os.path.join(OUT, "realbench_calibration.csv"), index=False)
    pd.DataFrame(meta).to_csv(os.path.join(OUT, "realbench_meta.csv"), index=False)
    return res


# ------------------------------------------------------------------------------------------------ block 2
def block2_ranking(res):
    print("[2] model-ranking concordance", flush=True)
    piv = res.pivot(index="model", columns="dataset", values="auc")
    out = []
    for k in [c for c in piv.columns if c != "intellicredit"]:
        rho, _ = spearmanr(piv["intellicredit"], piv[k])
        out.append(dict(real_dataset=k, spearman_model_ranking=float(rho),
                        best_on_real=piv[k].idxmax(), best_on_synthetic=piv["intellicredit"].idxmax(),
                        lr_to_best_gap_real=float(piv[k].max() - piv.loc["Logistic scorecard", k]),
                        lr_to_best_gap_synth=float(piv["intellicredit"].max() - piv.loc["Logistic scorecard", "intellicredit"])))
    pd.DataFrame(out).to_csv(os.path.join(OUT, "model_ranking_concordance.csv"), index=False)
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------ block 3
def _effect(x, y):
    x = pd.Series(np.asarray(x, dtype=float))
    y = pd.Series(np.asarray(y))
    ok = x.notna() & np.isfinite(x)
    x, y = x[ok], y[ok]
    if x.nunique() < 2:
        return 0.0, 0.5
    rho = float(spearmanr(x, y)[0])
    return rho, float(roc_auc_score(y, x))


CONCEPTS = [
    # concept, expected sign of "higher value -> higher default", synthetic column, [(dataset, column, readable)]
    ("Younger borrowers default more", "-", "age",
     [("gmsc", "age", "Age"), ("taiwan", "AGE", "Age"), ("german", "age", "Age")]),
    ("Higher income / credit capacity lowers default", "-", "avg_monthly_income",
     [("gmsc", "MonthlyIncome", "Monthly income"), ("taiwan", "LIMIT_BAL", "Credit limit (income proxy)")]),
    ("More dependents raise default", "+", "num_dependents",
     [("gmsc", "NumberOfDependents", "Dependents"), ("german", "num_dependents", "Dependents")]),
    ("Late payments predict default (count / severity)", "+", "avg_utility_delay_days",
     [("gmsc", "NumberOfTimes90DaysLate", "Times 90+ days late"), ("taiwan", "PAY_0", "Latest repayment delay"),
      ("heloc", "NumTrades90Ever2DerogPubRec", "Trades ever 90+ late")]),
    ("A high on-time payment share protects", "-", "f_utility_reliability_score",
     [("heloc", "PercentTradesNeverDelq", "% trades never delinquent"), ("taiwan", "PAID_SHARE", "Share of bills repaid")]),
    ("Heavy debt burden raises default", "+", "f_debt_to_inflow_ratio",
     [("gmsc", "RevolvingUtilizationOfUnsecuredLines", "Revolving utilisation"), ("gmsc", "DebtRatio", "Debt ratio"),
      ("heloc", "NetFractionRevolvingBurden", "Revolving burden"), ("taiwan", "UTILISATION", "Card utilisation"),
      ("german", "installment_commitment", "Instalment / income")]),
    ("A bigger loan relative to means raises default", "+", "f_loan_to_income",
     [("german", "credit_amount", "Credit amount"), ("german", "duration", "Duration (months)")]),
    ("A longer track record lowers default", "-", "gig_tenure_months",
     [("heloc", "AverageMInFile", "Average months on file"), ("heloc", "MSinceOldestTradeOpen", "Months since oldest trade")]),
    ("A savings / repayment buffer lowers default", "-", "f_savings_buffer_ratio",
     [("taiwan", "PAID_SHARE", "Share of bills repaid")]),
]
UNTESTABLE = [
    ("Income instability raises default", "+", "f_income_volatility_index",
     "No public real credit dataset records income volatility."),
    ("Many recent credit enquiries raise default", "+", None,
     "HELOC shows it (NumInqLast6M), but the synthetic data has no enquiry feature."),
]


def block3_effects(ds):
    print("[3] effect-direction concordance", flush=True)
    syn = ds["intellicredit"]
    rows = []
    for concept, expected, scol, reals in CONCEPTS:
        s_rho, s_auc = _effect(syn["raw"][scol], syn["y"])
        for dkey, col, nice in reals:
            r_rho, r_auc = _effect(ds[dkey]["raw"][col], ds[dkey]["y"])
            rows.append(dict(concept=concept, expected=expected, synthetic_feature=scol.replace("f_", ""), synthetic_rho=s_rho,
                             synthetic_auc=s_auc, real_dataset=dkey, real_feature=nice, real_rho=r_rho, real_auc=r_auc,
                             real_matches_expected=bool(np.sign(r_rho) == (1 if expected == "+" else -1)),
                             synthetic_matches_expected=bool(np.sign(s_rho) == (1 if expected == "+" else -1) and abs(s_rho) >= FLAT),
                             synthetic_flat=bool(abs(s_rho) < FLAT)))
    eff = pd.DataFrame(rows)
    eff.to_csv(os.path.join(OUT, "effect_concordance_detail.csv"), index=False)

    summ = []
    for concept, expected, scol, reals in CONCEPTS:
        g = eff[eff.concept == concept]
        sign = 1 if expected == "+" else -1
        s_rho = g.synthetic_rho.iloc[0]
        real_ok = int(g.real_matches_expected.sum())
        if abs(s_rho) < FLAT:
            verdict = "Negligible in synthetic data"
        elif np.sign(s_rho) == sign:
            verdict = "Same direction" if real_ok == len(g) else "Same direction (real mixed)"
        else:
            verdict = "Opposite direction"
        summ.append(dict(concept=concept, expected_direction="higher value, higher default" if expected == "+" else "higher value, lower default",
                         synthetic_feature=scol.replace("f_", ""), synthetic_rho=s_rho,
                         real_checks=len(g), real_agree_with_expected=real_ok,
                         real_rho_range=f"{g.real_rho.min():+.2f} to {g.real_rho.max():+.2f}", verdict=verdict))
    for concept, expected, scol, note in UNTESTABLE:
        s_rho = _effect(syn["raw"][scol], syn["y"])[0] if scol else np.nan
        summ.append(dict(concept=concept, expected_direction="higher value, higher default", synthetic_feature=(scol or "").replace("f_", ""),
                         synthetic_rho=s_rho, real_checks=0, real_agree_with_expected=0, real_rho_range="n/a", verdict="Cannot be tested: " + note))
    pd.DataFrame(summ).to_csv(os.path.join(OUT, "effect_concordance.csv"), index=False)
    return pd.DataFrame(summ)


# ------------------------------------------------------------------------------------------------ block 4
def block4_shap(ds):
    print("[4] SHAP top drivers", flush=True)
    rows = []
    for key, d in ds.items():
        X, y = d["X"], d["y"]
        m = make_models(len(y))["LightGBM (IntelliCredit recipe)"].fit(X, y)
        contrib = m.predict(X.sample(min(len(X), 20000), random_state=SEED), pred_contrib=True)[:, :-1]
        imp = pd.Series(np.abs(contrib).mean(axis=0), index=X.columns)
        imp = (imp / imp.sum()).sort_values(ascending=False)
        for rank, (f, v) in enumerate(imp.head(6).items(), 1):
            rows.append(dict(dataset=key, rank=rank, feature=f, share_of_total=float(v)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "shap_top_drivers.csv"), index=False)
    return out


# ------------------------------------------------------------------------------------------------ block 5
def _dyn(series):
    s = pd.Series(series).astype(float)
    s = s[s > 0]
    d = np.log(s).diff().dropna()
    return dict(n=int(len(s)), level_ar1=float(s.autocorr(1)), change_sd=float(d.std()), change_ar1=float(d.autocorr(1)),
                peak_to_median=float(s.max() / s.median()), cv=float(s.std() / s.mean()))


def block5_dynamics():
    print("[5] time-series dynamics", flush=True)
    pm = pd.read_csv(os.path.join(ROOT, "data", "portfolio", "portfolio_monthly.csv"), parse_dates=["date"])
    synq = pm.set_index("date")["default_rate"].resample("QS").mean()
    npl = pd.read_csv(os.path.join(ROOT, "data", "real", "india_bank_npl_quarterly.csv"), parse_dates=["quarter_start"]).set_index("quarter_start")["npl_ratio_pct"]
    us = pd.read_csv(os.path.join(ROOT, "data", "real", "us_consumer_credit_quarterly.csv"), parse_dates=["quarter_start"]).set_index("quarter_start")
    window = slice("2019-01-01", "2024-12-31")
    rows = [dict(series="Simulated portfolio default rate (monthly, 2019-24)", **_dyn(pm["default_rate"])),
            dict(series="Simulated portfolio default rate (quarterly mean)", **_dyn(synq)),
            dict(series="India bank NPL ratio (IMF, quarterly 2010-25)", **_dyn(npl)),
            dict(series="India bank NPL ratio (IMF, 2019-24 only)", **_dyn(npl.loc[window])),
            dict(series="US consumer-loan delinquency (Fed, quarterly 1987-2026)", **_dyn(us["consumer_loan_delinquency_pct"])),
            dict(series="US credit-card delinquency (Fed, quarterly 1987-2026)", **_dyn(us["credit_card_delinquency_pct"]))]
    tab = pd.DataFrame(rows)
    # pure sampling noise: the book holds ~1,200 loans, so monthly default counts are small and noisy
    nd = pm.loc[pm["new_defaults"] > 0, "new_defaults"].astype(float)
    tab["sampling_noise_change_sd"] = np.nan
    tab.loc[0, "sampling_noise_change_sd"] = float(np.sqrt(2 * np.mean(1 / nd)))
    tab.loc[0, "avg_active_loans"] = float(pm["active_loans"].mean())
    tab.to_csv(os.path.join(OUT, "dynamics.csv"), index=False)

    j = pd.concat([synq.rename("sim"), npl.rename("india")], axis=1).loc[window].dropna()
    pair = dict(overlap_quarters=int(len(j)), corr_level=float(j.sim.corr(j.india)),
                corr_change=float(np.log(j.sim).diff().corr(np.log(j.india).diff())),
                sim_2024_vs_2019=float(j.sim.iloc[-4:].mean() / j.sim.iloc[:4].mean()),
                india_2024_vs_2019=float(j.india.iloc[-4:].mean() / j.india.iloc[:4].mean()))
    json.dump(pair, open(os.path.join(OUT, "dynamics_pair.json"), "w"), indent=1)
    return tab, pair


# ------------------------------------------------------------------------------------------------ block 6
def block6_reweight():
    print("[6] income re-weighting", flush=True)
    import joblib
    model = joblib.load(os.path.join(ROOT, "models", "underwriting_model.joblib"))
    b = pd.read_csv(BORROWER_PROFILES_PATH)
    _, te = train_test_split(b.index, test_size=TEST_SIZE, stratify=b["loan_default_status"], random_state=RANDOM_STATE)
    t = b.loc[te].copy()
    t["pd"] = model.predict_proba(build_features(t))[:, 1]
    t["be"] = break_even_pd(t["interest_rate_annual"], t["loan_tenure_months"])
    y = t["loan_default_status"].values
    inc = t["avg_monthly_income"].values
    bins = np.digitize(inc, [15000, 30000, 50000])           # 0:<15k 1:15-30k 2:30-50k 3:50k+
    share = np.array([(bins == i).mean() for i in range(4)])

    def metrics(w):
        w = np.asarray(w, dtype=float)
        appr = t["pd"].values < t["be"].values
        return dict(low_income_share=float(w[bins == 0].sum() / w.sum()),
                    default_rate=float(np.average(y, weights=w)), mean_predicted_pd=float(np.average(t["pd"], weights=w)),
                    auc=float(roc_auc_score(y, t["pd"], sample_weight=w)),
                    approval_rate=float(np.average(appr, weights=w)),
                    default_rate_approved=float(np.average(y[appr], weights=w[appr])),
                    default_rate_declined=float(np.average(y[~appr], weights=w[~appr])))

    rows = [dict(scenario="As generated (15% earn under Rs 15k)", **metrics(np.ones(len(t))))]
    for target in (0.30, 0.40, 0.50):
        tgt = np.array([target] + list((1 - target) * share[1:] / share[1:].sum()))
        w = (tgt / share)[bins]
        ess = float(w.sum() ** 2 / (w ** 2).sum())
        rows.append(dict(scenario=f"Re-weighted so {target:.0%} earn under Rs 15k" + (" (Economic Survey 2025-26)" if target == 0.40 else ""),
                         effective_sample=ess, **metrics(w)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "income_reweighting.csv"), index=False)
    return out


# ------------------------------------------------------------------------------------------------ block 7
GROUPS = {
    "Application form only": ["age", "gig_tenure_months", "has_bank_account", "num_dependents", "location_code", "education_code",
                              "avg_monthly_income", "loan_to_income", "loan_tenure_months", "interest_rate_annual"],
    "Application form + past loans": None,    # filled below
    "Alternative data only": ["income_volatility_index", "income_trend_slope", "utility_reliability_score", "avg_utility_delay_days",
                              "max_utility_delay_days", "avg_days_between_recharges", "debt_to_inflow_ratio", "savings_buffer_ratio",
                              "avg_app_logins", "avg_upi_transactions"],
    "Everything (IntelliCredit)": None,
}
GROUPS["Application form + past loans"] = GROUPS["Application form only"] + ["previous_micro_loans", "repayment_ratio"]


def approvals_at_loss(y, p, target):
    """Largest share of applicants that can be approved (safest first) while the default rate of approved <= target."""
    o = np.argsort(p)
    cum = np.cumsum(np.asarray(y)[o]) / np.arange(1, len(y) + 1)
    ok = np.where(cum <= target)[0]
    return float((ok.max() + 1) / len(y)) if len(ok) else 0.0


def block7_ablation(ds):
    print("[7] feature-group ablation", flush=True)
    d = ds["intellicredit"]
    X, y = d["X"], d["y"]
    GROUPS["Everything (IntelliCredit)"] = list(X.columns)
    tr, te = train_test_split(X.index, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows, preds = [], {}
    for g, cols in GROUPS.items():
        for mname in ("Logistic scorecard", "LightGBM (IntelliCredit recipe)"):
            aucs = []
            for a, b in skf.split(X, y):
                m = make_models(len(y))[mname].fit(X.iloc[a][cols], y.iloc[a])
                aucs.append(roc_auc_score(y.iloc[b], m.predict_proba(X.iloc[b][cols])[:, 1]))
            m = make_models(len(y))[mname].fit(X.loc[tr, cols], y.loc[tr])
            p = m.predict_proba(X.loc[te, cols])[:, 1]
            preds[(g, mname)] = p
            rows.append(dict(feature_group=g, model=mname, n_features=len(cols), auc=float(np.mean(aucs)), auc_sd=float(np.std(aucs))))
    ab = pd.DataFrame(rows)
    ab.to_csv(os.path.join(OUT, "feature_group_ablation.csv"), index=False)

    yt = y.loc[te].values
    base = ("Application form + past loans", "Logistic scorecard")
    full = ("Everything (IntelliCredit)", "LightGBM (IntelliCredit recipe)")
    eq = []
    for tgt in (0.04, 0.05, 0.06, 0.08):
        a0, a1 = approvals_at_loss(yt, preds[base], tgt), approvals_at_loss(yt, preds[full], tgt)
        eq.append(dict(target_default_rate_of_approved=tgt, conventional_scorecard_approves=a0, intellicredit_approves=a1,
                       extra_approvals_pct=(a1 / a0 - 1) if a0 > 0 else np.nan))
    eqd = pd.DataFrame(eq)
    eqd.to_csv(os.path.join(OUT, "approvals_at_equal_loss.csv"), index=False)
    return ab, eqd


# ------------------------------------------------------------------------------------------------ main
def main():
    t0 = time.time()
    ds = load_datasets()
    res = block1_benchmark(ds)
    rk = block2_ranking(res)
    eff = block3_effects(ds)
    block4_shap(ds)
    dyn, pair = block5_dynamics()
    rw = block6_reweight()
    ab, eqd = block7_ablation(ds)
    json.dump(dict(run_at=datetime.now().isoformat(timespec="seconds"), quick=QUICK, seed=SEED,
                   seconds=round(time.time() - t0), lightgbm=lgb.__version__, xgboost=xgb.__version__, pandas=pd.__version__),
              open(os.path.join(OUT, "run_info.json"), "w"), indent=1)
    print(f"\nDone in {time.time() - t0:.0f}s. Outputs in {OUT}")
    print(res.pivot(index="model", columns="dataset", values="auc").round(3).to_string())
    print(rk.round(3).to_string())
    print(eff[["concept", "synthetic_rho", "real_checks", "real_agree_with_expected", "verdict"]].to_string())
    print(rw.round(3).to_string())
    print(ab.round(3).to_string())
    print(eqd.round(3).to_string())
    print(dyn.round(3).to_string())
    print(pair)


if __name__ == "__main__":
    main()
