"""
Phase 4: Model Estimation
=========================
Project : sl-remittance-model
Author  : (your name)
Purpose : Estimate four regression models explaining Sri Lanka worker
          remittance inflows using log-log OLS, crisis dummy, growth
          rate, and AR(1) specifications.

Output  : outputs/tables/regression_results.txt
          (all four model summaries in one readable file)

Models
------
1. Baseline OLS (log-log)
   ln_remittances = β0 + β1·ln_lkr_usd + β2·ln_brent + β3·ln_gcc_gdp + ε

2. Crisis Dummy OLS
   Same as (1) but adds crisis_period dummy and post_crisis dummy

3. Growth Rate OLS
   remittances_yoy_growth = β0 + β1·lkr_depreciation + β2·brent_yoy_change + ε

4. AR(1) — Autoregressive model
   ln_remittances(t) = β0 + β1·ln_remittances(t-1) + β2·ln_lkr_usd + β3·ln_brent + ε
"""

# ── 0. IMPORTS ──────────────────────────────────────────────────────────────
# pandas  : load and manipulate our data table
# numpy   : maths helpers (log, etc.)
# statsmodels : the econometrics engine — runs OLS regressions
# os      : lets us create folders if they don't exist yet
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats

# ── 1. PATHS ────────────────────────────────────────────────────────────────
# __file__  = the path of THIS script  (src/04_model_estimation.py)
# os.path.dirname(...) goes one folder up → src/
# os.path.join(..., '..') goes one more up → sl-remittance-model/
BASE_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'merged_dataset.csv')
OUT_DIR   = os.path.join(BASE_DIR, 'outputs', 'tables')
OUT_FILE  = os.path.join(OUT_DIR,  'regression_results.txt')

os.makedirs(OUT_DIR, exist_ok=True)   # create folder if it doesn't exist

# ── 2. LOAD DATA ─────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

print(f"Loaded {len(df)} rows  |  {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Columns: {list(df.columns)}\n")

# ── 3. HELPER FUNCTION ───────────────────────────────────────────────────────
def model_summary_text(title, result, extra_notes=""):
    """
    Build a human-readable text block for one regression result.
    We include:
      • Coefficient table (β, standard error, t-stat, p-value)
      • R², Adjusted R², F-stat
      • Durbin-Watson statistic  (checks for autocorrelation in residuals)
      • Breusch-Pagan p-value   (checks for heteroskedasticity)
    """
    sep  = "=" * 72
    sep2 = "-" * 72

    lines = []
    lines.append(sep)
    lines.append(f"  {title}")
    lines.append(sep)

    # ── Coefficient table ──
    lines.append(f"\n{'Variable':<22} {'Coef':>10} {'Std Err':>10} "
                 f"{'t-stat':>10} {'p-value':>10}  {'Sig':>4}")
    lines.append(sep2)

    for var in result.params.index:
        coef   = result.params[var]
        se     = result.bse[var]
        tstat  = result.tvalues[var]
        pval   = result.pvalues[var]
        # Significance stars: *** p<0.01  ** p<0.05  * p<0.1
        stars  = ("***" if pval < 0.01
                  else "**" if pval < 0.05
                  else "*"  if pval < 0.10
                  else "")
        lines.append(f"  {var:<20} {coef:>10.4f} {se:>10.4f} "
                     f"{tstat:>10.4f} {pval:>10.4f}  {stars:>4}")

    lines.append(sep2)

    # ── Fit statistics ──
    nobs   = int(result.nobs)
    r2     = result.rsquared
    r2adj  = result.rsquared_adj
    fstat  = result.fvalue
    fpval  = result.f_pvalue

    lines.append(f"\n  Observations : {nobs}")
    lines.append(f"  R²           : {r2:.4f}")
    lines.append(f"  Adjusted R²  : {r2adj:.4f}")
    lines.append(f"  F-statistic  : {fstat:.4f}  (p = {fpval:.4e})")

    # ── Durbin-Watson ──
    # Value near 2 = no autocorrelation (good)
    # Value < 1.5  = positive autocorrelation (residuals correlated across time)
    dw = durbin_watson(result.resid)
    lines.append(f"  Durbin-Watson: {dw:.4f}"
                 + ("  ← OK"             if 1.5 < dw < 2.5
                    else "  ← possible autocorrelation"))

    # ── Breusch-Pagan heteroskedasticity test ──
    # H0: residuals have constant variance (homoskedastic)
    # Low p-value → heteroskedastic → we should use robust std errors
    try:
        _, bp_pval, _, _ = het_breuschpagan(result.resid, result.model.exog)
        lines.append(f"  Breusch-Pagan p-value: {bp_pval:.4f}"
                     + ("  ← homoskedastic (p > 0.05)" if bp_pval > 0.05
                        else "  ← heteroskedastic (p ≤ 0.05) — use robust SE"))
    except Exception:
        lines.append("  Breusch-Pagan: could not compute")

    if extra_notes:
        lines.append(f"\n  Notes: {extra_notes}")

    lines.append("")   # blank line at end
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL 1 — BASELINE OLS (LOG-LOG)
# ══════════════════════════════════════════════════════════════════════════════
# Interpretation of log-log coefficients:
#   "A 1% increase in X is associated with a β% change in remittances."
#   These are called ELASTICITIES and are the standard in trade econometrics.
#
# sm.add_constant() adds the intercept term β0 to our X matrix.
# Without it, the regression line would be forced through the origin — wrong!

print("Estimating Model 1: Baseline OLS (log-log) ...")

# Drop rows where any of these columns are NaN (safety check)
m1_vars = ['ln_remittances', 'ln_lkr_usd', 'ln_brent', 'ln_gcc_gdp']
df_m1   = df[m1_vars].dropna()

Y1 = df_m1['ln_remittances']
X1 = sm.add_constant(df_m1[['ln_lkr_usd', 'ln_brent', 'ln_gcc_gdp']])

res1 = sm.OLS(Y1, X1).fit()   # .fit() runs the regression

notes1 = ("Log-log model. Coefficients are elasticities. "
          "Dependent variable: ln(remittances_usd_mn).")

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL 2 — CRISIS DUMMY OLS
# ══════════════════════════════════════════════════════════════════════════════
# We add two binary (0/1) dummy variables:
#   crisis_period = 1 for the 2022 acute crisis months
#   post_crisis   = 1 for all months after the crisis began
#
# The crisis dummy captures the LEVEL SHIFT in remittances during the crash.
# The post_crisis dummy captures the NEW REGIME that persists afterwards.

print("Estimating Model 2: Crisis Dummy OLS ...")

m2_vars = ['ln_remittances', 'ln_lkr_usd', 'ln_brent',
           'ln_gcc_gdp', 'crisis_period', 'post_crisis']
df_m2   = df[m2_vars].dropna()

Y2 = df_m2['ln_remittances']
X2 = sm.add_constant(df_m2[['ln_lkr_usd', 'ln_brent',
                              'ln_gcc_gdp', 'crisis_period', 'post_crisis']])

res2 = sm.OLS(Y2, X2).fit()

notes2 = ("Adds crisis_period and post_crisis dummies to baseline. "
          "Dummy coefficients show ln-point shift in remittances level.")

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL 3 — GROWTH RATE OLS
# ══════════════════════════════════════════════════════════════════════════════
# Instead of log-levels, we model YEAR-ON-YEAR GROWTH RATES.
# Question: does faster LKR depreciation / bigger oil price swings
# predict bigger changes in remittance flows?
#
# This model is stationary by construction (growth rates don't drift upward
# the way levels do), so it passes unit-root concerns automatically.

print("Estimating Model 3: Growth Rate OLS ...")

m3_vars = ['remittances_yoy_growth', 'lkr_depreciation', 'brent_yoy_change']
df_m3   = df[m3_vars].dropna()

Y3 = df_m3['remittances_yoy_growth']
X3 = sm.add_constant(df_m3[['lkr_depreciation', 'brent_yoy_change']])

res3 = sm.OLS(Y3, X3).fit()

notes3 = ("All variables in YoY % change form. "
          "Coefficients: how many pp change in remittance growth per 1pp "
          "change in LKR depreciation or oil price growth.")

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL 4 — AR(1) WITH COVARIATES
# ══════════════════════════════════════════════════════════════════════════════
# AR(1) = AutoRegressive lag-1 model.
# We add ln_remittances(t-1) as a regressor.
# This asks: once we know last month's remittances, how much additional
# explanatory power do oil prices and the exchange rate add?
#
# df['ln_rem_lag1'] = df['ln_remittances'].shift(1)
# .shift(1) moves the column down by 1 row, so row t gets the value from row t-1.
# Row 0 (first month) will be NaN — we drop it automatically via .dropna().

print("Estimating Model 4: AR(1) ...")

df['ln_rem_lag1'] = df['ln_remittances'].shift(1)

m4_vars = ['ln_remittances', 'ln_rem_lag1', 'ln_lkr_usd', 'ln_brent']
df_m4   = df[m4_vars].dropna()

Y4 = df_m4['ln_remittances']
X4 = sm.add_constant(df_m4[['ln_rem_lag1', 'ln_lkr_usd', 'ln_brent']])

res4 = sm.OLS(Y4, X4).fit()

notes4 = ("AR(1) specification. ln_rem_lag1 = ln_remittances(t-1). "
          "High AR coefficient indicates strong persistence / inertia.")


# ── 4. PRINT TO CONSOLE ───────────────────────────────────────────────────────
print("\n" + "─" * 72)
print("  REGRESSION RESULTS SUMMARY")
print("─" * 72 + "\n")

for title, result, notes in [
    ("MODEL 1 — Baseline OLS Log-Log",          res1, notes1),
    ("MODEL 2 — Crisis Dummy OLS",               res2, notes2),
    ("MODEL 3 — Growth Rate OLS",                res3, notes3),
    ("MODEL 4 — AR(1) with Covariates",          res4, notes4),
]:
    block = model_summary_text(title, result, notes)
    print(block)


# ── 5. SAVE TO FILE ──────────────────────────────────────────────────────────
header = (
    "REGRESSION RESULTS — sl-remittance-model\n"
    "Project: Determinants of Sri Lanka Worker Remittances 2009-2025\n"
    "Generated by: src/04_model_estimation.py\n"
    + "=" * 72 + "\n\n"
    "Significance codes:  *** p<0.01   ** p<0.05   * p<0.10\n\n"
)

with open(OUT_FILE, 'w', encoding='utf-8') as f:
    f.write(header)
    for title, result, notes in [
        ("MODEL 1 — Baseline OLS Log-Log",          res1, notes1),
        ("MODEL 2 — Crisis Dummy OLS",               res2, notes2),
        ("MODEL 3 — Growth Rate OLS",                res3, notes3),
        ("MODEL 4 — AR(1) with Covariates",          res4, notes4),
    ]:
        f.write(model_summary_text(title, result, notes))

print(f"\n✓ Results saved to: {OUT_FILE}")


# ── 6. KEY FINDINGS SUMMARY ──────────────────────────────────────────────────
# Print a short plain-English interpretation so you can QA the results quickly.

print("\n" + "═" * 72)
print("  QUICK INTERPRETATION GUIDE")
print("═" * 72)

print(f"""
MODEL 1 (Baseline log-log)
  R² = {res1.rsquared:.3f}  → the model explains {res1.rsquared*100:.1f}% of variance in ln(remittances)
  ln_lkr_usd  coef = {res1.params['ln_lkr_usd']:+.4f}
    → A 1% LKR depreciation is associated with a {res1.params['ln_lkr_usd']:+.2f}% change in remittances
  ln_brent    coef = {res1.params['ln_brent']:+.4f}
    → A 1% rise in oil price is associated with a {res1.params['ln_brent']:+.2f}% change in remittances
  ln_gcc_gdp  coef = {res1.params['ln_gcc_gdp']:+.4f}
    → A 1% rise in GCC GDP is associated with a {res1.params['ln_gcc_gdp']:+.2f}% change in remittances

MODEL 2 (Crisis dummies)
  R² = {res2.rsquared:.3f}  (vs {res1.rsquared:.3f} in Model 1  → {"improved" if res2.rsquared > res1.rsquared else "unchanged"})
  crisis_period coef = {res2.params['crisis_period']:+.4f}
    → During the 2022 crisis, remittances shifted by {res2.params['crisis_period']:+.4f} log-points
  post_crisis   coef = {res2.params['post_crisis']:+.4f}
    → After the crisis, the permanent regime shift = {res2.params['post_crisis']:+.4f} log-points

MODEL 3 (Growth rates)
  R² = {res3.rsquared:.3f}
  lkr_depreciation  coef = {res3.params['lkr_depreciation']:+.4f}
  brent_yoy_change   coef = {res3.params['brent_yoy_change']:+.4f}

MODEL 4 (AR1)
  R² = {res4.rsquared:.3f}
  ln_rem_lag1 coef = {res4.params['ln_rem_lag1']:+.4f}
    → {"Strong persistence — last month's remittances are highly predictive" if res4.params['ln_rem_lag1'] > 0.7 else "Moderate persistence"}
""")

print("✓ Phase 4 complete.")