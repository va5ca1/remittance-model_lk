"""
Phase 5: Chow Structural Break Test
=====================================
Project : sl-remittance-model
Purpose : Formally test whether regression coefficients changed after
          the April 2022 Sri Lanka sovereign debt crisis.

Tests performed
---------------
1. Chow F-test (manual)
   Splits sample at April 2022, compares pooled vs split-sample RSS.
   H0: coefficients are identical in both sub-periods ("no break")
   H1: at least one coefficient changed ("structural break exists")

2. Interaction Dummy Test (equivalent, cross-verification)
   Adds post_crisis x each variable to the regression.
   Significant interactions confirm coefficients shifted post-crisis.

3. CUSUM Test (visual)
   Plots cumulative sum of recursive residuals with 5% confidence bands.
   If the CUSUM line crosses the bands -> parameter instability at that date.

4. Sub-period coefficient comparison table
   Side-by-side pre vs post coefficients to show HOW they changed.

Output files
------------
  outputs/tables/chow_test_results.txt
  outputs/figures/chow_cusum_plot.png
"""

# ── 0. IMPORTS ────────────────────────────────────────────────────────────────
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from scipy import stats
from statsmodels.regression.recursive_ls import RecursiveLS

# ── 1. PATHS ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'merged_dataset.csv')
FIG_DIR   = os.path.join(BASE_DIR, 'outputs', 'figures')
TAB_DIR   = os.path.join(BASE_DIR, 'outputs', 'tables')
FIG_OUT   = os.path.join(FIG_DIR, 'chow_cusum_plot.png')
TAB_OUT   = os.path.join(TAB_DIR, 'chow_test_results.txt')

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# ── 2. LOAD & PREPARE DATA ────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

BREAK_DATE = pd.Timestamp('2022-04-01')

YVARS = 'ln_remittances'
XVARS = ['ln_lkr_usd', 'ln_brent', 'ln_gcc_gdp']

df_clean = df[['date', YVARS] + XVARS].dropna().reset_index(drop=True)

df_pre  = df_clean[df_clean['date'] <  BREAK_DATE].copy()
df_post = df_clean[df_clean['date'] >= BREAK_DATE].copy()

n_full = len(df_clean)
n_pre  = len(df_pre)
n_post = len(df_post)
k      = len(XVARS) + 1

print(f"Full sample : {n_full} observations  "
      f"({df_clean['date'].min().date()} -> {df_clean['date'].max().date()})")
print(f"Pre-crisis  : {n_pre}  observations  "
      f"({df_pre['date'].min().date()} -> {df_pre['date'].max().date()})")
print(f"Post-crisis : {n_post}  observations  "
      f"({df_post['date'].min().date()} -> {df_post['date'].max().date()})")
print(f"Parameters k = {k}  (intercept + {len(XVARS)} regressors)\n")

# ── 3. CHOW F-TEST ────────────────────────────────────────────────────────────
print("=" * 60)
print("  TEST 1: CHOW F-TEST")
print("=" * 60)

def run_ols(data):
    Y = data[YVARS]
    X = sm.add_constant(data[XVARS])
    res = sm.OLS(Y, X).fit()
    rss = np.sum(res.resid ** 2)
    return res, rss

res_full, rss_full = run_ols(df_clean)
res_pre,  rss_pre  = run_ols(df_pre)
res_post, rss_post = run_ols(df_post)

rss_unrestricted = rss_pre + rss_post
rss_restricted   = rss_full

df_numerator   = k
df_denominator = n_pre + n_post - 2 * k

chow_f = ((rss_restricted - rss_unrestricted) / df_numerator) / \
         (rss_unrestricted / df_denominator)

chow_p = 1 - stats.f.cdf(chow_f, df_numerator, df_denominator)

f_crit_05 = stats.f.ppf(0.95, df_numerator, df_denominator)
f_crit_01 = stats.f.ppf(0.99, df_numerator, df_denominator)

print(f"\n  RSS (full sample, restricted) : {rss_restricted:.4f}")
print(f"  RSS (pre-crisis sub-sample)   : {rss_pre:.4f}")
print(f"  RSS (post-crisis sub-sample)  : {rss_post:.4f}")
print(f"  RSS (unrestricted = pre+post) : {rss_unrestricted:.4f}")
print(f"\n  Degrees of freedom (numerator)  : {df_numerator}")
print(f"  Degrees of freedom (denominator): {df_denominator}")
print(f"\n  Chow F-statistic  : {chow_f:.4f}")
print(f"  p-value           : {chow_p:.4e}")
print(f"  Critical value 5% : {f_crit_05:.4f}")
print(f"  Critical value 1% : {f_crit_01:.4f}")

if chow_p < 0.01:
    chow_verdict = "REJECT H0 at 1% -- strong evidence of structural break at April 2022"
elif chow_p < 0.05:
    chow_verdict = "REJECT H0 at 5% -- evidence of structural break at April 2022"
elif chow_p < 0.10:
    chow_verdict = "REJECT H0 at 10% -- weak evidence of structural break"
else:
    chow_verdict = "FAIL TO REJECT H0 -- no significant structural break detected"

print(f"\n  Verdict: {chow_verdict}\n")


# ── 4. INTERACTION DUMMY TEST ─────────────────────────────────────────────────
print("=" * 60)
print("  TEST 2: INTERACTION DUMMY TEST")
print("=" * 60)

df_interact = df_clean.copy()
df_interact['post_crisis'] = (df_interact['date'] >= BREAK_DATE).astype(int)

for var in XVARS:
    df_interact[f'post_{var}'] = df_interact['post_crisis'] * df_interact[var]

interaction_cols = ['post_crisis'] + [f'post_{v}' for v in XVARS]

Y_i = df_interact[YVARS]
X_i = sm.add_constant(df_interact[XVARS + interaction_cols])

res_interact = sm.OLS(Y_i, X_i).fit()

print(f"\n  {'Variable':<25} {'Coef':>9} {'p-value':>10}  {'Sig':>4}")
print("  " + "-" * 55)
for var in res_interact.params.index:
    coef  = res_interact.params[var]
    pval  = res_interact.pvalues[var]
    stars = ("***" if pval < 0.01 else "**" if pval < 0.05
             else "*" if pval < 0.10 else "")
    print(f"  {var:<25} {coef:>9.4f} {pval:>10.4f}  {stars:>4}")

print(f"\n  R²           : {res_interact.rsquared:.4f}")
print(f"  Adjusted R²  : {res_interact.rsquared_adj:.4f}")

interaction_indices = [list(res_interact.params.index).index(c)
                       for c in interaction_cols]
f_test = res_interact.f_test(
    np.eye(len(res_interact.params))[interaction_indices]
)
print(f"\n  Joint F-test on interaction terms:")
print(f"  F = {float(f_test.fvalue):.4f}  |  p = {float(f_test.pvalue):.4e}")
if float(f_test.pvalue) < 0.05:
    print("  -> Confirms: coefficients changed significantly after April 2022")


# ── 5. SUB-PERIOD COEFFICIENT COMPARISON ─────────────────────────────────────
print("\n" + "=" * 60)
print("  SUB-PERIOD COEFFICIENT COMPARISON")
print("=" * 60)

print(f"\n  {'Variable':<15} {'Pre-2022':>12} {'Post-2022':>12} "
      f"{'Change':>12} {'Pre p':>8} {'Post p':>8}")
print("  " + "-" * 70)

for var in res_pre.params.index:
    pre_coef  = res_pre.params[var]
    post_coef = res_post.params[var]
    change    = post_coef - pre_coef
    pre_p     = res_pre.pvalues[var]
    post_p    = res_post.pvalues[var]
    print(f"  {var:<15} {pre_coef:>12.4f} {post_coef:>12.4f} "
          f"{change:>+12.4f} {pre_p:>8.4f} {post_p:>8.4f}")

print(f"\n  Sub-period R^2: Pre = {res_pre.rsquared:.4f}  |  "
      f"Post = {res_post.rsquared:.4f}")
print(f"  Sub-period N : Pre = {n_pre}          |  Post = {n_post}")


# ── 6. CUSUM TEST ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TEST 3: CUSUM TEST (building figure...)")
print("=" * 60)

Y_cusum = df_clean[YVARS]
X_cusum = sm.add_constant(df_clean[XVARS])

rls_model  = RecursiveLS(Y_cusum, X_cusum)
rls_result = rls_model.fit()

cusum       = rls_result.cusum
dates_cusum = df_clean['date'].iloc[k:]

# Manually compute Brown-Durbin-Evans (1975) 5% significance bounds.
# Formula: bound(t) = c*sqrt(n) + 2c*t/sqrt(n)
# where n = total obs minus k parameters, c = 0.948 for the 5% level.
# The bounds widen linearly over time — that is by design.
nresid      = n_full - k
c_alpha     = 0.948
t_arr       = np.arange(len(cusum))
bound_upper =  c_alpha * np.sqrt(nresid) + 2 * c_alpha * t_arr / np.sqrt(nresid)
bound_lower = -bound_upper
cusum_bounds = (bound_lower, bound_upper)

cusum_breach = (np.any(cusum > bound_upper) or
                np.any(cusum < bound_lower))

print(f"\n  CUSUM exits 5% confidence band: "
      f"{'YES -> instability detected' if cusum_breach else 'NO -> stable'}")

# ── 7. FIGURE ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 10))
fig.patch.set_facecolor('#0d1117')

for ax in axes:
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#c9d1d9', labelsize=9)
    ax.xaxis.label.set_color('#c9d1d9')
    ax.yaxis.label.set_color('#c9d1d9')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

# Panel A -- CUSUM
ax1 = axes[0]
ax1.plot(dates_cusum, cusum,
         color='#58a6ff', linewidth=1.8, label='CUSUM', zorder=3)
ax1.plot(dates_cusum, bound_lower,
         color='#f85149', linewidth=1.2, linestyle='--',
         alpha=0.8, label='5% critical bounds')
ax1.plot(dates_cusum, bound_upper,
         color='#f85149', linewidth=1.2, linestyle='--', alpha=0.8)
ax1.fill_between(dates_cusum, bound_lower, bound_upper,
                 color='#f85149', alpha=0.08)
ax1.axhline(0, color='#8b949e', linewidth=0.8, linestyle=':')
ax1.axvline(BREAK_DATE, color='#ffa657', linewidth=1.5,
            linestyle='--', label='Apr 2022 break', zorder=4)
ax1.set_title('CUSUM Test -- Recursive Residuals',
              color='#e6edf3', fontsize=13, fontweight='bold', pad=10)
ax1.set_ylabel('Cumulative Sum', color='#c9d1d9')
ax1.legend(facecolor='#21262d', edgecolor='#30363d',
           labelcolor='#c9d1d9', fontsize=9)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Panel B -- Coefficient comparison bar chart
ax2 = axes[1]

coef_vars  = XVARS
pre_coefs  = [res_pre.params[v]  for v in coef_vars]
post_coefs = [res_post.params[v] for v in coef_vars]
pre_ci     = [1.96 * res_pre.bse[v]  for v in coef_vars]
post_ci    = [1.96 * res_post.bse[v] for v in coef_vars]
labels     = ['ln(LKR/USD)', 'ln(Brent Oil)', 'ln(GCC GDP)']
x          = np.arange(len(labels))
w          = 0.32

bars1 = ax2.bar(x - w/2, pre_coefs, w,
                color='#58a6ff', alpha=0.85, label='Pre-crisis (<Apr 2022)',
                yerr=pre_ci, capsize=4,
                error_kw={'ecolor': '#8b949e', 'elinewidth': 1.2})
bars2 = ax2.bar(x + w/2, post_coefs, w,
                color='#ffa657', alpha=0.85, label='Post-crisis (>=Apr 2022)',
                yerr=post_ci, capsize=4,
                error_kw={'ecolor': '#8b949e', 'elinewidth': 1.2})

ax2.axhline(0, color='#8b949e', linewidth=0.8, linestyle=':')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, color='#c9d1d9', fontsize=10)
ax2.set_ylabel('Coefficient (log-log elasticity)', color='#c9d1d9')
ax2.set_title('Coefficient Shift: Pre vs Post April 2022',
              color='#e6edf3', fontsize=13, fontweight='bold', pad=10)
ax2.legend(facecolor='#21262d', edgecolor='#30363d',
           labelcolor='#c9d1d9', fontsize=9)

for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    ax2.annotate(f'{h:.2f}',
                 xy=(bar.get_x() + bar.get_width() / 2, h),
                 xytext=(0, 4 if h >= 0 else -12),
                 textcoords='offset points',
                 ha='center', va='bottom',
                 color='#c9d1d9', fontsize=8)

sig_str = ("p < 0.001" if chow_p < 0.001 else f"p = {chow_p:.3f}")
fig.text(0.5, 0.01,
         f"Chow F-statistic = {chow_f:.3f}  ({sig_str})  |  "
         f"Break date: April 2022  |  Pre n={n_pre}, Post n={n_post}",
         ha='center', color='#8b949e', fontsize=9, style='italic')

plt.suptitle("Structural Break Analysis -- Sri Lanka Remittances",
             color='#e6edf3', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(FIG_OUT, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print(f"\n  Figure saved -> {FIG_OUT}")


# ── 8. SAVE TEXT RESULTS ──────────────────────────────────────────────────────
sep  = "=" * 72
sep2 = "-" * 72

with open(TAB_OUT, 'w', encoding='utf-8') as f:
    f.write("CHOW STRUCTURAL BREAK TEST RESULTS\n")
    f.write("Project: sl-remittance-model\n")
    f.write("Break date: April 2022 (Sri Lanka sovereign debt crisis)\n")
    f.write(sep + "\n\n")

    f.write("SAMPLE INFORMATION\n")
    f.write(sep2 + "\n")
    f.write(f"  Full sample  : {n_full} obs  "
            f"({df_clean['date'].min().date()} to "
            f"{df_clean['date'].max().date()})\n")
    f.write(f"  Pre-crisis   : {n_pre} obs  (up to Mar 2022)\n")
    f.write(f"  Post-crisis  : {n_post} obs  (Apr 2022 onwards)\n")
    f.write(f"  Parameters k : {k}\n\n")

    f.write("TEST 1: CHOW F-TEST\n")
    f.write(sep2 + "\n")
    f.write(f"  RSS restricted (full)    : {rss_restricted:.6f}\n")
    f.write(f"  RSS unrestricted (split) : {rss_unrestricted:.6f}\n")
    f.write(f"  Chow F-statistic         : {chow_f:.4f}\n")
    f.write(f"  df (numerator, denom)    : ({df_numerator}, {df_denominator})\n")
    f.write(f"  p-value                  : {chow_p:.4e}\n")
    f.write(f"  Critical value (5%)      : {f_crit_05:.4f}\n")
    f.write(f"  Critical value (1%)      : {f_crit_01:.4f}\n")
    f.write(f"\n  VERDICT: {chow_verdict}\n\n")

    f.write("TEST 2: INTERACTION DUMMY TEST\n")
    f.write(sep2 + "\n")
    f.write(f"  Joint F on interactions  : {float(f_test.fvalue):.4f}\n")
    f.write(f"  p-value                  : {float(f_test.pvalue):.4e}\n\n")

    f.write("TEST 3: CUSUM TEST\n")
    f.write(sep2 + "\n")
    f.write(f"  CUSUM exits 5% bounds    : "
            f"{'YES -- parameter instability detected' if cusum_breach else 'NO -- stable'}\n\n")

    f.write("SUB-PERIOD COEFFICIENTS\n")
    f.write(sep2 + "\n")
    f.write(f"  {'Variable':<15} {'Pre-crisis':>12} {'Post-crisis':>12} "
            f"{'Delta':>12} {'Pre p':>8} {'Post p':>8}\n")
    f.write("  " + "-" * 65 + "\n")
    for var in res_pre.params.index:
        f.write(f"  {var:<15} "
                f"{res_pre.params[var]:>12.4f} "
                f"{res_post.params[var]:>12.4f} "
                f"{(res_post.params[var]-res_pre.params[var]):>+12.4f} "
                f"{res_pre.pvalues[var]:>8.4f} "
                f"{res_post.pvalues[var]:>8.4f}\n")
    f.write(f"\n  Pre-crisis  R^2 : {res_pre.rsquared:.4f}\n")
    f.write(f"  Post-crisis R^2 : {res_post.rsquared:.4f}\n")

print(f"\n Results saved -> {TAB_OUT}")
print("\n Phase 5 complete.")