import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm

# ── Load data ──────────────────────────────────────────────
df = pd.read_csv('data/processed/merged_dataset.csv',
                 parse_dates=['date'], index_col='date')

print(f"Loaded {len(df)} rows | {df.index.min().date()} → {df.index.max().date()}")

# ── Re-estimate AR(1) model (best forecasting model, R²=0.80) ──
model_df = df[['ln_remittances','ln_lkr_usd','ln_brent',
               'ln_gcc_gdp','post_crisis']].dropna()

y = model_df['ln_remittances']
X = sm.add_constant(model_df[['ln_lkr_usd','ln_brent',
                               'ln_gcc_gdp','post_crisis']])

# Add AR(1) lag
y_lag = y.shift(1)
X['ln_rem_lag'] = y_lag
mask = X.notna().all(axis=1) & y.notna()
X_fit, y_fit = X[mask], y[mask]

model = sm.OLS(y_fit, X_fit).fit()
ar_coef  = model.params['ln_rem_lag']
print(f"\nModel R²: {model.rsquared:.3f}  |  AR coef: {ar_coef:.3f}")
print(f"Forecast horizon: 12 months from {df.index.max().date()}\n")

# ── Last known values ───────────────────────────────────────
last = df.iloc[-1]
last_ln_rem  = last['ln_remittances']
last_ln_lkr  = last['ln_lkr_usd']
last_ln_brent= last['ln_brent']
last_ln_gcc  = last['ln_gcc_gdp']

# ── Scenario definitions ────────────────────────────────────
# Each entry = monthly log-change applied to the three drivers
scenarios = {
    'Baseline\n(Stable)': {
        'color': 'steelblue', 'ls': '-',
        'd_lkr':   0.002,   # ~2.4% annual LKR depreciation
        'd_brent': 0.000,   # oil flat ~$75
        'd_gcc':   0.003,   # GCC GDP +3.6% annual
    },
    'Gulf Boom\n(Oil surge)': {
        'color': 'forestgreen', 'ls': '--',
        'd_lkr':   0.001,
        'd_brent': 0.020,   # oil rises ~27% over 12m
        'd_gcc':   0.006,
    },
    'GCC Recession\n(Oil crash)': {
        'color': 'crimson', 'ls': '--',
        'd_lkr':   0.005,
        'd_brent': -0.035,  # oil falls ~35% over 12m
        'd_gcc':  -0.008,
    },
    'LKR Crisis\nRepeat': {
        'color': 'darkorange', 'ls': '--',
        'd_lkr':   0.025,   # rupee collapses ~35% over 12m
        'd_brent': 0.000,
        'd_gcc':   0.002,
    },
}

HORIZON = 12
forecast_dates = pd.date_range(
    start=df.index.max() + pd.DateOffset(months=1),
    periods=HORIZON, freq='ME'
)

# ── Run scenarios ───────────────────────────────────────────
results = {}

for name, cfg in scenarios.items():
    ln_lkr   = last_ln_lkr
    ln_brent = last_ln_brent
    ln_gcc   = last_ln_gcc
    ln_rem   = last_ln_rem
    preds    = []

    for _ in range(HORIZON):
        ln_lkr   += cfg['d_lkr']
        ln_brent += cfg['d_brent']
        ln_gcc   += cfg['d_gcc']

        X_pred = np.array([1, ln_lkr, ln_brent, ln_gcc, 1, ln_rem])  # post_crisis=1
        ln_rem_pred = np.dot(X_pred, model.params)

        # AR blend: 70% model prediction, 30% momentum from last period
        ln_rem = 0.70 * ln_rem_pred + 0.30 * ln_rem
        preds.append(np.exp(ln_rem))

    results[name] = preds

# ── Plot ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 7))

# Historical (last 3 years for context)
hist = df['remittances_usd_mn']['2022':]
ax.plot(hist.index, hist.values, color='black', lw=2,
        label='Historical', zorder=5)

# Forecast divider
ax.axvline(df.index.max(), color='grey', ls=':', lw=1.5, alpha=0.8)
ax.text(df.index.max(), ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 200,
        '  Forecast →', fontsize=9, color='grey', va='bottom')

# Scenario lines
for name, cfg in scenarios.items():
    vals = results[name]
    label = name.replace('\n', ' ')
    ax.plot(forecast_dates, vals,
            color=cfg['color'], ls=cfg['ls'],
            lw=2.2, label=label, alpha=0.9)

    # End-point label
    ax.annotate(f"${vals[-1]:.0f}m",
                xy=(forecast_dates[-1], vals[-1]),
                xytext=(8, 0), textcoords='offset points',
                fontsize=8, color=cfg['color'], va='center')

ax.set_title('12-Month Remittance Forecasts — Scenario Analysis\n'
             'Sri Lanka (AR(1) + macro drivers)',
             fontsize=12, fontweight='bold')
ax.set_ylabel('Remittances (USD Millions)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.tick_params(axis='x', rotation=30)
ax.legend(loc='upper left', frameon=True, fontsize=9)
ax.set_xlim([hist.index.min(), forecast_dates[-1] + pd.DateOffset(months=1)])

plt.tight_layout()
plt.savefig('outputs/figures/forecast_scenarios.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved → outputs/figures/forecast_scenarios.png")

# ── Save forecast table ──────────────────────────────────────
clean_names = {k: k.replace('\n', ' ') for k in results}
fdf = pd.DataFrame(
    {clean_names[k]: v for k, v in results.items()},
    index=forecast_dates
)
fdf.index.name = 'date'
fdf.to_csv('outputs/tables/forecast_scenarios.csv')
print("Saved → outputs/tables/forecast_scenarios.csv")

# ── Print summary ────────────────────────────────────────────
print("\n12-month forecast summary (USD Millions):")
print(f"{'Scenario':<35} {'Month 1':>10} {'Month 6':>10} {'Month 12':>10}")
print("-" * 68)
for name, vals in results.items():
    label = name.replace('\n', ' ')
    print(f"{label:<35} {vals[0]:>10.1f} {vals[5]:>10.1f} {vals[11]:>10.1f}")