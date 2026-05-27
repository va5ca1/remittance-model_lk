import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import statsmodels.api as sm

df = pd.read_csv('data/processed/merged_dataset.csv',
                 parse_dates=['date'], index_col='date')

crisis = pd.Timestamp('2022-04-01')

# ── Re-estimate Model 2 (crisis dummy OLS) for fitted values ──
mdf = df[['ln_remittances','ln_lkr_usd','ln_brent',
          'ln_gcc_gdp','post_crisis','crisis_period']].dropna()
y = mdf['ln_remittances']
X = sm.add_constant(mdf[['ln_lkr_usd','ln_brent',
                          'ln_gcc_gdp','post_crisis','crisis_period']])
model = sm.OLS(y, X).fit()

actual  = np.exp(y)
fitted  = np.exp(model.fittedvalues)

# ── Canvas ────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 11))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

CRISIS_COLOR = '#c0392b'
PRE_COLOR    = '#2980b9'
POST_COLOR   = '#e74c3c'
OIL_COLOR    = '#e67e22'

# ─────────────────────────────────────────────────────────────
# Panel 1 — Remittances timeline
# ─────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.fill_between(df.index, 0, df['remittances_usd_mn'],
                 alpha=0.15, color=PRE_COLOR)
ax1.plot(df.index, df['remittances_usd_mn'],
         color=PRE_COLOR, lw=1.8, label='Remittances')
ax1.axvline(crisis, color=CRISIS_COLOR, ls='--', lw=1.5)
ax1.annotate('Sovereign default\n& rupee float',
             xy=(crisis, 650),
             xytext=(pd.Timestamp('2019-06-01'), 780),
             arrowprops=dict(arrowstyle='->', color=CRISIS_COLOR, lw=1.2),
             fontsize=8, color=CRISIS_COLOR, fontweight='bold')

# Shade crisis year
ax1.axvspan(pd.Timestamp('2022-01-01'), pd.Timestamp('2023-01-01'),
            alpha=0.08, color=CRISIS_COLOR)

ax1.set_title('Panel A — Worker Remittances (2009–2025)',
              fontsize=10, fontweight='bold', pad=8)
ax1.set_ylabel('USD Millions')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.tick_params(axis='x', rotation=30)
ax1.set_ylim(bottom=0)

# ─────────────────────────────────────────────────────────────
# Panel 2 — Gulf connection (dual axis)
# ─────────────────────────────────────────────────────────────
ax2  = fig.add_subplot(gs[0, 1])
ax2r = ax2.twinx()

l1, = ax2.plot(df.index, df['remittances_usd_mn'],
               color=PRE_COLOR, lw=1.8, label='Remittances')
l2, = ax2r.plot(df.index, df['brent_usd'],
                color=OIL_COLOR, lw=1.5, alpha=0.85, label='Brent oil')
ax2.axvline(crisis, color=CRISIS_COLOR, ls='--', lw=1.2, alpha=0.6)

ax2.set_ylabel('Remittances (USD Mn)', color=PRE_COLOR, fontsize=9)
ax2r.set_ylabel('Brent (USD/barrel)', color=OIL_COLOR, fontsize=9)
ax2.tick_params(axis='y', labelcolor=PRE_COLOR)
ax2r.tick_params(axis='y', labelcolor=OIL_COLOR)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.tick_params(axis='x', rotation=30)

lines = [l1, l2]
ax2.legend(lines, [l.get_label() for l in lines],
           loc='upper left', fontsize=8, framealpha=0.9)
ax2.set_title('Panel B — The Gulf Connection: Remittances & Oil',
              fontsize=10, fontweight='bold', pad=8)

# ─────────────────────────────────────────────────────────────
# Panel 3 — Regime scatter (pre vs post crisis)
# ─────────────────────────────────────────────────────────────
ax3  = fig.add_subplot(gs[1, 0])
pre  = df[df.index <  crisis]
post = df[df.index >= crisis]

ax3.scatter(pre['lkr_usd'],  pre['remittances_usd_mn'],
            c=PRE_COLOR,  alpha=0.55, s=35,
            edgecolors='white', lw=0.3, label='Pre-crisis (n=159)')
ax3.scatter(post['lkr_usd'], post['remittances_usd_mn'],
            c=POST_COLOR, alpha=0.75, s=45,
            edgecolors='white', lw=0.3, label='Post-crisis (n=36)')

for subset, color in [(pre, PRE_COLOR), (post, POST_COLOR)]:
    m, b = np.polyfit(subset['lkr_usd'], subset['remittances_usd_mn'], 1)
    x = np.linspace(subset['lkr_usd'].min(), subset['lkr_usd'].max(), 100)
    ax3.plot(x, m*x + b, color=color, lw=1.8, ls='--', alpha=0.8)

ax3.set_xlabel('LKR / USD', fontsize=9)
ax3.set_ylabel('Remittances (USD Mn)', fontsize=9)
ax3.legend(fontsize=8, framealpha=0.9)
ax3.set_title('Panel C — Regime Shift: Pre vs Post Crisis',
              fontsize=10, fontweight='bold', pad=8)

ax3.annotate('Rupee collapse\nchanges relationship',
             xy=(300, post['remittances_usd_mn'].mean()),
             xytext=(200, 780),
             arrowprops=dict(arrowstyle='->', color='grey', lw=1),
             fontsize=7.5, color='grey')

# ─────────────────────────────────────────────────────────────
# Panel 4 — Model fit
# ─────────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(mdf.index, actual.values, color='black',
         lw=1.6, alpha=0.85, label='Actual', zorder=3)
ax4.plot(mdf.index, fitted.values, color=POST_COLOR,
         lw=1.6, ls='--', alpha=0.85, label='Model fitted', zorder=2)
ax4.fill_between(mdf.index, actual.values, fitted.values,
                 alpha=0.12, color='grey', label='Residual')
ax4.axvline(crisis, color=CRISIS_COLOR, ls=':', lw=1.2, alpha=0.7)

ax4.set_ylabel('Remittances (USD Mn)', fontsize=9)
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax4.tick_params(axis='x', rotation=30)
ax4.legend(fontsize=8, framealpha=0.9)
ax4.set_title(f'Panel D — Model Fit (R² = {model.rsquared:.3f}, crisis dummies)',
              fontsize=10, fontweight='bold', pad=8)

# ─────────────────────────────────────────────────────────────
# Main title
# ─────────────────────────────────────────────────────────────
fig.suptitle(
    'Determinants of Sri Lankan Worker Remittances (2009–2025)\n'
    'GCC Economic Conditions, Exchange Rate Policy & the 2022 Sovereign Crisis',
    fontsize=13, fontweight='bold', y=1.01
)

plt.savefig('outputs/figures/final_dashboard.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("Saved → outputs/figures/final_dashboard.png")
print("\nPhase 7 complete. Ready for Phase 8: README.md")