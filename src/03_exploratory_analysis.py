import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_csv('data/processed/merged_dataset.csv', parse_dates=['date'], index_col='date')

crisis = pd.Timestamp('2022-04-01')
imf    = pd.Timestamp('2022-08-01')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Sri Lanka Remittance Model — Exploratory Analysis',
             fontsize=14, fontweight='bold', y=1.01)

# --- Panel 1: Remittances over time ---
ax = axes[0, 0]
ax.fill_between(df.index, 0, df['remittances_usd_mn'], alpha=0.2, color='steelblue')
ax.plot(df.index, df['remittances_usd_mn'], color='steelblue', lw=1.8)
ax.axvline(crisis, color='red',    ls='--', lw=1.5, label='Crisis onset (Apr 2022)')
ax.axvline(imf,    color='orange', ls='--', lw=1.5, label='IMF deal (Aug 2022)')
ax.set_title('Worker Remittances to Sri Lanka', fontweight='bold')
ax.set_ylabel('USD Millions')
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# --- Panel 2: LKR/USD exchange rate ---
ax = axes[0, 1]
ax.plot(df.index, df['lkr_usd'], color='darkred', lw=1.8)
ax.axvline(crisis, color='red', ls='--', lw=1.5, label='Rupee float (Apr 2022)')
ax.fill_between(df.index, df['lkr_usd'],
                where=df.index >= crisis, alpha=0.15, color='red', label='Post-float period')
ax.set_title('LKR / USD Exchange Rate', fontweight='bold')
ax.set_ylabel('LKR per 1 USD')
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# --- Panel 3: Brent oil price ---
ax = axes[1, 0]
ax.plot(df.index, df['brent_usd'], color='darkorange', lw=1.8)
ax.axvline(crisis, color='red', ls='--', lw=1.5, alpha=0.7)
ax.set_title('Brent Crude Oil Price (Gulf Proxy)', fontweight='bold')
ax.set_ylabel('USD per Barrel')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# --- Panel 4: Remittances vs oil, pre/post crisis ---
ax = axes[1, 1]
pre  = df[df.index <  crisis]
post = df[df.index >= crisis]
ax.scatter(pre['brent_usd'],  pre['remittances_usd_mn'],
           c='steelblue', alpha=0.6, s=40, edgecolors='black', lw=0.4, label='Pre-crisis')
ax.scatter(post['brent_usd'], post['remittances_usd_mn'],
           c='crimson',   alpha=0.6, s=40, edgecolors='black', lw=0.4, label='Post-crisis')

# Fit lines for each period
for subset, color in [(pre, 'steelblue'), (post, 'crimson')]:
    m, b = np.polyfit(subset['brent_usd'], subset['remittances_usd_mn'], 1)
    x = np.linspace(subset['brent_usd'].min(), subset['brent_usd'].max(), 100)
    ax.plot(x, m*x + b, color=color, lw=1.5, ls='--')

ax.set_title('Remittances vs Oil Price (Pre/Post Crisis)', fontweight='bold')
ax.set_xlabel('Brent (USD/barrel)')
ax.set_ylabel('Remittances (USD Mn)')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('outputs/figures/exploratory_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved → outputs/figures/exploratory_analysis.png")

# --- Print summary stats ---
print("\nSummary Statistics:")
print(df[['remittances_usd_mn','lkr_usd','brent_usd','gcc_gdp_usd_bn']].describe().round(2).to_string())