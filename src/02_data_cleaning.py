import pandas as pd
import numpy as np

def clean_remittances():
    path = "data/raw/table2.14.2_20260430_e.xlsx"
    raw = pd.read_excel(path, sheet_name='Sheet1', header=None, skiprows=3)

    years = [int(y) for y in raw.iloc[1, 2:].tolist() if pd.notna(y)]

    records = []
    for i in range(2, 14):
        month_name = raw.iloc[i, 1]
        if pd.isna(month_name):
            continue
        for j, year in enumerate(years):
            val = raw.iloc[i, j + 2]
            if pd.notna(val):
                try:
                    date = pd.to_datetime(f"{year}-{month_name}-01", format="%Y-%B-%d")
                    records.append({'date': date, 'remittances_usd_mn': float(val)})
                except:
                    pass

    df = pd.DataFrame(records).sort_values('date').reset_index(drop=True)
    print(f"  Remittances:  {len(df)} rows | {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def clean_exchange_rate():
    path = "data/raw/Monthly_Average_Exchange_Rates_20241002.xlsx"

    # Read entire sheet without skipping, find header row dynamically
    raw = pd.read_excel(path, sheet_name='Avg ExRate', header=None)

    # Find which row contains 'Year' and 'Month'
    header_row = None
    for i in range(len(raw)):
        vals = raw.iloc[i].astype(str).tolist()
        if 'Year' in vals and 'Month' in vals:
            header_row = i
            break

    print(f"  FX header found at row: {header_row}")

    # Re-read using that row as header
    df = pd.read_excel(path, sheet_name='Avg ExRate', header=header_row)
    df = df.dropna(how='all')

    # Find column names dynamically
    cols = df.columns.tolist()
    year_col  = next(c for c in cols if str(c).strip() == 'Year')
    month_col = next(c for c in cols if str(c).strip() == 'Month')
    usd_col   = next(c for c in cols if str(c).strip() == 'USD')

    df = df[[year_col, month_col, usd_col]].copy()
    df.columns = ['year', 'month', 'lkr_usd']

    # Forward-fill year (Excel merged cells leave blanks for Feb-Dec)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['year'] = df['year'].ffill()

    # Keep only real month name rows
    valid_months = ['January','February','March','April','May','June',
                    'July','August','September','October','November','December']
    df = df[df['month'].astype(str).str.strip().isin(valid_months)]
    df = df.dropna(subset=['year', 'lkr_usd'])

    df['year'] = df['year'].astype(int)
    df['date'] = pd.to_datetime(
        df['year'].astype(str) + '-' + df['month'].astype(str) + '-01',
        format='%Y-%B-%d', errors='coerce'
    )
    df['lkr_usd'] = pd.to_numeric(df['lkr_usd'], errors='coerce')
    df = df[['date', 'lkr_usd']].dropna().sort_values('date').reset_index(drop=True)

    print(f"  Exchange rate:{len(df)} rows | {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def clean_oil_prices():
    path = "data/raw/CMO-Historical-Data-Monthly.xlsx"
    df = pd.read_excel(path, sheet_name='Monthly Prices',
                       skiprows=[0, 1, 2, 3, 5], header=0)

    date_col = df.columns[0]
    df = df[[date_col, 'Crude oil, Brent']].copy()
    df.columns = ['date_str', 'brent_usd']
    df = df[df['date_str'].astype(str).str.match(r'^\d{4}M\d{2}$')]
    df['date'] = pd.to_datetime(
        df['date_str'].astype(str).apply(lambda x: f"{x[:4]}-{x[5:]}-01")
    )
    df['brent_usd'] = pd.to_numeric(df['brent_usd'], errors='coerce')
    df = df[['date', 'brent_usd']].dropna().sort_values('date').reset_index(drop=True)
    print(f"  Oil prices:   {len(df)} rows | {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def clean_gcc_gdp():
    path = "data/raw/gcc_gdp_annual.csv"
    df = pd.read_csv(path, parse_dates=['date'])
    df.columns = ['date', 'gcc_gdp_usd_bn']
    df = df.set_index('date')
    df = df.resample('ME').interpolate('linear')
    df = df.reset_index()
    print(f"  GCC GDP:      {len(df)} rows | {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def build_dataset():
    print("\n" + "="*55)
    print("  BUILDING MERGED DATASET")
    print("="*55)

    rem = clean_remittances()
    fx  = clean_exchange_rate()
    oil = clean_oil_prices()
    gcc = clean_gcc_gdp()

    for df in [rem, fx, oil, gcc]:
        df['date'] = df['date'] + pd.offsets.MonthEnd(0)

    merged = rem.merge(fx,  on='date', how='outer')
    merged = merged.merge(oil, on='date', how='outer')
    merged = merged.merge(gcc, on='date', how='outer')
    merged = merged.sort_values('date').set_index('date')
    merged = merged[merged.index >= '2009-01-01']
    merged = merged.ffill(limit=3)
    merged = merged.dropna(subset=['remittances_usd_mn', 'lkr_usd', 'brent_usd'])

    merged['remittances_lkr_mn']    = merged['remittances_usd_mn'] * merged['lkr_usd']
    merged['remittances_yoy_growth'] = merged['remittances_usd_mn'].pct_change(12) * 100
    merged['lkr_depreciation']      = merged['lkr_usd'].pct_change(12) * 100
    merged['brent_yoy_change']      = merged['brent_usd'].pct_change(12) * 100
    merged['ln_remittances']        = np.log(merged['remittances_usd_mn'])
    merged['ln_lkr_usd']            = np.log(merged['lkr_usd'])
    merged['ln_brent']              = np.log(merged['brent_usd'])
    merged['ln_gcc_gdp']            = np.log(merged['gcc_gdp_usd_bn'])
    merged['post_crisis']           = (merged.index >= '2022-04-01').astype(int)
    merged['crisis_period']         = ((merged.index >= '2022-01-01') &
                                       (merged.index <= '2022-12-31')).astype(int)

    merged.to_csv('data/processed/merged_dataset.csv')
    print(f"\n  Final dataset: {len(merged)} rows, {merged.shape[1]} columns")
    print(f"  Date range:    {merged.index.min().date()} to {merged.index.max().date()}")
    print(f"\n  Missing values:")
    print(merged.isna().sum().to_string())
    print("\n  Saved → data/processed/merged_dataset.csv")
    print("="*55)
    return merged


if __name__ == "__main__":
    df = build_dataset()
    print("\n  Preview:")
    print(df[['remittances_usd_mn', 'lkr_usd', 'brent_usd', 'gcc_gdp_usd_bn']].head(8).to_string())