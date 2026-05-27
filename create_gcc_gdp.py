import pandas as pd

# GCC Total GDP (USD Billions) - Source: IMF World Economic Outlook
# Sum of Saudi Arabia, UAE, Kuwait, Qatar, Oman, Bahrain
gcc_data = {
    2009: 863,
    2010: 1079,
    2011: 1396,
    2012: 1551,
    2013: 1621,
    2014: 1620,
    2015: 1269,
    2016: 1235,
    2017: 1304,
    2018: 1506,
    2019: 1481,
    2020: 1209,
    2021: 1461,
    2022: 1920,
    2023: 1915,
    2024: 1960,
    2025: 2050,
}

df = pd.DataFrame(list(gcc_data.items()), columns=['date', 'value'])
df['date'] = pd.to_datetime(df['date'], format='%Y')
df.to_csv('data/raw/gcc_gdp_annual.csv', index=False)
print("Saved gcc_gdp_annual.csv")
print(df.to_string())