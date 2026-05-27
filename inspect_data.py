import pandas as pd

# Handle each file individually with the right engine

print("\n" + "="*60)
print("FILE: Remittances")
print("="*60)
xl = pd.ExcelFile("data/raw/table2.14.2_20260430_e.xlsx")
print(f"Sheets: {xl.sheet_names}")
df = pd.read_excel("data/raw/table2.14.2_20260430_e.xlsx", sheet_name=xl.sheet_names[0], nrows=8)
print(df.to_string())

print("\n" + "="*60)
print("FILE: Exchange Rate")
print("="*60)
xl = pd.ExcelFile("data/raw/Monthly_Average_Exchange_Rates_20241002.xlsx")
print(f"Sheets: {xl.sheet_names}")
df = pd.read_excel("data/raw/Monthly_Average_Exchange_Rates_20241002.xlsx", sheet_name=xl.sheet_names[0], nrows=8)
print(df.to_string())

print("\n" + "="*60)
print("FILE: Oil Prices")
print("="*60)
xl = pd.ExcelFile("data/raw/CMO-Historical-Data-Monthly.xlsx")
print(f"Sheets: {xl.sheet_names}")
df = pd.read_excel("data/raw/CMO-Historical-Data-Monthly.xlsx", sheet_name=xl.sheet_names[0], nrows=8)
print(df.to_string())

print("\n" + "="*60)
print("FILE: GCC GDP (IMF)")
print("="*60)
try:
    df = pd.read_html("data/raw/imf-dm-export-20260527.xls")[0]
    print("(Read as HTML table)")
    print(df.head(8).to_string())
except Exception as e:
    print(f"HTML read failed: {e}")
    try:
        df = pd.read_csv("data/raw/imf-dm-export-20260527.xls", sep="\t", nrows=8)
        print("(Read as tab-separated)")
        print(df.to_string())
    except Exception as e2:
        print(f"TSV read also failed: {e2}")