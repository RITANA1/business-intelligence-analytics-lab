
from pathlib import Path
import pandas as pd

# Locate project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
REPORT_DIR = PROJECT_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

# Find all CSV files
csv_files = list(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV files found in the data folder.")

results = []

print("\nOLIST DATA QUALITY ASSESSMENT")
print("=" * 50)

for file in csv_files:
    df = pd.read_csv(file)

    rows, columns = df.shape
    missing = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())

    results.append({
        "file": file.name,
        "rows": rows,
        "columns": columns,
        "missing_values": missing,
        "duplicate_rows": duplicates
    })

    print(f"\nFile: {file.name}")
    print(f"Rows: {rows:,}")
    print(f"Columns: {columns}")
    print(f"Missing values: {missing:,}")
    print(f"Duplicate rows: {duplicates:,}")
    print("\nColumn data types:")
    print(df.dtypes.to_string())

# Save report
report = pd.DataFrame(results)
report.to_csv(REPORT_DIR / "data_quality_report.csv", index=False)

print("\n" + "=" * 50)
print("ASSESSMENT COMPLETED")
print("Report saved in reports/data_quality_report.csv")
print(report.to_string(index=False))
