
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
REPORT_DIR = PROJECT_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

# Store detailed results
quality_results = []

print("DETAILED DATA QUALITY INVESTIGATION")
print("=" * 60)

# Inspect every CSV file
for file in sorted(DATA_DIR.glob("*.csv")):

    df = pd.read_csv(file)

    for column in df.columns:

        missing = int(df[column].isna().sum())

        quality_results.append({
            "dataset": file.name,
            "column": column,
            "data_type": str(df[column].dtype),
            "missing_values": missing,
            "missing_percentage": round(
                missing / len(df) * 100, 2
            ),
            "unique_values": int(df[column].nunique())
        })

# Create detailed report
report = pd.DataFrame(quality_results)

report.to_csv(
    REPORT_DIR / "column_quality_report.csv",
    index=False
)

# Show only columns containing missing values
print("\nCOLUMNS WITH MISSING VALUES")
print("=" * 60)

problem_columns = report[
    report["missing_values"] > 0
].sort_values(
    "missing_percentage",
    ascending=False
)

print(problem_columns.to_string(index=False))


# Investigate order statuses
print("\nORDER STATUS DISTRIBUTION")
print("=" * 60)

orders = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv")

print(orders["order_status"].value_counts(dropna=False))


# Investigate missing dates by order status
date_columns = [
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date"
]

print("\nMISSING ORDER DATES BY STATUS")
print("=" * 60)

for column in date_columns:

    print(f"\n{column}")

    result = pd.crosstab(
        orders["order_status"],
        orders[column].isna()
    )

    print(result)


print("\nINVESTIGATION COMPLETED")
print("Report saved: reports/column_quality_report.csv")
