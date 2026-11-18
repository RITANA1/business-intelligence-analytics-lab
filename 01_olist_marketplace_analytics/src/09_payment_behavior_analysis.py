
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
REPORT_DIR = PROJECT_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

# 1. Load payment data
payments = pd.read_csv(
    DATA_DIR / "olist_order_payments_dataset.csv"
)

# 2. Overall payment statistics
print("\nPAYMENT OVERVIEW")
print("Total payment records:", len(payments))
print("Unique orders:", payments["order_id"].nunique())
print("Total payment value:", round(payments["payment_value"].sum(), 2))
print("Average payment value per record:", round(
    payments["payment_value"].mean(), 2
))

# 3. Payment method analysis
payment_methods = payments.groupby("payment_type").agg(
    transactions=("order_id", "count"),
    unique_orders=("order_id", "nunique"),
    total_payment_value=("payment_value", "sum"),
    average_payment_value=("payment_value", "mean")
).sort_values("total_payment_value", ascending=False)

print("\nPAYMENT METHOD PERFORMANCE")
print(payment_methods.round(2))

# 4. Installment analysis
installments = payments.groupby("payment_installments").agg(
    transactions=("order_id", "count"),
    total_payment_value=("payment_value", "sum"),
    average_payment_value=("payment_value", "mean")
).sort_values("payment_installments")

print("\nPAYMENT INSTALLMENT ANALYSIS")
print(installments.round(2))

# 5. Save reports
payment_methods.to_csv(
    REPORT_DIR / "payment_method_analysis.csv"
)

installments.to_csv(
    REPORT_DIR / "payment_installment_analysis.csv"
)

# 6. Visualize payment methods
payment_methods["transactions"].sort_values().plot(
    kind="barh",
    figsize=(9, 5)
)

plt.title("Transactions by Payment Method")
plt.xlabel("Number of Payment Records")
plt.ylabel("Payment Method")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "payment_methods.png"
)
plt.close()

# 7. Visualize installment distribution
installments["transactions"].head(12).plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("Transactions by Number of Installments")
plt.xlabel("Number of Installments")
plt.ylabel("Number of Payment Records")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "payment_installments.png"
)
plt.close()

print("\nPAYMENT BEHAVIOR ANALYSIS COMPLETED!")
