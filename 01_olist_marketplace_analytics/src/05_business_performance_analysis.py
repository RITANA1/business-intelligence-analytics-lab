
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
CLEAN_DIR = PROJECT_DIR / "data_cleaned"
REPORT_DIR = PROJECT_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

# 1. Load datasets
orders = pd.read_csv(CLEAN_DIR / "orders_cleaned.csv")
items = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")

orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"]
)

# 2. Total orders
print("\nTOTAL ORDERS")
print(len(orders))

# 3. Order status
print("\nORDER STATUS DISTRIBUTION")
print(orders["order_status"].value_counts())

# 4. Monthly order volume
orders["month"] = (
    orders["order_purchase_timestamp"].dt.to_period("M")
)

monthly_orders = orders.groupby("month").size()

print("\nMONTHLY ORDERS")
print(monthly_orders)

# 5. Calculate revenue from order items
revenue_by_order = items.groupby("order_id")["price"].sum()

total_revenue = items["price"].sum()

print("\nTOTAL PRODUCT REVENUE")
print(round(total_revenue, 2))

# 6. Average order value
average_order_value = revenue_by_order.mean()

print("\nAVERAGE ORDER VALUE")
print(round(average_order_value, 2))

# 7. Monthly revenue
items_with_month = items.merge(
    orders[["order_id", "month"]],
    on="order_id",
    how="inner"
)

monthly_revenue = items_with_month.groupby("month")["price"].sum()

print("\nMONTHLY REVENUE")
print(monthly_revenue)

# 8. Save reports
monthly_orders.to_csv(REPORT_DIR / "monthly_orders.csv")
monthly_revenue.to_csv(REPORT_DIR / "monthly_revenue.csv")

# 9. Visualize monthly orders
monthly_orders.index = monthly_orders.index.astype(str)

monthly_orders.plot(kind="bar", figsize=(12, 5))
plt.title("Monthly Order Volume")
plt.xlabel("Month")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(REPORT_DIR / "monthly_orders.png")
plt.show()

print("\nBUSINESS PERFORMANCE ANALYSIS COMPLETED!")
