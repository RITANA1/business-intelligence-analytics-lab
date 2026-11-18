
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
customers = pd.read_csv(DATA_DIR / "olist_customers_dataset.csv")
items = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")

# 2. Convert dates
date_columns = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for column in date_columns:
    orders[column] = pd.to_datetime(
        orders[column], errors="coerce"
    )

# 3. Calculate freight per order
freight = items.groupby("order_id").agg(
    freight_value=("freight_value", "sum"),
    product_value=("price", "sum")
).reset_index()

# 4. Combine delivery and customer information
data = orders.merge(
    customers[["customer_id", "customer_state"]],
    on="customer_id",
    how="left"
)

data = data.merge(freight, on="order_id", how="left")

# 5. Keep delivered orders with valid dates
delivered = data[
    (data["order_status"] == "delivered")
    & data["order_delivered_customer_date"].notna()
    & data["order_estimated_delivery_date"].notna()
    & data["order_purchase_timestamp"].notna()
].copy()

# 6. Calculate delivery metrics
delivered["delivery_days"] = (
    delivered["order_delivered_customer_date"]
    - delivered["order_purchase_timestamp"]
).dt.total_seconds() / 86400

delivered["delay_days"] = (
    delivered["order_delivered_customer_date"]
    - delivered["order_estimated_delivery_date"]
).dt.total_seconds() / 86400

delivered["was_late"] = delivered["delay_days"] > 0

print("\nOVERALL LOGISTICS PERFORMANCE")
print("Delivered orders analyzed:", len(delivered))
print("Average delivery time:",
      round(delivered["delivery_days"].mean(), 2), "days")
print("Average difference from estimated date:",
      round(delivered["delay_days"].mean(), 2), "days")
print("Late delivery rate:",
      round(delivered["was_late"].mean() * 100, 2), "%")
print("Average freight per order:",
      round(delivered["freight_value"].mean(), 2))

# 7. Delivery performance by customer state
state_logistics = delivered.groupby("customer_state").agg(
    delivered_orders=("order_id", "count"),
    average_delivery_days=("delivery_days", "mean"),
    average_delay_days=("delay_days", "mean"),
    late_delivery_rate=("was_late", "mean"),
    average_freight=("freight_value", "mean")
)

state_logistics["late_delivery_rate"] *= 100

state_logistics = state_logistics.sort_values(
    "late_delivery_rate", ascending=False
)

print("\nDELIVERY PERFORMANCE BY CUSTOMER STATE")
print(state_logistics.round(2).to_string())

# 8. Save reports
delivered.to_csv(
    REPORT_DIR / "order_logistics_metrics.csv", index=False
)

state_logistics.to_csv(
    REPORT_DIR / "logistics_by_customer_state.csv"
)

# 9. Visualize average delivery time by state
state_logistics.sort_values(
    "average_delivery_days"
)["average_delivery_days"].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Average Delivery Time by Customer State")
plt.xlabel("Customer State")
plt.ylabel("Average Delivery Days")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "delivery_time_by_state.png"
)
plt.close()

# 10. Visualize late delivery rate by state
state_logistics.sort_values(
    "late_delivery_rate"
)["late_delivery_rate"].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Late Delivery Rate by Customer State")
plt.xlabel("Customer State")
plt.ylabel("Late Delivery Rate (%)")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "late_delivery_rate_by_state.png"
)
plt.close()

print("\nLOGISTICS PERFORMANCE ANALYSIS COMPLETED!")
