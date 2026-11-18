
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
CLEAN_DIR = PROJECT_DIR / "data_cleaned"
REPORT_DIR = PROJECT_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

# 1. Load data
orders = pd.read_csv(CLEAN_DIR / "orders_cleaned.csv")
reviews = pd.read_csv(CLEAN_DIR / "reviews_cleaned.csv")

# 2. Convert dates
date_columns = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for column in date_columns:
    orders[column] = pd.to_datetime(orders[column], errors="coerce")

# 3. Connect reviews with orders
data = reviews.merge(
    orders,
    on="order_id",
    how="inner"
)

# 4. Overall review score
print("\nCUSTOMER SATISFACTION")
print("Total reviews:", len(data))
print("Average review score:", round(data["review_score"].mean(), 2))

print("\nREVIEW SCORE DISTRIBUTION")
print(data["review_score"].value_counts().sort_index())

# 5. Monthly customer satisfaction
data["month"] = data["order_purchase_timestamp"].dt.to_period("M")

monthly_scores = data.groupby("month")["review_score"].mean()

print("\nMONTHLY AVERAGE REVIEW SCORE")
print(monthly_scores)

# 6. Calculate delivery delay
delivered = data[
    data["order_delivered_customer_date"].notna()
    & data["order_estimated_delivery_date"].notna()
].copy()

delivered["delivery_delay_days"] = (
    delivered["order_delivered_customer_date"]
    - delivered["order_estimated_delivery_date"]
).dt.total_seconds() / 86400

# Positive = late delivery
# Negative = early delivery

print("\nDELIVERY PERFORMANCE")
print("Orders with delivery dates:", len(delivered))
print(
    "Average difference from estimated delivery:",
    round(delivered["delivery_delay_days"].mean(), 2),
    "days"
)

# 7. Review scores by delivery performance
delivered["delivery_result"] = delivered["delivery_delay_days"].apply(
    lambda days: "Late" if days > 0 else "On time or early"
)

scores_by_delivery = delivered.groupby(
    "delivery_result"
)["review_score"].mean()

print("\nCUSTOMER SCORES BY DELIVERY PERFORMANCE")
print(scores_by_delivery)

# 8. Save reports
monthly_scores.to_csv(REPORT_DIR / "monthly_customer_satisfaction.csv")
scores_by_delivery.to_csv(REPORT_DIR / "satisfaction_by_delivery.csv")

# 9. Visualize review distribution
data["review_score"].value_counts().sort_index().plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Customer Review Score Distribution")
plt.xlabel("Review Score")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig(REPORT_DIR / "review_score_distribution.png")
plt.close()

# 10. Visualize satisfaction by delivery performance
scores_by_delivery.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Customer Satisfaction by Delivery Performance")
plt.xlabel("Delivery Result")
plt.ylabel("Average Review Score")
plt.tight_layout()
plt.savefig(REPORT_DIR / "satisfaction_by_delivery.png")
plt.close()

print("\nCUSTOMER EXPERIENCE ANALYSIS COMPLETED!")
