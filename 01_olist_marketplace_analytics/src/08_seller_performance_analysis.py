
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
REPORT_DIR = PROJECT_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

# 1. Load datasets
items = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")
sellers = pd.read_csv(DATA_DIR / "olist_sellers_dataset.csv")
reviews = pd.read_csv(DATA_DIR / "olist_order_reviews_dataset.csv")

# 2. Calculate sales metrics per seller
seller_sales = items.groupby("seller_id").agg(
    total_revenue=("price", "sum"),
    items_sold=("order_item_id", "count"),
    unique_orders=("order_id", "nunique"),
    average_item_price=("price", "mean")
)

# 3. Calculate one review per seller-order combination
seller_orders = items[
    ["seller_id", "order_id"]
].drop_duplicates()

seller_reviews = seller_orders.merge(
    reviews[["order_id", "review_score"]],
    on="order_id",
    how="left"
)

seller_ratings = seller_reviews.groupby("seller_id").agg(
    average_review_score=("review_score", "mean"),
    reviewed_orders=("review_score", "count")
)

# 4. Combine seller information
seller_performance = seller_sales.join(
    seller_ratings,
    how="left"
).reset_index()

seller_performance = seller_performance.merge(
    sellers[["seller_id", "seller_state"]],
    on="seller_id",
    how="left"
)

# 5. Top sellers by revenue
top_sellers = seller_performance.sort_values(
    "total_revenue", ascending=False
)

print("\nTOP 10 SELLERS BY REVENUE")
print(top_sellers.head(10).round(2).to_string(index=False))

# 6. Top sellers by customer rating
rated_sellers = seller_performance[
    seller_performance["reviewed_orders"] >= 10
].sort_values(
    "average_review_score", ascending=False
)

print("\nTOP 10 SELLERS BY CUSTOMER RATING")
print(
    rated_sellers.head(10)[
        ["seller_id", "average_review_score", "reviewed_orders"]
    ].round(2).to_string(index=False)
)

# 7. Seller performance by state
state_performance = seller_performance.groupby("seller_state").agg(
    total_revenue=("total_revenue", "sum"),
    total_sellers=("seller_id", "nunique"),
    total_orders=("unique_orders", "sum")
).sort_values("total_revenue", ascending=False)

print("\nSELLER PERFORMANCE BY STATE")
print(state_performance.head(10).round(2))

# 8. Save reports
seller_performance.to_csv(
    REPORT_DIR / "seller_performance.csv", index=False
)

state_performance.to_csv(
    REPORT_DIR / "seller_state_performance.csv"
)

# 9. Visualize top 10 sellers
top_sellers.head(10).sort_values(
    "total_revenue"
).plot(
    x="seller_id",
    y="total_revenue",
    kind="barh",
    legend=False,
    figsize=(10, 6)
)

plt.title("Top 10 Sellers by Revenue")
plt.xlabel("Revenue")
plt.ylabel("Seller ID")
plt.tight_layout()
plt.savefig(REPORT_DIR / "top_sellers_by_revenue.png")
plt.close()

print("\nSELLER PERFORMANCE ANALYSIS COMPLETED!")
