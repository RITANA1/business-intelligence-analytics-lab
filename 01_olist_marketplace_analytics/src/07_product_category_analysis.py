
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
REPORT_DIR = PROJECT_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

# 1. Load datasets
items = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")
products = pd.read_csv(DATA_DIR / "olist_products_dataset.csv")
translations = pd.read_csv(
    DATA_DIR / "product_category_name_translation.csv"
)

# 2. Join product information with sales
sales = items.merge(
    products[["product_id", "product_category_name"]],
    on="product_id",
    how="left"
)

sales = sales.merge(
    translations,
    on="product_category_name",
    how="left"
)

# 3. Handle missing category names
sales["product_category_name_english"] = (
    sales["product_category_name_english"]
    .fillna("Unknown")
)

# 4. Category performance
category_performance = sales.groupby(
    "product_category_name_english"
).agg(
    total_revenue=("price", "sum"),
    units_sold=("order_item_id", "count"),
    unique_orders=("order_id", "nunique"),
    average_price=("price", "mean")
).sort_values(
    "total_revenue", ascending=False
)

print("\nTOP 15 CATEGORIES BY REVENUE")
print(category_performance.head(15).round(2))

# 5. Top-selling categories by quantity
print("\nTOP 15 CATEGORIES BY UNITS SOLD")
print(
    category_performance
    .sort_values("units_sold", ascending=False)
    .head(15)
)

# 6. Save complete report
category_performance.to_csv(
    REPORT_DIR / "category_performance.csv"
)

# 7. Visualize top 10 categories by revenue
top_categories = category_performance.head(10)

top_categories["total_revenue"].sort_values().plot(
    kind="barh",
    figsize=(11, 7)
)

plt.title("Top 10 Product Categories by Revenue")
plt.xlabel("Revenue")
plt.ylabel("Product Category")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "top_categories_by_revenue.png"
)
plt.close()

print("\nPRODUCT CATEGORY ANALYSIS COMPLETED!")
