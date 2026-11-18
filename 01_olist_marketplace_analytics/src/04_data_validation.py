
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_DIR / "data_cleaned"

print("DATA VALIDATION")
print("=" * 50)

# 1. Orders
orders = pd.read_csv(CLEAN_DIR / "orders_cleaned.csv")

date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

# Convert date columns into datetime
for column in date_columns:
    orders[column] = pd.to_datetime(
        orders[column], errors="coerce"
    )

print("\nORDERS")
print("Rows:", len(orders))
print("Duplicate order IDs:", orders["order_id"].duplicated().sum())

# 2. Geolocation
geo = pd.read_csv(CLEAN_DIR / "geolocation_cleaned.csv")

print("\nGEOLOCATION")
print("Rows:", len(geo))
print("Exact duplicate rows:", geo.duplicated().sum())

# 3. Reviews
reviews = pd.read_csv(CLEAN_DIR / "reviews_cleaned.csv")

print("\nREVIEWS")
print("Rows:", len(reviews))
print("Missing review titles:", reviews["review_comment_title"].isna().sum())
print("Missing review messages:", reviews["review_comment_message"].isna().sum())

# 4. Products
products = pd.read_csv(CLEAN_DIR / "products_cleaned.csv")

print("\nPRODUCTS")
print("Rows:", len(products))
print("Missing values:", products.isna().sum().sum())

# 5. Date validation
print("\nDATE VALIDATION")
print(orders[date_columns].dtypes)

print("\nVALIDATION COMPLETED")
