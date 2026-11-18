
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
CLEAN_DIR = PROJECT_DIR / "data_cleaned"

CLEAN_DIR.mkdir(exist_ok=True)

# 1. Clean orders
orders = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv")

date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for column in date_columns:
    orders[column] = pd.to_datetime(
        orders[column], errors="coerce"
    )

orders.to_csv(CLEAN_DIR / "orders_cleaned.csv", index=False)

print("Orders cleaned and saved.")
print("Orders:", len(orders))


# 2. Clean geolocation
geo = pd.read_csv(DATA_DIR / "olist_geolocation_dataset.csv")

before = len(geo)
geo = geo.drop_duplicates()
after = len(geo)

geo.to_csv(CLEAN_DIR / "geolocation_cleaned.csv", index=False)

print("\nGeolocation cleaning")
print("Before:", before)
print("After:", after)
print("Exact duplicates removed:", before - after)


# 3. Preserve reviews without deleting missing comments
reviews = pd.read_csv(
    DATA_DIR / "olist_order_reviews_dataset.csv"
)

reviews.to_csv(CLEAN_DIR / "reviews_cleaned.csv", index=False)

print("\nReviews preserved:", len(reviews))


# 4. Save products without deleting missing attributes
products = pd.read_csv(DATA_DIR / "olist_products_dataset.csv")

products.to_csv(CLEAN_DIR / "products_cleaned.csv", index=False)

print("\nProducts preserved:", len(products))

print("\nDATA CLEANING COMPLETED!")
print("Cleaned files saved in:", CLEAN_DIR)
