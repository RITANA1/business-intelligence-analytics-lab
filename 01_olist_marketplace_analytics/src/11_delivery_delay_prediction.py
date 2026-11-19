
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

from sklearn.inspection import permutation_importance


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

orders = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv")
order_items = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")
customers = pd.read_csv(DATA_DIR / "olist_customers_dataset.csv")

print("\nDATA LOADING")
print("=" * 55)
print("Orders:", len(orders))
print("Order items:", len(order_items))
print("Customers:", len(customers))


# ============================================================
# 3. PREPARE DATES
# ============================================================

date_columns = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for column in date_columns:
    orders[column] = pd.to_datetime(
        orders[column], errors="coerce"
    )


# ============================================================
# 4. FILTER VALID DELIVERED ORDERS
# ============================================================

data = orders[
    (orders["order_status"] == "delivered") &
    orders["order_purchase_timestamp"].notna() &
    orders["order_delivered_customer_date"].notna() &
    orders["order_estimated_delivery_date"].notna()
].copy()


# ============================================================
# 5. JOIN CUSTOMER DATA
# ============================================================

data = data.merge(
    customers[["customer_id", "customer_state"]],
    on="customer_id",
    how="left",
    validate="many_to_one"
)


# ============================================================
# 6. CREATE TARGET
# ============================================================

data["actual_late"] = (
    data["order_delivered_customer_date"] >
    data["order_estimated_delivery_date"]
).astype(int)


# ============================================================
# 7. FEATURE ENGINEERING
# ============================================================

data["purchase_month"] = (
    data["order_purchase_timestamp"].dt.month
)

data["purchase_weekday"] = (
    data["order_purchase_timestamp"].dt.day_of_week
)

data["purchase_hour"] = (
    data["order_purchase_timestamp"].dt.hour
)

data["promised_delivery_days"] = (
    data["order_estimated_delivery_date"] -
    data["order_purchase_timestamp"]
).dt.total_seconds() / 86400


# ============================================================
# 8. AGGREGATE ORDER ITEMS
# ============================================================

order_features = order_items.groupby("order_id").agg(
    item_count=("order_item_id", "count"),
    product_value=("price", "sum"),
    freight_value=("freight_value", "sum")
).reset_index()

data = data.merge(
    order_features,
    on="order_id",
    how="inner",
    validate="one_to_one"
)


# ============================================================
# 9. SELECT FEATURES
# ============================================================

feature_columns = [
    "customer_state",
    "item_count",
    "product_value",
    "freight_value",
    "purchase_month",
    "purchase_weekday",
    "purchase_hour",
    "promised_delivery_days"
]

numeric_features = [
    "item_count",
    "product_value",
    "freight_value",
    "purchase_month",
    "purchase_weekday",
    "purchase_hour",
    "promised_delivery_days"
]

categorical_features = ["customer_state"]

data = data.replace([np.inf, -np.inf], np.nan)

data = data.sort_values(
    "order_purchase_timestamp"
).reset_index(drop=True)

X = data[feature_columns].copy()
y = data["actual_late"].copy()


# ============================================================
# 10. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

n = len(data)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val = X.iloc[train_end:validation_end]
y_val = y.iloc[train_end:validation_end]

X_test = X.iloc[validation_end:]
y_test = y.iloc[validation_end:]

print("\nCHRONOLOGICAL DATA SPLIT")
print("=" * 55)

print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

print(
    "Training period:",
    data["order_purchase_timestamp"].iloc[0],
    "to",
    data["order_purchase_timestamp"].iloc[train_end - 1]
)

print(
    "Validation period:",
    data["order_purchase_timestamp"].iloc[train_end],
    "to",
    data["order_purchase_timestamp"].iloc[validation_end - 1]
)

print(
    "Testing period:",
    data["order_purchase_timestamp"].iloc[validation_end],
    "to",
    data["order_purchase_timestamp"].iloc[-1]
)


# ============================================================
# 11. PREPROCESSING PIPELINE
# ============================================================

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])


# ============================================================
# 12. RANDOM FOREST MODEL
# ============================================================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

print("\nTRAINING MODEL...")

model.fit(X_train, y_train)

print("MODEL TRAINING COMPLETED!")


# ============================================================
# 13. VALIDATION AND THRESHOLD SELECTION
# ============================================================

val_probabilities = model.predict_proba(X_val)[:, 1]

thresholds = np.round(
    np.arange(0.10, 0.91, 0.05), 2
)

threshold_results = []

for threshold in thresholds:

    predictions = (
        val_probabilities >= threshold
    ).astype(int)

    threshold_results.append({
        "threshold": threshold,
        "precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_val, predictions, zero_division=0
        )
    })

threshold_df = pd.DataFrame(threshold_results)

threshold_df.to_csv(
    REPORTS_DIR / "threshold_comparison.csv",
    index=False
)

best_index = threshold_df["f1_score"].idxmax()

best_threshold = float(
    threshold_df.loc[best_index, "threshold"]
)

print("\nSELECTED VALIDATION THRESHOLD:", best_threshold)


# ============================================================
# 14. FINAL TEST EVALUATION
# ============================================================

test_probabilities = model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

auc = roc_auc_score(y_test, test_probabilities)

avg_precision = average_precision_score(
    y_test, test_probabilities
)

precision = precision_score(
    y_test, test_predictions, zero_division=0
)

recall = recall_score(
    y_test, test_predictions, zero_division=0
)

f1 = f1_score(
    y_test, test_predictions, zero_division=0
)

accuracy = accuracy_score(
    y_test, test_predictions
)

print("\nFINAL TEST RESULTS")
print("=" * 55)

print("ROC-AUC:", round(auc, 4))
print("Average Precision:", round(avg_precision, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))
print("Accuracy:", round(accuracy, 4))

print("\nCLASSIFICATION REPORT")

print(classification_report(
    y_test,
    test_predictions,
    labels=[0, 1],
    target_names=["On time or early", "Late"],
    zero_division=0
))


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
)

print("\nCONFUSION MATRIX")
print(cm)

fig, ax = plt.subplots(figsize=(7, 6))

ax.imshow(cm)

ax.set_title("Delivery Delay Prediction - Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["On time", "Late"])
ax.set_yticklabels(["On time", "Late"])

for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "delivery_confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# 16. SAVE PREDICTIONS
# ============================================================

prediction_results = pd.DataFrame({
    "actual_late": y_test.to_numpy(),
    "predicted_late": test_predictions,
    "late_probability": test_probabilities
})

prediction_results.to_csv(
    REPORTS_DIR / "delivery_prediction_results.csv",
    index=False
)


# ============================================================
# 17. SAVE MODEL METRICS
# ============================================================

metrics = pd.DataFrame([{
    "total_test_orders": len(y_test),
    "actual_late_rate": y_test.mean(),
    "selected_threshold": best_threshold,
    "roc_auc": auc,
    "average_precision": avg_precision,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "accuracy": accuracy
}])

metrics.to_csv(
    REPORTS_DIR / "delivery_model_metrics.csv",
    index=False
)


# ============================================================
# 18. FEATURE IMPORTANCE
# ============================================================

importance = permutation_importance(
    model,
    X_val,
    y_val,
    n_repeats=3,
    random_state=42,
    scoring="average_precision",
    n_jobs=-1
)

feature_importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": importance.importances_mean
})

feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
)

feature_importance.to_csv(
    REPORTS_DIR / "feature_importance.csv",
    index=False
)

top_features = feature_importance.sort_values(
    "importance"
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.title("Permutation Feature Importance")
plt.xlabel("Decrease in Validation Average Precision")

plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "feature_importance.png",
    dpi=300
)

plt.close()


# ============================================================
# 19. THRESHOLD COMPARISON CHART
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    threshold_df["threshold"],
    threshold_df["precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    threshold_df["threshold"],
    threshold_df["recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    threshold_df["threshold"],
    threshold_df["f1_score"],
    marker="o",
    label="F1-score"
)

plt.title("Validation Performance by Threshold")
plt.xlabel("Threshold")
plt.ylabel("Score")

plt.ylim(0, 1)
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "threshold_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\nFINAL MODEL SUMMARY")
print("=" * 55)

print("Total analyzed orders:", len(data))
print("Training orders:", len(X_train))
print("Validation orders:", len(X_val))
print("Test orders:", len(X_test))

print("Test late delivery rate:", round(y_test.mean() * 100, 2), "%")
print("Selected threshold:", best_threshold)
print("Test ROC-AUC:", round(auc, 4))
print("Test F1-score:", round(f1, 4))

print("\nREPORTS SAVED:")
print("- threshold_comparison.csv")
print("- threshold_comparison.png")
print("- feature_importance.csv")
print("- feature_importance.png")
print("- delivery_prediction_results.csv")
print("- delivery_model_metrics.csv")
print("- delivery_confusion_matrix.png")

print("\nDELIVERY PREDICTION ANALYSIS COMPLETED!")
