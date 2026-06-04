from __future__ import annotations

from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {
    "customer_id",
    "order_id",
    "order_date",
    "category",
    "amount",
    "quantity",
}


def _validate_inputs(
    orders_df: pd.DataFrame,
    anchor_date: Any,
    feature_window_days: int,
    label_window_days: int,
) -> pd.Timestamp:
    missing_columns = REQUIRED_COLUMNS - set(orders_df.columns)
    if missing_columns:
        columns = ", ".join(sorted(missing_columns))
        raise ValueError(f"orders_df is missing required columns: {columns}")

    if feature_window_days <= 0:
        raise ValueError("feature_window_days must be greater than 0")
    if label_window_days <= 0:
        raise ValueError("label_window_days must be greater than 0")

    return pd.Timestamp(anchor_date).normalize()


def _favorite_category(feature_orders: pd.DataFrame) -> pd.Series:
    if feature_orders.empty:
        return pd.Series(dtype="object", name="favorite_category")

    category_counts = (
        feature_orders.groupby(["customer_id", "category"])
        .size()
        .rename("category_orders")
        .reset_index()
        .sort_values(["customer_id", "category_orders", "category"], ascending=[True, False, True])
    )
    return category_counts.drop_duplicates("customer_id").set_index("customer_id")["category"]


def build_training_dataset(
    orders_df: pd.DataFrame,
    anchor_date: Any,
    feature_window_days: int = 90,
    label_window_days: int = 30,
) -> pd.DataFrame:
    """Build customer-level features and a future repurchase label.

    Feature values are calculated only from orders on or before ``anchor_date``.
    The label is calculated only from orders after ``anchor_date`` and within
    ``label_window_days``. This split prevents future behavior from leaking
    into model features.
    """
    anchor = _validate_inputs(orders_df, anchor_date, feature_window_days, label_window_days)
    orders = orders_df.copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"]).dt.normalize()

    all_customers = pd.Index(orders["customer_id"].dropna().unique(), name="customer_id")

    feature_start = anchor - pd.Timedelta(days=feature_window_days)
    last_30_start = anchor - pd.Timedelta(days=30)
    label_end = anchor + pd.Timedelta(days=label_window_days)

    historical_orders = orders[orders["order_date"] <= anchor]
    feature_orders = orders[
        (orders["order_date"] > feature_start)
        & (orders["order_date"] <= anchor)
    ]
    orders_last_30 = orders[
        (orders["order_date"] > last_30_start)
        & (orders["order_date"] <= anchor)
    ]
    label_orders = orders[
        (orders["order_date"] > anchor)
        & (orders["order_date"] <= label_end)
    ]

    features = pd.DataFrame(index=all_customers).sort_index()

    features["orders_last_30d"] = orders_last_30.groupby("customer_id")["order_id"].nunique()
    features["orders_last_90d"] = feature_orders.groupby("customer_id")["order_id"].nunique()
    features["total_spend_90d"] = feature_orders.groupby("customer_id")["amount"].sum()
    features["avg_order_value_90d"] = feature_orders.groupby("customer_id")["amount"].mean()
    features["category_diversity_90d"] = feature_orders.groupby("customer_id")["category"].nunique()
    features["avg_quantity_per_order"] = feature_orders.groupby("customer_id")["quantity"].mean()

    last_order_date = historical_orders.groupby("customer_id")["order_date"].max()
    first_order_date = historical_orders.groupby("customer_id")["order_date"].min()
    features["days_since_last_order"] = (anchor - last_order_date).dt.days
    features["customer_tenure_days"] = (anchor - first_order_date).dt.days
    features["favorite_category"] = _favorite_category(feature_orders)

    labels = label_orders.groupby("customer_id")["order_id"].nunique().gt(0).astype(int)
    features["label"] = labels

    numeric_defaults = {
        "orders_last_30d": 0,
        "orders_last_90d": 0,
        "total_spend_90d": 0.0,
        "avg_order_value_90d": 0.0,
        "category_diversity_90d": 0,
        "avg_quantity_per_order": 0.0,
        "label": 0,
    }
    features = features.fillna(numeric_defaults)
    features["favorite_category"] = features["favorite_category"].fillna("unknown")

    integer_columns = [
        "orders_last_30d",
        "orders_last_90d",
        "category_diversity_90d",
        "label",
    ]
    features[integer_columns] = features[integer_columns].astype(int)

    return features.reset_index()
