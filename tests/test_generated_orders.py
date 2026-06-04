from pathlib import Path

import pandas as pd


def test_orders_csv_schema_and_size():
    orders_path = Path(__file__).resolve().parents[1] / "data" / "orders.csv"
    orders = pd.read_csv(orders_path)

    assert len(orders) == 50_000
    assert orders["customer_id"].nunique() == 5_000
    assert orders["order_id"].nunique() == 50_000
    assert list(orders.columns) == [
        "customer_id",
        "order_id",
        "order_date",
        "product_id",
        "category",
        "brand",
        "amount",
        "quantity",
    ]
    assert orders["order_date"].min() == "2024-12-04"
    assert orders["order_date"].max() == "2026-06-03"
