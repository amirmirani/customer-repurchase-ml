import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.feature_engineering import build_training_dataset


def test_build_training_dataset_avoids_label_leakage():
    orders = pd.DataFrame(
        [
            {
                "customer_id": "C001",
                "order_id": "O001",
                "order_date": "2026-01-01",
                "category": "Books",
                "amount": 20.0,
                "quantity": 1,
            },
            {
                "customer_id": "C001",
                "order_id": "O002",
                "order_date": "2026-01-20",
                "category": "Electronics",
                "amount": 100.0,
                "quantity": 2,
            },
            {
                "customer_id": "C001",
                "order_id": "O003",
                "order_date": "2026-02-05",
                "category": "Fashion",
                "amount": 80.0,
                "quantity": 1,
            },
            {
                "customer_id": "C002",
                "order_id": "O004",
                "order_date": "2025-10-01",
                "category": "Home",
                "amount": 50.0,
                "quantity": 1,
            },
        ]
    )

    result = build_training_dataset(
        orders,
        anchor_date="2026-01-31",
        feature_window_days=30,
        label_window_days=30,
    ).set_index("customer_id")

    assert result.loc["C001", "orders_last_30d"] == 1
    assert result.loc["C001", "orders_last_90d"] == 1
    assert result.loc["C001", "total_spend_90d"] == 100.0
    assert result.loc["C001", "avg_order_value_90d"] == 100.0
    assert result.loc["C001", "days_since_last_order"] == 11
    assert result.loc["C001", "customer_tenure_days"] == 30
    assert result.loc["C001", "category_diversity_90d"] == 1
    assert result.loc["C001", "favorite_category"] == "Electronics"
    assert result.loc["C001", "avg_quantity_per_order"] == 2.0
    assert result.loc["C001", "label"] == 1

    assert result.loc["C002", "orders_last_90d"] == 0
    assert result.loc["C002", "total_spend_90d"] == 0.0
    assert result.loc["C002", "favorite_category"] == "unknown"
    assert result.loc["C002", "label"] == 0
