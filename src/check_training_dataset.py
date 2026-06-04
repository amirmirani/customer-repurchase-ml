from __future__ import annotations

from pathlib import Path

import pandas as pd

from feature_engineering import build_training_dataset


ANCHOR_DATE = "2026-03-01"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    orders_path = project_root / "data" / "orders.csv"
    output_path = project_root / "data" / "training_dataset.csv"

    orders = pd.read_csv(orders_path)
    training_dataset = build_training_dataset(
        orders,
        anchor_date=ANCHOR_DATE,
    )

    feature_columns = [
        column
        for column in training_dataset.columns
        if column not in {"customer_id", "label"}
    ]
    label_distribution = training_dataset["label"].value_counts().sort_index()
    positive_label_rate = training_dataset["label"].mean()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    training_dataset.to_csv(output_path, index=False)

    print(f"Anchor date: {ANCHOR_DATE}")
    print(f"Rows: {len(training_dataset):,}")
    print(f"Features: {len(feature_columns):,}")
    print(f"Positive label rate: {positive_label_rate:.4f}")
    print("Label distribution:")
    print(label_distribution.to_string())
    print()
    print("First 10 rows:")
    print(training_dataset.head(10).to_string(index=False))
    print()
    print(f"Saved dataset to: {output_path}")


if __name__ == "__main__":
    main()
