from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    orders_path = project_root / "data" / "orders.csv"
    report_path = project_root / "reports" / "customer_variety_summary.csv"

    orders = pd.read_csv(orders_path, parse_dates=["order_date"])

    customer_summary = (
        orders.groupby("customer_id")
        .agg(
            total_orders=("order_id", "nunique"),
            unique_products=("product_id", "nunique"),
            unique_categories=("category", "nunique"),
            unique_brands=("brand", "nunique"),
            total_amount=("amount", "sum"),
            avg_order_amount=("amount", "mean"),
            total_quantity=("quantity", "sum"),
            first_order_date=("order_date", "min"),
            last_order_date=("order_date", "max"),
        )
        .reset_index()
    )

    customer_summary["active_days"] = (
        customer_summary["last_order_date"] - customer_summary["first_order_date"]
    ).dt.days + 1

    customer_summary["orders_per_active_month"] = (
        customer_summary["total_orders"] / (customer_summary["active_days"] / 30.44)
    ).round(2)

    customer_summary = customer_summary.sort_values(
        ["total_orders", "unique_products", "total_amount"],
        ascending=False,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    customer_summary.to_csv(report_path, index=False)

    print("Customer variety report")
    print("=" * 80)
    print(f"Orders file: {orders_path}")
    print(f"Rows: {len(orders):,}")
    print(f"Customers: {orders['customer_id'].nunique():,}")
    print(f"Report saved to: {report_path}")
    print()

    print("Overall variety stats")
    print("-" * 80)
    print(customer_summary.describe().round(2).to_string())
    print()

    print("Top 10 customers by order/product variety")
    print("-" * 80)
    columns = [
        "customer_id",
        "total_orders",
        "unique_products",
        "unique_categories",
        "unique_brands",
        "total_amount",
        "orders_per_active_month",
    ]
    print(customer_summary[columns].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
