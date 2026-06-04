from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
N_CUSTOMERS = 5_000
N_ORDERS = 50_000
START_DATE = "2024-12-04"
END_DATE = "2026-06-03"


CATEGORIES = {
    "Electronics": ["Apple", "Samsung", "Sony", "Anker", "Logitech"],
    "Fashion": ["Nike", "Adidas", "Zara", "Levi's", "Uniqlo"],
    "Home": ["Ikea", "Philips", "Dyson", "OXO", "Ninja"],
    "Beauty": ["L'Oreal", "Nivea", "Dove", "CeraVe", "Maybelline"],
    "Sports": ["Puma", "Under Armour", "Wilson", "Reebok", "Columbia"],
    "Books": ["Penguin", "HarperCollins", "O'Reilly", "No Starch", "Pearson"],
    "Grocery": ["Nestle", "Kellogg's", "PepsiCo", "Heinz", "Quaker"],
    "Toys": ["Lego", "Mattel", "Hasbro", "Fisher-Price", "Ravensburger"],
}

CATEGORY_PRICE_RANGES = {
    "Electronics": (35, 850),
    "Fashion": (15, 220),
    "Home": (20, 420),
    "Beauty": (8, 95),
    "Sports": (12, 260),
    "Books": (6, 80),
    "Grocery": (3, 70),
    "Toys": (8, 160),
}


def build_customer_profiles(rng: np.random.Generator) -> pd.DataFrame:
    customer_ids = np.arange(1, N_CUSTOMERS + 1)
    segments = rng.choice(
        ["frequent", "churned", "high_value", "regular"],
        size=N_CUSTOMERS,
        p=[0.18, 0.22, 0.10, 0.50],
    )

    # Segment weights create realistic purchase concentration without allowing
    # any single segment to dominate the whole order history.
    base_weight = np.select(
        [
            segments == "frequent",
            segments == "churned",
            segments == "high_value",
            segments == "regular",
        ],
        [3.2, 0.9, 2.0, 1.25],
    )
    customer_variation = rng.lognormal(mean=0.0, sigma=0.55, size=N_CUSTOMERS)

    return pd.DataFrame(
        {
            "customer_id": customer_ids,
            "segment": segments,
            "order_weight": base_weight * customer_variation,
        }
    )


def allocate_orders(profiles: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    weights = profiles["order_weight"].to_numpy()
    probabilities = weights / weights.sum()

    # Give every customer at least one order, then distribute the remaining
    # orders according to segment-specific purchase propensity.
    extra_orders = rng.multinomial(N_ORDERS - N_CUSTOMERS, probabilities)
    return extra_orders + 1


def random_order_dates(
    segment: str,
    n_orders: int,
    rng: np.random.Generator,
    start: np.datetime64,
    end: np.datetime64,
) -> np.ndarray:
    total_days = int((end - start).astype("timedelta64[D]").astype(int)) + 1

    if segment == "churned":
        # Churned customers mostly bought in the first year, with very little
        # recent activity.
        day_offsets = (rng.beta(1.15, 3.8, size=n_orders) * (total_days - 1)).astype(int)
    elif segment == "frequent":
        # Frequent customers keep buying across the full period.
        day_offsets = rng.integers(0, total_days, size=n_orders)
    elif segment == "high_value":
        # High-value customers skew slightly recent, useful for repurchase labels.
        day_offsets = (rng.beta(2.2, 1.8, size=n_orders) * (total_days - 1)).astype(int)
    else:
        day_offsets = (rng.beta(1.4, 1.6, size=n_orders) * (total_days - 1)).astype(int)

    return start + day_offsets.astype("timedelta64[D]")


def choose_product(
    segment: str,
    rng: np.random.Generator,
) -> tuple[str, str, str, float, int]:
    if segment == "high_value":
        category_probs = [0.32, 0.12, 0.18, 0.08, 0.12, 0.04, 0.05, 0.09]
    elif segment == "frequent":
        category_probs = [0.12, 0.18, 0.10, 0.15, 0.11, 0.08, 0.20, 0.06]
    elif segment == "churned":
        category_probs = [0.08, 0.20, 0.09, 0.14, 0.07, 0.12, 0.24, 0.06]
    else:
        category_probs = [0.14, 0.17, 0.12, 0.13, 0.10, 0.10, 0.17, 0.07]

    category = rng.choice(list(CATEGORIES.keys()), p=category_probs)
    brand = rng.choice(CATEGORIES[category])
    product_suffix = rng.integers(1, 800)
    brand_code = brand[:3].upper().replace("'", "")
    product_id = f"{category[:3].upper()}-{brand_code}-{product_suffix:04d}"

    min_price, max_price = CATEGORY_PRICE_RANGES[category]
    unit_price = rng.lognormal(mean=np.log((min_price + max_price) / 3), sigma=0.55)
    unit_price = float(np.clip(unit_price, min_price, max_price))

    quantity_probs = [0.68, 0.20, 0.08, 0.03, 0.01]
    quantity = int(rng.choice([1, 2, 3, 4, 5], p=quantity_probs))

    if segment == "high_value":
        unit_price *= rng.uniform(1.25, 2.1)
        if rng.random() < 0.18:
            quantity += int(rng.integers(1, 4))
    elif segment == "frequent":
        unit_price *= rng.uniform(0.85, 1.25)
    elif segment == "churned":
        unit_price *= rng.uniform(0.75, 1.05)

    amount = round(unit_price * quantity, 2)
    return product_id, category, brand, amount, quantity


def generate_orders() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    profiles = build_customer_profiles(rng)
    order_counts = allocate_orders(profiles, rng)

    start = np.datetime64(START_DATE)
    end = np.datetime64(END_DATE)
    records = []
    order_number = 1

    for profile, n_orders in zip(profiles.itertuples(index=False), order_counts):
        dates = np.sort(random_order_dates(profile.segment, int(n_orders), rng, start, end))

        for order_date in dates:
            product_id, category, brand, amount, quantity = choose_product(profile.segment, rng)
            records.append(
                {
                    "customer_id": f"C{profile.customer_id:05d}",
                    "order_id": f"O{order_number:07d}",
                    "order_date": pd.Timestamp(order_date).strftime("%Y-%m-%d"),
                    "product_id": product_id,
                    "category": category,
                    "brand": brand,
                    "amount": amount,
                    "quantity": quantity,
                }
            )
            order_number += 1

    orders = pd.DataFrame.from_records(records)
    return orders.sample(frac=1, random_state=SEED).reset_index(drop=True)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "data" / "orders.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    orders = generate_orders()
    orders.to_csv(output_path, index=False)

    print(f"Wrote {len(orders):,} orders to {output_path}")
    print(f"Customers: {orders['customer_id'].nunique():,}")
    print(f"Date range: {orders['order_date'].min()} to {orders['order_date'].max()}")


if __name__ == "__main__":
    main()
