from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


WIDTH = 1100
HEIGHT = 720
MARGIN_LEFT = 115
MARGIN_RIGHT = 55
MARGIN_TOP = 85
MARGIN_BOTTOM = 95

BG = "#ffffff"
TEXT = "#202124"
MUTED = "#5f6368"
GRID = "#e8eaed"
BLUE = "#2563eb"
GREEN = "#059669"
ORANGE = "#d97706"
PURPLE = "#7c3aed"
RED = "#dc2626"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


TITLE_FONT = font(28, bold=True)
LABEL_FONT = font(18)
SMALL_FONT = font(14)


def save_image(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def base_canvas(title: str, subtitle: str | None = None) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.text((35, 28), title, fill=TEXT, font=TITLE_FONT)
    if subtitle:
        draw.text((35, 61), subtitle, fill=MUTED, font=SMALL_FONT)
    return image, draw


def nice_tick(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def draw_axes(draw: ImageDraw.ImageDraw, y_max: float) -> None:
    x0 = MARGIN_LEFT
    y0 = HEIGHT - MARGIN_BOTTOM
    x1 = WIDTH - MARGIN_RIGHT
    y1 = MARGIN_TOP

    draw.line((x0, y0, x1, y0), fill=TEXT, width=2)
    draw.line((x0, y0, x0, y1), fill=TEXT, width=2)

    tick_count = 5
    for i in range(tick_count + 1):
        value = y_max * i / tick_count
        y = y0 - (y0 - y1) * i / tick_count
        draw.line((x0, y, x1, y), fill=GRID, width=1)
        label = nice_tick(value)
        bbox = draw.textbbox((0, 0), label, font=SMALL_FONT)
        draw.text((x0 - 12 - (bbox[2] - bbox[0]), y - 8), label, fill=MUTED, font=SMALL_FONT)


def bar_chart(
    values: pd.Series,
    title: str,
    output_path: Path,
    color: str = BLUE,
    subtitle: str | None = None,
    max_labels: int = 18,
) -> None:
    values = values.head(max_labels)
    image, draw = base_canvas(title, subtitle)
    y_max = max(float(values.max()), 1.0)
    draw_axes(draw, y_max)

    plot_x0 = MARGIN_LEFT + 18
    plot_x1 = WIDTH - MARGIN_RIGHT - 15
    plot_y0 = HEIGHT - MARGIN_BOTTOM
    plot_y1 = MARGIN_TOP
    plot_w = plot_x1 - plot_x0
    plot_h = plot_y0 - plot_y1
    gap = 8
    bar_w = max(8, (plot_w - gap * (len(values) - 1)) / len(values))

    for i, (label, value) in enumerate(values.items()):
        x = plot_x0 + i * (bar_w + gap)
        bar_h = plot_h * float(value) / y_max
        y = plot_y0 - bar_h
        draw.rounded_rectangle((x, y, x + bar_w, plot_y0), radius=4, fill=color)

        text = str(label)
        if len(text) > 12:
            text = text[:11] + "."
        bbox = draw.textbbox((0, 0), text, font=SMALL_FONT)
        tx = x + (bar_w - (bbox[2] - bbox[0])) / 2
        draw.text((tx, plot_y0 + 14), text, fill=MUTED, font=SMALL_FONT)

    save_image(image, output_path)


def horizontal_bar_chart(
    values: pd.Series,
    title: str,
    output_path: Path,
    color: str = GREEN,
    subtitle: str | None = None,
) -> None:
    values = values.sort_values(ascending=True)
    image, draw = base_canvas(title, subtitle)

    x0 = 210
    x1 = WIDTH - 150
    y0 = HEIGHT - 85
    y1 = MARGIN_TOP + 10
    max_value = max(float(values.max()), 1.0)
    gap = 12
    bar_h = max(16, ((y0 - y1) - gap * (len(values) - 1)) / len(values))

    for i, (label, value) in enumerate(values.items()):
        y = y0 - (i + 1) * bar_h - i * gap
        bar_w = (x1 - x0) * float(value) / max_value
        draw.text((40, y + 2), str(label), fill=TEXT, font=LABEL_FONT)
        draw.rounded_rectangle((x0, y, x0 + bar_w, y + bar_h), radius=5, fill=color)
        draw.text((x0 + bar_w + 10, y + 2), f"${value:,.0f}", fill=MUTED, font=SMALL_FONT)

    draw.line((x0, y0 + 8, x1, y0 + 8), fill=TEXT, width=2)
    save_image(image, output_path)


def histogram(
    series: pd.Series,
    bins: int,
    title: str,
    output_path: Path,
    color: str,
    subtitle: str | None = None,
) -> None:
    counts, edges = pd.cut(series, bins=bins, retbins=True, include_lowest=True)
    values = counts.value_counts().sort_index()
    labels = [f"{math.floor(edges[i])}-{math.floor(edges[i + 1])}" for i in range(len(edges) - 1)]
    values.index = labels
    bar_chart(values, title, output_path, color=color, subtitle=subtitle, max_labels=bins)


def line_chart(
    values: pd.Series,
    title: str,
    output_path: Path,
    color: str = RED,
    subtitle: str | None = None,
) -> None:
    image, draw = base_canvas(title, subtitle)
    y_max = max(float(values.max()), 1.0)
    draw_axes(draw, y_max)

    plot_x0 = MARGIN_LEFT + 18
    plot_x1 = WIDTH - MARGIN_RIGHT - 15
    plot_y0 = HEIGHT - MARGIN_BOTTOM
    plot_y1 = MARGIN_TOP
    plot_w = plot_x1 - plot_x0
    plot_h = plot_y0 - plot_y1

    points = []
    for i, value in enumerate(values):
        x = plot_x0 + plot_w * i / max(len(values) - 1, 1)
        y = plot_y0 - plot_h * float(value) / y_max
        points.append((x, y))

    if len(points) > 1:
        draw.line(points, fill=color, width=4)

    for x, y in points:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)

    label_step = max(1, len(values) // 9)
    for i, label in enumerate(values.index.astype(str)):
        if i % label_step != 0 and i != len(values) - 1:
            continue
        x = plot_x0 + plot_w * i / max(len(values) - 1, 1)
        draw.text((x - 28, plot_y0 + 14), label, fill=MUTED, font=SMALL_FONT)

    save_image(image, output_path)


def run_eda() -> dict[str, Path]:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "orders.csv"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    orders = pd.read_csv(data_path, parse_dates=["order_date"])
    customer_metrics = (
        orders.groupby("customer_id")
        .agg(
            orders=("order_id", "nunique"),
            revenue=("amount", "sum"),
            avg_order_value=("amount", "mean"),
        )
        .sort_values("revenue", ascending=False)
    )
    customer_metrics.to_csv(reports_dir / "customer_revenue_summary.csv")

    category_summary = (
        orders.groupby("category")
        .agg(orders=("order_id", "nunique"), revenue=("amount", "sum"))
        .sort_values("orders", ascending=False)
    )
    category_summary.to_csv(reports_dir / "category_summary.csv")

    monthly_orders = orders.set_index("order_date").resample("MS")["order_id"].nunique()
    monthly_orders.index = monthly_orders.index.strftime("%Y-%m")
    monthly_orders.to_csv(reports_dir / "monthly_orders.csv", header=["orders"])

    chart_paths = {
        "orders_per_customer": reports_dir / "orders_per_customer_distribution.png",
        "revenue_distribution": reports_dir / "revenue_distribution.png",
        "category_distribution": reports_dir / "category_distribution.png",
        "monthly_orders_trend": reports_dir / "monthly_orders_trend.png",
        "top_customers_by_revenue": reports_dir / "top_customers_by_revenue.png",
    }

    histogram(
        customer_metrics["orders"],
        bins=16,
        title="Orders per customer distribution",
        subtitle="Number of customers in each order-count range",
        output_path=chart_paths["orders_per_customer"],
        color=BLUE,
    )
    histogram(
        customer_metrics["revenue"],
        bins=16,
        title="Revenue per customer distribution",
        subtitle="Number of customers in each revenue range",
        output_path=chart_paths["revenue_distribution"],
        color=GREEN,
    )
    bar_chart(
        category_summary["orders"],
        title="Category distribution",
        subtitle="Order count by product category",
        output_path=chart_paths["category_distribution"],
        color=ORANGE,
    )
    line_chart(
        monthly_orders,
        title="Monthly orders trend",
        subtitle="Orders by month across the 18-month history",
        output_path=chart_paths["monthly_orders_trend"],
        color=RED,
    )
    horizontal_bar_chart(
        customer_metrics["revenue"].head(10),
        title="Top customers by revenue",
        subtitle="Top 10 customers ranked by total purchase amount",
        output_path=chart_paths["top_customers_by_revenue"],
        color=PURPLE,
    )

    print("EDA completed")
    print(f"Orders: {len(orders):,}")
    print(f"Customers: {orders['customer_id'].nunique():,}")
    print(f"Reports directory: {reports_dir}")
    for name, path in chart_paths.items():
        print(f"{name}: {path}")

    return chart_paths


if __name__ == "__main__":
    run_eda()
