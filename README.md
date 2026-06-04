# Customer Repurchase ML

Synthetic e-commerce order data and starter project structure for customer repurchase prediction.

## Project Structure

- `data/` - Generated datasets
- `notebooks/` - Exploratory notebooks
- `src/` - Data generation and modeling code
- `models/` - Saved model artifacts
- `reports/` - Analysis outputs
- `tests/` - Automated tests
- `api/` - API service code

## Generate Data

```bash
python src/generate_orders.py
```

The generator uses random seed `42` and writes:

```text
data/orders.csv
```

Dataset shape:

- 5,000 customers
- 50,000 orders
- 18 months of order history
- Frequent, churned, high-value, and regular customer behavior patterns
