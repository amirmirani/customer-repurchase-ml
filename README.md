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

## FastAPI Inference Service

Run locally:

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

## Docker

Build the image:

```bash
docker build -t customer-repurchase-api .
```

Run with Docker:

```bash
docker run --rm -p 8000:8000 -v "${PWD}/models:/app/models:ro" customer-repurchase-api
```

Run with Docker Compose:

```bash
docker compose up --build
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```
