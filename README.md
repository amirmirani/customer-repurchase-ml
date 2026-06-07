# Customer Repurchase ML

End-to-end customer repurchase prediction project using synthetic e-commerce order data.

The project includes:

- Synthetic order data generation
- Customer-level feature engineering
- Exploratory analysis and charts
- Model training and evaluation
- Baseline comparison
- MLflow experiment tracking
- FastAPI inference service
- Docker and Docker Compose support

## Project Structure

```text
customer-repurchase-ml/
├── api/                  # FastAPI inference service
├── data/                 # Orders and training datasets
├── models/               # Model artifact and feature schema
├── notebooks/            # Exploratory analysis notebook
├── reports/              # Metrics, summaries, and charts
├── src/                  # Data, feature, EDA, and training code
├── tests/                # Automated tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Generate Synthetic Orders

```bash
python src/generate_orders.py
```

Output:

```text
data/orders.csv
```

Dataset shape:

- 5,000 customers
- 50,000 orders
- 18 months of history
- Frequent, churned, high-value, and regular customer behavior patterns

## Exploratory Analysis

Generate EDA reports and charts:

```bash
python src/eda_analysis.py
```

Notebook:

```text
notebooks/01_exploratory_analysis.ipynb
```

Reports include:

- Orders per customer distribution
- Revenue distribution
- Category distribution
- Monthly orders trend
- Top customers by revenue

## Feature Engineering

Build a customer-level training dataset:

```bash
python src/check_training_dataset.py
```

Output:

```text
data/training_dataset.csv
```

Features:

- `orders_last_30d`
- `orders_last_90d`
- `total_spend_90d`
- `avg_order_value_90d`
- `days_since_last_order`
- `customer_tenure_days`
- `category_diversity_90d`
- `favorite_category`
- `avg_quantity_per_order`

Label:

- `1` if the customer purchases in the next 30 days
- `0` otherwise

The feature builder prevents data leakage by using only orders on or before the anchor date for features, and only future orders after the anchor date for labels.

## Train Models

```bash
python src/train.py
```

Models trained:

- Logistic Regression
- Random Forest
- XGBoost

Metrics calculated:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1

Outputs:

```text
models/model.pkl
reports/metrics.json
reports/model_comparison.csv
reports/best_model_run_id.txt
```

The current best model is the Random Forest model saved at:

```text
models/model.pkl
```

## MLflow Tracking

The training pipeline logs each model run to local MLflow tracking:

```text
./mlruns
```

Experiment name:

```text
customer_repurchase_prediction
```

To view the MLflow UI locally after running training:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open:

```text
http://127.0.0.1:5000
```

Note: `mlruns/` is ignored by Git to keep the repository clean. Core metrics and the best run id are saved under `reports/`.

## FastAPI Inference Service

Run locally:

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

Health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

Prediction example:

```bash
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"orders_last_30d\":2,\"orders_last_90d\":5,\"total_spend_90d\":3500000,\"avg_order_value_90d\":700000,\"days_since_last_order\":12,\"customer_tenure_days\":300,\"category_diversity_90d\":4,\"avg_quantity_per_order\":2,\"favorite_category\":\"Electronics\"}"
```

Example response:

```json
{
  "repurchase_probability": 0.7814,
  "prediction": 1
}
```

The API loads:

```text
models/model.pkl
models/features.json
```

Missing request features are filled with defaults from `models/features.json`.

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

Open:

```text
http://127.0.0.1:8000/docs
```

## Tests

Run tests:

```bash
pytest tests
```

## Important Files

- `src/generate_orders.py` - synthetic order data generator
- `src/feature_engineering.py` - customer-level feature builder
- `src/check_training_dataset.py` - training dataset creation script
- `src/train.py` - model training, MLflow tracking, metrics, and model export
- `api/main.py` - FastAPI app
- `models/model.pkl` - best trained model
- `models/features.json` - model input feature schema and defaults
- `reports/model_comparison.csv` - baseline and model comparison
