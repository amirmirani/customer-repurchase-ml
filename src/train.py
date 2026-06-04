from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing training dependency. Install requirements with: "
        "pip install -r requirements.txt"
    ) from exc

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing xgboost. Install requirements with: pip install -r requirements.txt"
    ) from exc


TARGET_COLUMN = "label"
ID_COLUMNS = ["customer_id"]
CATEGORICAL_COLUMNS = ["favorite_category"]
RANDOM_STATE = 42


def split_first_80_last_20(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split_index = int(len(dataset) * 0.8)
    train_df = dataset.iloc[:split_index].copy()
    validation_df = dataset.iloc[split_index:].copy()

    x_train = train_df.drop(columns=ID_COLUMNS + [TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    x_validation = validation_df.drop(columns=ID_COLUMNS + [TARGET_COLUMN])
    y_validation = validation_df[TARGET_COLUMN]

    return x_train, x_validation, y_train, y_validation


def make_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    numeric_columns = [
        column for column in feature_columns if column not in CATEGORICAL_COLUMNS
    ]
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            (
                "favorite_category",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            ),
            ("numeric", numeric_pipeline, numeric_columns),
        ]
    )


def make_models(feature_columns: list[str]) -> dict[str, Pipeline]:
    preprocessor = make_preprocessor(feature_columns)
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1_000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", make_preprocessor(feature_columns)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_depth=10,
                        min_samples_leaf=10,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                ("preprocess", make_preprocessor(feature_columns)),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=250,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def get_model_hyperparameters(model: Pipeline) -> dict[str, object]:
    estimator = model.named_steps["model"]
    parameters = estimator.get_params()
    return {
        key: value
        for key, value in parameters.items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }


def calculate_metrics(
    y_true: pd.Series,
    probabilities: pd.Series,
    predictions: pd.Series,
) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def get_feature_names(model: Pipeline) -> list[str]:
    preprocessor = model.named_steps["preprocess"]
    return preprocessor.get_feature_names_out().tolist()


def get_feature_importance(model: Pipeline) -> pd.DataFrame:
    feature_names = get_feature_names(model)
    estimator = model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importance = abs(estimator.coef_[0])
    else:
        importance = [0.0] * len(feature_names)

    return (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def train_and_evaluate(
    dataset: pd.DataFrame,
    reports_dir: Path,
) -> tuple[
    str,
    Pipeline,
    dict[str, dict[str, float]],
    dict[str, list[dict[str, float]]],
    dict[str, str],
]:
    x_train, x_validation, y_train, y_validation = split_first_80_last_20(dataset)
    feature_columns = x_train.columns.tolist()
    models = make_models(feature_columns)

    metrics: dict[str, dict[str, float]] = {}
    importances: dict[str, list[dict[str, float]]] = {}
    trained_models: dict[str, Pipeline] = {}
    run_ids: dict[str, str] = {}

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name) as run:
            model.fit(x_train, y_train)
            probabilities = model.predict_proba(x_validation)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)

            model_metrics = calculate_metrics(y_validation, probabilities, predictions)
            metrics[model_name] = model_metrics
            feature_importance = get_feature_importance(model)
            importances[model_name] = (
                feature_importance.head(20)
                .round({"importance": 6})
                .to_dict(orient="records")
            )

            feature_list_path = reports_dir / f"{model_name}_feature_list.json"
            feature_list_payload = {
                "input_features": feature_columns,
                "transformed_features": get_feature_names(model),
            }
            with feature_list_path.open("w", encoding="utf-8") as file:
                json.dump(feature_list_payload, file, indent=2)

            mlflow.log_param("model_name", model_name)
            mlflow.log_params(get_model_hyperparameters(model))
            mlflow.log_metrics(model_metrics)
            mlflow.log_artifact(feature_list_path, artifact_path="features")
            mlflow.sklearn.log_model(model, artifact_path="model")

            run_ids[model_name] = run.info.run_id
            trained_models[model_name] = model

    best_model_name = max(metrics, key=lambda name: metrics[name]["roc_auc"])
    return best_model_name, trained_models[best_model_name], metrics, importances, run_ids


def calculate_baseline_metrics(y_true: pd.Series) -> dict[str, float]:
    probabilities = pd.Series(0.0, index=y_true.index)
    predictions = pd.Series(0, index=y_true.index)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def build_model_comparison(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    comparison = (
        pd.DataFrame.from_dict(metrics, orient="index")
        .reset_index()
        .rename(columns={"index": "model"})
        .sort_values("roc_auc", ascending=False)
        .reset_index(drop=True)
    )
    return comparison


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "training_dataset.csv"
    model_path = project_root / "models" / "model.pkl"
    metrics_path = project_root / "reports" / "metrics.json"
    reports_dir = project_root / "reports"
    comparison_path = reports_dir / "model_comparison.csv"
    best_run_id_path = reports_dir / "best_model_run_id.txt"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("customer_repurchase_prediction")

    dataset = pd.read_csv(dataset_path)
    best_model_name, best_model, metrics, importances, run_ids = train_and_evaluate(
        dataset,
        reports_dir=reports_dir,
    )
    _, _, _, y_validation = split_first_80_last_20(dataset)
    all_metrics = {"baseline_all_negative": calculate_baseline_metrics(y_validation), **metrics}
    comparison = build_model_comparison(all_metrics)
    best_run_id = run_ids[best_model_name]

    with model_path.open("wb") as file:
        pickle.dump(best_model, file)

    metrics_payload = {
        "split": {
            "strategy": "first_80_percent_train_last_20_percent_validation",
            "train_rows": int(len(dataset) * 0.8),
            "validation_rows": int(len(dataset) - int(len(dataset) * 0.8)),
        },
        "best_model": best_model_name,
        "best_model_run_id": best_run_id,
        "mlflow_tracking_uri": "./mlruns",
        "mlflow_experiment": "customer_repurchase_prediction",
        "mlflow_run_ids": run_ids,
        "metrics": all_metrics,
        "feature_importance": importances,
    }
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, indent=2)
    comparison.to_csv(comparison_path, index=False)
    best_run_id_path.write_text(best_run_id, encoding="utf-8")

    print("Training completed")
    print(f"Rows: {len(dataset):,}")
    print(f"Best model: {best_model_name}")
    print()
    print("Metrics:")
    print(json.dumps(all_metrics, indent=2))
    print()
    print("Model comparison:")
    print(comparison.to_string(index=False))
    print()
    print("Feature importance:")
    for model_name, model_importances in importances.items():
        print(f"\n{model_name}")
        for row in model_importances[:10]:
            print(f"  {row['feature']}: {row['importance']}")
    print()
    print(f"Saved best model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved model comparison to: {comparison_path}")
    print(f"Saved best model run id to: {best_run_id_path}")
    print()
    print("MLflow run summary:")
    print(f"Tracking URI: ./mlruns")
    print(f"Experiment: customer_repurchase_prediction")
    print(f"Best model run id: {best_run_id}")
    for model_name, run_id in run_ids.items():
        metric_summary = metrics[model_name]
        print(
            f"  {model_name}: run_id={run_id}, "
            f"roc_auc={metric_summary['roc_auc']:.4f}, "
            f"pr_auc={metric_summary['pr_auc']:.4f}, "
            f"f1={metric_summary['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
