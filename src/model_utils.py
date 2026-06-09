# Утилиты для сравнения моделей и сохранения результатов.

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from src.data_utils import RANDOM_STATE

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


# Считаем метрики(регр)
def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# Считает метрики(класс)
def classification_metrics(y_true, y_pred, y_proba=None) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return metrics


# Возвращает модели и сетки гиперпараметров для регрессии
def get_regression_models() -> dict:
    return {
        "Ridge": (
            Ridge(),
            {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
        "ElasticNet": (
            ElasticNet(max_iter=5000, random_state=RANDOM_STATE),
            {
                "model__alpha": [0.001, 0.01, 0.1, 1.0],
                "model__l1_ratio": [0.2, 0.5, 0.8],
            },
        ),
        "RandomForest": (
            RandomForestRegressor(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_leaf": [1, 3],
            },
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3, 4],
            },
        ),
        "SVR": (
            SVR(),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__gamma": ["scale", "auto"],
                "model__epsilon": [0.01, 0.1],
            },
        ),
    }


# Возвращает модели и сетки гиперпараметров для классификации
def get_classification_models() -> dict:
    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
            {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__penalty": ["l2"],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_leaf": [1, 3],
            },
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
        "SVC": (
            SVC(probability=True, random_state=RANDOM_STATE),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__gamma": ["scale", "auto"],
                "model__kernel": ["rbf"],
            },
        ),
    }


# Подбирает гиперпараметры регрессионных моделей
def tune_regression_models(x_train, y_train) -> pd.DataFrame:
    results = []
    models = get_regression_models()
    for name, (estimator, param_grid) in models.items():
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
        search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        search.fit(x_train, y_train)
        cv_rmse = -search.best_score_
        results.append(
            {
                "model": name,
                "best_params": search.best_params_,
                "cv_rmse": cv_rmse,
                "best_estimator": search.best_estimator_,
            }
        )
    return pd.DataFrame(results)


# Подбирает гиперпараметры классификационных моделей
def tune_classification_models(x_train, y_train) -> pd.DataFrame:
    results = []
    models = get_classification_models()
    for name, (estimator, param_grid) in models.items():
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("model", estimator)]
        )
        search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring="f1",
            n_jobs=-1,
        )
        search.fit(x_train, y_train)
        results.append(
            {
                "model": name,
                "best_params": search.best_params_,
                "cv_f1": search.best_score_,
                "best_estimator": search.best_estimator_,
            }
        )
    return pd.DataFrame(results)


# Сохраняем результаты эксперимента в JSON
def save_results(task_name: str, payload: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{task_name}.json"

    def default(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    serializable = json.loads(json.dumps(payload, default=default))
    with open(path, "w", encoding="utf-8") as file:
        json.dump(serializable, file, ensure_ascii=False, indent=2)
    return path


# Извлекаем важность признаков из обученной модели
def get_feature_importance(model, feature_columns: list[str], top_n: int = 15) -> pd.DataFrame:
    estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_).ravel()
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    importance = pd.DataFrame({"feature": feature_columns, "importance": values})
    return importance.sort_values("importance", ascending=False).head(top_n)
