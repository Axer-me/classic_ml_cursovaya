# Регрессия для предсказания CC50.

from src.data_utils import load_raw_data, make_regression_split
from src.model_utils import (
    get_feature_importance,
    regression_metrics,
    save_results,
    tune_regression_models,
)


def main() -> None:
    df = load_raw_data()
    split = make_regression_split(df, "cc50", log_transform=True)

    print("Подбор гиперпараметров для CC50 (log1p)...")
    tuning = tune_regression_models(split["x_train"], split["y_train"])
    tuning_sorted = tuning.sort_values("cv_rmse")
    print(tuning_sorted[["model", "cv_rmse"]].to_string(index=False))

    best_row = tuning_sorted.iloc[0]
    best_model = best_row["best_estimator"]
    y_pred = best_model.predict(split["x_test"])
    test_metrics = regression_metrics(split["y_test"], y_pred)
    print(f"\nЛучшая модель: {best_row['model']}")
    print(f"Тестовые метрики (log-шкала): {test_metrics}")

    importance = get_feature_importance(best_model, split["feature_columns"])
    results = {
        "task": "regression_cc50",
        "target": "CC50, mM",
        "log_transform": True,
        "comparison": tuning_sorted[["model", "cv_rmse", "best_params"]].to_dict("records"),
        "best_model": best_row["model"],
        "best_params": best_row["best_params"],
        "test_metrics": test_metrics,
        "top_features": importance.to_dict("records"),
    }
    path = save_results("regression_cc50", results)
    print(f"Результаты сохранены: {path}")


if __name__ == "__main__":
    main()
