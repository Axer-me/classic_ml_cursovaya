# Классификация, SI > 8

from src.data_utils import load_raw_data, make_classification_split
from src.model_utils import (
    classification_metrics,
    get_feature_importance,
    save_results,
    tune_classification_models,
)


def main() -> None:
    df = load_raw_data()
    split = make_classification_split(df, "si", threshold=8.0)
    positive_share = split["y_train"].mean()
    print(f"Порог SI: 8.0, доля класса 1 в train: {positive_share:.1%}")

    tuning = tune_classification_models(split["x_train"], split["y_train"])
    tuning_sorted = tuning.sort_values("cv_f1", ascending=False)
    print(tuning_sorted[["model", "cv_f1"]].to_string(index=False))

    best_row = tuning_sorted.iloc[0]
    best_model = best_row["best_estimator"]
    y_pred = best_model.predict(split["x_test"])
    y_proba = best_model.predict_proba(split["x_test"])[:, 1]
    test_metrics = classification_metrics(split["y_test"], y_pred, y_proba)
    print(f"\nЛучшая модель: {best_row['model']}")
    print(f"Тестовые метрики: {test_metrics}")

    importance = get_feature_importance(best_model, split["feature_columns"])
    results = {
        "task": "classification_si_gt8",
        "target": "SI",
        "threshold": 8.0,
        "positive_class_share": float(positive_share),
        "comparison": tuning_sorted[["model", "cv_f1", "best_params"]].to_dict("records"),
        "best_model": best_row["model"],
        "best_params": best_row["best_params"],
        "test_metrics": test_metrics,
        "top_features": importance.to_dict("records"),
    }
    path = save_results("classification_si_gt8", results)
    print(f"Результаты сохранены: {path}")


if __name__ == "__main__":
    main()
