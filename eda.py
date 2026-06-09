# ЕДА
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.data_utils import (
    DATA_PATH,
    TARGET_COLUMNS,
    get_feature_columns,
    load_raw_data,
    remove_low_variance_features,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("deep")


# Сохраняет текущий график в каталог outputs/eda
def save_figure(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close()


# Сводная табличка
def basic_overview(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = get_feature_columns(df)
    overview = pd.DataFrame(
        {
            "показатель": [
                "Число наблюдений",
                "Число признаков",
                "Пропуски (всего)",
                "Константные признаки",
                "Признаков после отбора",
            ],
            "значение": [
                len(df),
                len(feature_cols),
                int(df.isnull().sum().sum()),
                len(feature_cols) - len(remove_low_variance_features(df, feature_cols)),
                len(remove_low_variance_features(df, feature_cols)),
            ],
        }
    )
    overview.to_csv(OUTPUT_DIR / "dataset_overview.csv", index=False, encoding="utf-8-sig")
    return overview


# Анализируем IC50, CC50 и SI
def analyze_targets(df: pd.DataFrame) -> pd.DataFrame:
    target_stats = df[list(TARGET_COLUMNS.values())].describe().T
    target_stats["skew"] = df[list(TARGET_COLUMNS.values())].skew()
    target_stats["kurtosis"] = df[list(TARGET_COLUMNS.values())].kurtosis()
    target_stats.to_csv(OUTPUT_DIR / "target_statistics.csv", encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, TARGET_COLUMNS.values()):
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Распределение {col}")
        ax.set_xlabel(col)
    save_figure("target_distributions.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, TARGET_COLUMNS.values()):
        stats.probplot(np.log1p(df[col]), dist="norm", plot=ax)
        ax.set_title(f"Q-Q plot log1p({col})")
    save_figure("target_qq_plots.png")

    pairplot_df = df[list(TARGET_COLUMNS.values())]
    sns.pairplot(pairplot_df, diag_kind="kde", corner=True)
    plt.savefig(OUTPUT_DIR / "target_pairplot.png", dpi=150, bbox_inches="tight")
    plt.close()

    return target_stats


# Проверка связи SI = CC50 / IC50
def verify_si_definition(df: pd.DataFrame) -> pd.DataFrame:
    calculated = df["CC50, mM"] / df["IC50, mM"]
    check = pd.DataFrame(
        {
            "SI_исходный": df["SI"],
            "SI_расчётный": calculated,
            "разница": (df["SI"] - calculated).abs(),
        }
    )
    check.to_csv(OUTPUT_DIR / "si_verification.csv", index=False, encoding="utf-8-sig")
    return check


# Визуал пропусков и константных признаков
def analyze_missing_and_constants(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing.to_csv(OUTPUT_DIR / "missing_values.csv", encoding="utf-8-sig")

    feature_cols = get_feature_columns(df)
    constant_cols = [
        col for col in feature_cols if df[col].nunique(dropna=False) <= 1
    ]
    pd.Series(constant_cols, name="constant_feature").to_csv(
        OUTPUT_DIR / "constant_features.csv", index=False, encoding="utf-8-sig"
    )


# анализ корреляций
def correlation_analysis(df: pd.DataFrame) -> None:
    key_features = [
        "MolWt",
        "MolLogP",
        "TPSA",
        "NumHDonors",
        "NumHAcceptors",
        "NumRotatableBonds",
        "qed",
        "RingCount",
        "NumAromaticRings",
        "fr_benzene",
    ]
    cols = list(TARGET_COLUMNS.values()) + key_features
    corr = df[cols].corr()
    corr.to_csv(OUTPUT_DIR / "target_descriptor_correlation.csv", encoding="utf-8-sig")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Корреляция целевых показателей и ключевых дескрипторов")
    save_figure("correlation_heatmap.png")


# Оценка выбросов
def outlier_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in TARGET_COLUMNS.values():
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        rows.append(
            {
                "target": col,
                "q1": q1,
                "q3": q3,
                "lower_bound": lower,
                "upper_bound": upper,
                "outliers_count": int(mask.sum()),
                "outliers_share": float(mask.mean()),
            }
        )
    outlier_df = pd.DataFrame(rows)
    outlier_df.to_csv(OUTPUT_DIR / "target_outliers_iqr.csv", index=False, encoding="utf-8-sig")
    return outlier_df


# Сщитаем пороги для классификации
def classification_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    thresholds = []
    for key, col in TARGET_COLUMNS.items():
        median = df[col].median()
        thresholds.append(
            {
                "задача": f"{key}_above_median",
                "порог": median,
                "доля_класса_1": float((df[col] > median).mean()),
            }
        )
    thresholds.append(
        {
            "задача": "si_above_8",
            "порог": 8.0,
            "доля_класса_1": float((df["SI"] > 8).mean()),
        }
    )
    threshold_df = pd.DataFrame(thresholds)
    threshold_df.to_csv(
        OUTPUT_DIR / "classification_thresholds.csv", index=False, encoding="utf-8-sig"
    )
    return threshold_df


# Запускаем ЕДА
def main() -> None:
    df = load_raw_data(DATA_PATH)
    print(f"Размер датасета: {df.shape}")

    overview = basic_overview(df)
    print(overview.to_string(index=False))

    target_stats = analyze_targets(df)
    print("\nСтатистика целевых переменных:")
    print(target_stats[["mean", "std", "skew"]].to_string())

    verify_si_definition(df)
    analyze_missing_and_constants(df)
    correlation_analysis(df)
    outlier_analysis(df)
    classification_thresholds(df)

    print(f"\nРезультаты EDA сохранены в {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
