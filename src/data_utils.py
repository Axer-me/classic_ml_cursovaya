# Загрузка и предобработка данных курсовой работы.

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "Данные_для_курсовои_Классическое_МО.xlsx"

TARGET_COLUMNS = {
    "ic50": "IC50, mM",
    "cc50": "CC50, mM",
    "si": "SI",
}

RANDOM_STATE = 42
TEST_SIZE = 0.2


# Загружаем исходный датасет из Excel
def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_PATH
    return pd.read_excel(path)


# Возвращаем список молекулярных признаков без целевых переменных
def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = set(TARGET_COLUMNS.values()) | {"Unnamed: 0"}
    return [col for col in df.columns if col not in exclude]


# Удаляет константные и почти константные фичи
def remove_low_variance_features(
    df: pd.DataFrame, feature_columns: list[str], threshold: float = 0.0
) -> list[str]:
    variances = df[feature_columns].var()
    return variances[variances > threshold].index.tolist()


# Подготавливает матрицу признаков
def prepare_features(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    if feature_columns is None:
        feature_columns = get_feature_columns(df)
    feature_columns = remove_low_variance_features(df, feature_columns)
    imputer = SimpleImputer(strategy="median")
    features = pd.DataFrame(
        imputer.fit_transform(df[feature_columns]),
        columns=feature_columns,
        index=df.index,
    )
    return features, feature_columns


# Формирует train/test для задачи регрессии
def make_regression_split(
    df: pd.DataFrame,
    target_key: str,
    log_transform: bool = True,
):
    target_col = TARGET_COLUMNS[target_key]
    features, feature_columns = prepare_features(df)
    y = df[target_col].astype(float)
    if log_transform:
        y = np.log1p(y)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(
        scaler.fit_transform(x_train),
        columns=feature_columns,
        index=x_train.index,
    )
    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test),
        columns=feature_columns,
        index=x_test.index,
    )
    return {
        "x_train": x_train,
        "x_test": x_test,
        "x_train_scaled": x_train_scaled,
        "x_test_scaled": x_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "feature_columns": feature_columns,
        "scaler": scaler,
        "log_transform": log_transform,
        "target_col": target_col,
    }


# Формирует train/test для задачи классификации
def make_classification_split(df: pd.DataFrame, target_key: str, threshold: float | str):
    target_col = TARGET_COLUMNS[target_key]
    features, feature_columns = prepare_features(df)
    if threshold == "median":
        cutoff = df[target_col].median()
    else:
        cutoff = float(threshold)
    y = (df[target_col] > cutoff).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(
        scaler.fit_transform(x_train),
        columns=feature_columns,
        index=x_train.index,
    )
    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test),
        columns=feature_columns,
        index=x_test.index,
    )
    return {
        "x_train": x_train,
        "x_test": x_test,
        "x_train_scaled": x_train_scaled,
        "x_test_scaled": x_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "feature_columns": feature_columns,
        "scaler": scaler,
        "cutoff": cutoff,
        "target_col": target_col,
    }
