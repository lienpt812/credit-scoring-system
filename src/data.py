from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import BAD_LABEL, FEATURE_COLUMNS, TARGET_COLUMN, TEST_PATH, TRAIN_PATH


def load_train_test(
    train_path: str | Path = TRAIN_PATH,
    test_path: str | Path = TEST_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_excel(train_path)
    test = pd.read_excel(test_path)
    return clean_dataframe(train), clean_dataframe(test)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in cleaned.select_dtypes(include="object").columns:
        cleaned[column] = cleaned[column].astype(str).str.strip()
    if TARGET_COLUMN in cleaned.columns:
        cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].map(normalize_target)
    return cleaned


def normalize_target(value: object) -> int:
    text = str(value).strip().strip("'").strip('"')
    return 1 if text == BAD_LABEL else 0


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[FEATURE_COLUMNS], df[TARGET_COLUMN].astype(int)
