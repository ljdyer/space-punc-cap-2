from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from character_only_llm.consts import TARGET_TEXT_COL, SOURCE_TEXT_COL


def fetch_restoration_data() -> pd.DataFrame:
    """
    This function will fetch the summarisation data.
    :return:
    """
    train_path = "/data/ahughes/ted_train.csv"
    return pd.read_csv(train_path)


def parse_restoration_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function will fetch the summarisation data.
    :return:
    """
    df = df.rename(columns={"no_punctuation": TARGET_TEXT_COL, "all_cleaned": SOURCE_TEXT_COL})
    df = df[[SOURCE_TEXT_COL, TARGET_TEXT_COL]]
    df[SOURCE_TEXT_COL] = "restore: " + df[SOURCE_TEXT_COL]
    return df


def generate_model_splits(df: pd.DataFrame) -> tuple:
    """
    This function will fetch the summarisation data.
    :return:
    """
    train, test = train_test_split(df, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    return train, val, test
