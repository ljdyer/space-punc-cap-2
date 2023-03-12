import more_itertools

import pandas as pd
from sklearn.model_selection import train_test_split

from character_only_llm.consts import TARGET_TEXT_COL, SOURCE_TEXT_COL


# ====================
def remove_formatting(string):
    string = string.lower().replace(' ', '').replace('.', '').replace(',', '')
    return string


# ====================
def chunked_text(text, n):
    return [''.join(chunk)
            for chunk
            in more_itertools.chunked(list(text), n)]


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
    all_cleaned = df['all_cleaned'].to_list()
    text = ' '.join(all_cleaned)
    target_text = chunked_text(text, 100)
    source_text = [remove_formatting(s) for s in target_text]
    df = pd.DataFrame({
        SOURCE_TEXT_COL: pd.Series(source_text),
        TARGET_TEXT_COL: pd.Series(target_text)
    })
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
