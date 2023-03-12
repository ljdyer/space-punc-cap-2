import pandas as pd
from sklearn.model_selection import train_test_split

from character_only_llm.consts import TARGET_TEXT_COL, SOURCE_TEXT_COL


def fetch_summarisation_data() -> pd.DataFrame:
    """
    This function will fetch the summarisation data.
    :return:
    """
    path = "https://raw.githubusercontent.com/Shivanandroy/T5-Finetuning-PyTorch/main/data/news_summary.csv"
    return pd.read_csv(path)


def parse_summarisation_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function will fetch the summarisation data.
    :return:
    """
    # simpleT5 expects dataframe to have 2 columns: "source_text" and "target_text"
    df = df.rename(columns={"headlines": TARGET_TEXT_COL, "text": SOURCE_TEXT_COL})
    df = df[['source_text', TARGET_TEXT_COL]]

    # T5 model expects a task related prefix:
    df[SOURCE_TEXT_COL] = "summarize: " + df[SOURCE_TEXT_COL]
    return df


def generate_model_splits(df: pd.DataFrame) -> tuple:
    """
    This function will fetch the summarisation data.
    :return:
    """
    train, test = train_test_split(df, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    return train, val, test
