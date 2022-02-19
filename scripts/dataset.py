from typing import Tuple

import pandas as pd

# Column names.
NOVEL_ID_COL = 'ncode'
USER_COL = 'userid'
TIMESTAMP_COL = 'general_firstup'
Y = 'fav_novel_cnt_bin'  # Prediction target

# Data types.
DTYPES = {
    NOVEL_ID_COL: 'object',
    'general_firstup': 'object',
    'title': 'object',
    'story': 'object',
    'keyword': 'object',
    USER_COL: 'int',
    'biggenre': 'int',
    'genre': 'int',
    'novel_type': 'uint',
    'isr15': 'uint',
    'isbl': 'uint',
    'isgl': 'uint',
    'iszankoku': 'uint',
    'istensei': 'uint',
    'istenni': 'uint',
    'pc_or_k': 'uint',
    Y: 'uint8'
}

# Name of discrete columns.
CATEGORICAL_COLUMNS = [
    'biggenre',
    'genre',
    'novel_type',
    'isr15',
    'isbl',
    'isgl',
    'iszankoku',
    'istensei',
    'istenni',
    'pc_or_k'
]

# Names of columns which may have missing values.
NULLABLE_COLUMNS = ('keyword')


def load_df(filepath: str, test: bool) -> pd.DataFrame:
    """Load training data or test data.

    Parameters
    ----------
    filepath: str
        Filepath of training data or test data.
    test: bool
        Set `True` when read test data otherwise `False`.
    Parameters

    Returns
    -------
    data: pd.DataFrame
    """
    # Load a data with explict data type.
    dtypes = DTYPES.copy()
    if test:
        # There are no label!
        _ = dtypes.pop(Y)
    df = pd.read_csv(filepath, usecols=list(dtypes.keys()),
                     dtype=dtypes, parse_dates=[TIMESTAMP_COL])

    def validate(df: pd.DataFrame) -> None:
        # Ensure there are no duplication.
        assert(df[NOVEL_ID_COL].duplicated().sum() == 0)
        # Ensure there are no unexpected missing value.
        for c in df.columns:
            if c not in NULLABLE_COLUMNS:
                assert(df[c].isnull().sum() == 0)

    validate(df)
    return df.set_index(NOVEL_ID_COL).sort_values([USER_COL, TIMESTAMP_COL])


def load_train_test(filepath_train: str, filepath_test: str) -> Tuple[pd.DataFrame]:
    """Load training data and test data.

    Parameters
    ----------
    filepath_train, filepath_test : str
        Filepath to 'train.csv' and 'test.csv'.

    Returns
    -------
    Tuple[pd.DataFrame]
        Length 2, 1st dataframe is training data and the other is test data.
    """
    # Load dataset.
    train = load_df(filepath_train, test=False)
    test = load_df(filepath_test, test=True)
    # Ensure column layout is almost same; only 1 difference is only training data has label.
    assert(set(train.columns.tolist()) - set(test.columns.tolist()) == {Y})
    assert(set(test.columns.tolist()) - set(train.columns.tolist()) == set())
    # Ensure there are no novels that appear in both training and test data.
    assert(set(train.index.tolist()) & set(test.index.tolist()) == set())
    return train, test
