from contextlib import contextmanager
from enum import Enum
import logging
import time
from typing import Any

import pandas as pd

from dataset import Y


@contextmanager
def timer(name, logger=None, level=logging.DEBUG) -> None:
    # https://amalog.hateblo.jp/entry/kaggle-snippets
    print_ = print if logger is None else lambda msg: logger.log(level, msg)
    t0 = time.time()
    print_(f'[{name}] start')
    yield
    print_(f'[{name}] done in {time.time() - t0:.3f} s')


def get_logger(name: str, filepath: str) -> logging.Logger:
    # https://docs.python.org/ja/3/howto/logging-cookbook.html

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def log_class_volume(df: pd.DataFrame, logger: logging.Logger) -> None:
    """クラスごとのレコード数をログに残す。

    Parameters
    ----------
    df : pd.DataFrame
        データフレーム。
    logger : logging.Logger
        ロガー。
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df, columns=[Y])
    num = df[Y].value_counts().sort_index()
    ratio = df[Y].value_counts(normalize=True).sort_index()
    classes = num.index
    for (c, n, r) in zip(classes, num.values, ratio.values):
        logger.info('Class {}: number of data is {}, ratio is {:.5f}'.format(c, n, r))


class ExtendedEnum(Enum):

    @classmethod
    def values(cls) -> list:
        return [cls[k].value for k in cls.__members__.keys()]

    @classmethod
    def is_valid_value(cls, v: Any) -> bool:
        return v in cls.values()

    @classmethod
    def check_is_valid_value(cls, v: Any, p: str) -> None:
        if not cls.is_valid_value(v):
            raise ValueError(f'`{p}` should be one of {cls.values()} but {v} was given.')
