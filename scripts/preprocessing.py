import argparse
from dataclasses import dataclass, field
import gc
from datetime import datetime
import hashlib
import logging
import os
import pickle
from typing import Callable, List, Optional, Tuple

from gensim.sklearn_api import D2VTransformer
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
import MeCab

from dataset import load_train_test, CATEGORICAL_COLUMNS, NOVEL_ID_COL, TIMESTAMP_COL, USER_COL, Y
from feature_extraction import (
    PostHistoryFeatureExtractor,
    TextFeatureExtractor,
    TimeSeriesTargetAggregator,
    TimeSeriesFrequencyEncoder,
    TimeSeriesTargetEncoder
)
from text import normalize, tokenize_v1
from utils import ExtendedEnum, get_logger, timer, log_class_volume

FILL_VALUE_KEYWORD = '未設定'
WORK_DIR = os.environ['ROOT']  # ここで作業する
assert(os.path.isdir(WORK_DIR))
DIR_DATA, DIR_FEATURES, DIR_LOG = 'data', 'features', 'log'
assert(os.path.isdir(DIR_DATA))
assert(os.path.isdir(DIR_FEATURES))
assert(os.path.isdir(DIR_LOG))

N_SPLITS = 5

SPLIT_DATE = '2021-07-12'  # Time series split における訓練／検証データの区切り


class Doc2VecVectorizer(BaseEstimator, TransformerMixin):
    @staticmethod
    def to_d2v_format(X) -> List[List[str]]:
        o = []
        for v in X:
            o.append(v.split(' '))
        return o

    def __init__(self, n_components: int, seed: int, workers: int, hashfxn: Callable):
        self.n_components = n_components
        self.seed = seed
        self.workers = workers
        self.hashfxn = hashfxn

    def fit(self, X, y=None) -> object:
        d2v = D2VTransformer(self.n_components,
                             seed=self.seed,
                             workers=self.seed,
                             hashfxn=self.hashfxn)
        d2v.fit(X)
        self.vectorizer_ = d2v
        return self

    def transform(self, X) -> np.ndarray:
        X = Doc2VecVectorizer.to_d2v_format(X)
        return self.vectorizer_.transform(X)


class Vectorizer(ExtendedEnum):
    DOC2VEC = 'doc2vec'
    COUNT = 'count'
    TFIDF = 'tfidf'


class Decomposer(ExtendedEnum):
    LDA = 'lda'
    SVD = 'svd'


def hashfxn(x):
    # Reproducibility for Doc2Vec
    return int(hashlib.md5(str(x).encode()).hexdigest(), 16)


def get_text_vectorizer(
        vectorizer: str, decomposer: str, n_components: int, random_state: int,
        n_jobs: Optional[int] = None) -> Pipeline:
    """テキスト系のカラムをベクトル化するパイプラインを得る。

    Parameters
    ----------
    vectorizer : str
        'doc2vec', 'count', 'tfidf' のどれかを指定すること。
        順番に Doc2Vec, Count vector, TF-IDF vector に対応する。
    decomposer : str
        'lda', 'svd' のどちらかを指定すること。
        順番に Latent Dirichlet Allocation, Truncated SVD に対応する。
        `vectorizer` が 'doc2vec' の時はどちらを指定しても同じ（無視される）。
    n_components: int
        次元数。
    random_state: int
        シード値。
    n_jobs: int
        並列数。なお `vectorizer` が 'doc2vec' だと指定しても無視するので注意。

    Returns
    -------
    Pipeline
    """
    Vectorizer.check_is_valid_value(vectorizer, 'vectozier')
    Decomposer.check_is_valid_value(decomposer, 'decomposer')

    if vectorizer == Vectorizer['DOC2VEC'].value:
        # 再現性の問題があるので `workders` は1, `seed`, `hashfxn` も指定する
        return Pipeline(
            steps=[
                ('vectorizer', Doc2VecVectorizer(n_components,
                 seed=random_state, workers=1, hashfxn=hashfxn))
            ]
        )
    else:
        token_pattern = '(?u)\\b\\w+\\b'  # 1文字でもトークンから削除しないように
        if vectorizer == Vectorizer['COUNT'].value:
            v = CountVectorizer(token_pattern=token_pattern)
        else:
            v = TfidfVectorizer(token_pattern=token_pattern)
        if decomposer == Decomposer['SVD'].value:
            d = TruncatedSVD(n_components=100, random_state=random_state)
        else:
            d = LDA(n_components=n_components, random_state=random_state, n_jobs=n_jobs)
        return Pipeline(
            steps=[
                ('vectorizer', v),
                ('decomposer', d)
            ]
        )


def preprocess_all(
        train: pd.DataFrame, test: pd.DataFrame, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """訓練データとテストデータで同時に掛けられる前処理（クレンジング、特徴量生成）。

    1. 投稿履歴より特徴量を抽出する
    2. テキスト項目 (`title`, `story`) より特徴量を抽出する
    3. `keyword` の欠損値を固定値で埋めた上で正規化する
    4. テキスト項目を分かち書きにする

    特徴量抽出器と抽出した特徴量はファイルに出力する。クレンジングしたカラムから成るデータフレームを返す。

    Parameters
    ----------
    train : pd.DataFrame
        訓練データ。
    test : pd.DataFrame
        テストデータ。
    logger : logging.Logger
        ロガー。

    Returns
    -------
    out_train, out_test: Tuple[pd.DataFrame, pd.DataFrame]:
        訓練データとテストデータ。前処理された `keyword`, `title`, `story` が格納されている。
    """
    with timer('Concatenate training/test set', logger):
        train_test = pd.concat([train, test], axis=0).sort_index()  # 両方一気にやる

    # 1-2.
    feature_extractors = (
        PostHistoryFeatureExtractor(CATEGORICAL_COLUMNS, USER_COL, TIMESTAMP_COL),  # 2.
        TextFeatureExtractor(['title', 'story'])  # 3.
    )
    X_train_test = []
    for fe in feature_extractors:
        with timer(fe.__class__.__name__, logger):
            fe.fit(train_test)
            X_train_test.append(fe.transform(train_test))
            filepath = os.path.join(DIR_FEATURES, f'{fe.__class__.__name__}.pkl')
            with open(filepath, 'wb') as f:
                pickle.dump(fe, f)
    with timer('Save features', logger):
        X_train_test = pd.concat(X_train_test, axis=1)
        X_train_test.to_csv(os.path.join(DIR_FEATURES, 'features_train_test.csv'))

    # クレンジングしたカラムを格納する
    out_train = pd.DataFrame(index=train.index)
    out_test = pd.DataFrame(index=test.index)

    # 3.
    with timer('Normalize `keyword`', logger):
        train_test['keyword'].fillna(FILL_VALUE_KEYWORD, inplace=True)
        train_test['keyword'] = train_test['keyword'].apply(
            lambda x: ' '.join([normalize(k) for k in x.split()])
        )
        out_train['keyword'] = train_test.loc[train.index, 'keyword']
        out_test['keyword'] = train_test.loc[test.index, 'keyword']

    # 4.
    with timer('Transform `title` and `story` into bag of words', logger):
        neologd_userdic_path = '/'.join(['dictionary', 'NEologd.20200910-u.dic'])
        if not os.path.isfile(neologd_userdic_path):
            raise FileNotFoundError(
                f'"{neologd_userdic_path}" が存在しません。neologd を MeCab のユーザ辞書にして保存して下さい。'
            )
        # ここでエラーが出たらおかしなファイルを指定している
        tagger = MeCab.Tagger(f'-u {neologd_userdic_path}')
        for c in ('title', 'story'):
            train_test[c] = train_test[c].apply(lambda x: tokenize_v1(tagger, x))
            out_train[c] = train_test.loc[train.index, c]
            out_test[c] = train_test.loc[test.index, c]
    return out_train, out_test


def preprocess_1fold(train: pd.DataFrame,
                     test: pd.DataFrame,
                     train_idx: np.ndarray,
                     valid_idx: np.ndarray,
                     *,
                     filename_suffix: str,
                     logger: logging.Logger,
                     random_state: int,
                     n_jobs: int,
                     categorical_columns: List[str] = CATEGORICAL_COLUMNS,
                     timestamp_col: str = TIMESTAMP_COL,
                     target: str = Y,
                     user_col: str = USER_COL,
                     use_user_col: bool = False) -> None:
    """訓練／検証／テストデータより特徴量を生成する。

    Parameters
    ----------
    train : pd.DataFrame
        訓練データ。
    test : pd.DataFrame
        テストデータ。
    train_idx : np.ndarray
        訓練データのインデックス。
    valid_idx : np.ndarray
        検証データのインデックス。
    filename_suffix: str
        出力ファイル名の接尾語。
    logger: logging.Logger
        ロガー。
    random_state: int
        シード。
    n_jobs: int
        並列数。
    categorical_columns : List[str]
        カテゴリとみなすカラム・
    timestamp_col : str, optional
        時間軸を表すカラム。
    target : str, optional
        ターゲットのカラム。
    user_col : str, optional
        ユーザのカラム。
    use_user_col: str, optional
        `user_col` を特徴量として使うか否か。
    """
    train, valid = train.loc[train_idx], train.loc[valid_idx]
    X_train, X_valid, X_test = [], [], []

    # 元のカラムをそのまま特徴として用いる
    passthrough_features = categorical_columns.copy()
    if use_user_col:
        passthrough_features.append(user_col)
    X_train.append(train[passthrough_features])
    X_test.append(test[passthrough_features])
    if valid.shape[0] > 0:
        X_valid.append(valid[passthrough_features])
    # テキスト項目のベクトル化以外の特徴量抽出
    feature_extractors = (
        TimeSeriesTargetAggregator(user_col, timestamp_col, target),
        TimeSeriesFrequencyEncoder(categorical_columns, timestamp_col),
        TimeSeriesTargetEncoder(categorical_columns, target, timestamp_col)
    )
    for fe in feature_extractors:
        with timer(fe.__class__.__name__, logger):
            fe.fit(train)
            X_train.append(fe.transform(train))
            X_test.append(fe.transform(test))
            if valid.shape[0] > 0:
                X_valid.append(fe.transform(valid))
            filepath = os.path.join(DIR_FEATURES, f'{fe.__class__.__name__}{filename_suffix}.pkl')
            with open(filepath, 'wb') as f:
                pickle.dump(fe, f)

    # テキスト項目のベクトル化
    text_vectorizer = ColumnTransformer(
        transformers=[('keyword',
                       get_text_vectorizer(
                           'count', 'svd', n_components=100,
                           random_state=random_state),
                       'keyword'),
                      ('story',
                       get_text_vectorizer(
                           'count', 'svd', n_components=100,
                           random_state=random_state),
                       'story'),
                      ('title1',
                       get_text_vectorizer(
                           'count', 'svd', n_components=100,
                           random_state=random_state),
                       'title'),
                      ('title2',
                       get_text_vectorizer(
                           'count', 'lda', n_components=100,
                           random_state=random_state,
                           n_jobs=n_jobs),
                       'title'),
                      ('title3',
                       get_text_vectorizer(
                           'tfidf', 'lda', n_components=100,
                           random_state=random_state,
                           n_jobs=n_jobs),
                       'title'), ])
    with timer('Vectorize text columns', logger):
        text_vectorizer.fit(train)
        vector_train = text_vectorizer.transform(train)
        columns = [f'vec{i + 1}' for i in range(vector_train.shape[1])]
        vector_train = pd.DataFrame(data=vector_train, index=train.index, columns=columns)
        X_train.append(vector_train)
        vector_test = text_vectorizer.transform(test)
        vector_test = pd.DataFrame(data=vector_test, index=test.index, columns=columns)
        X_test.append(vector_test)
        if valid.shape[0] > 0:
            vector_valid = text_vectorizer.transform(valid)
            vector_valid = pd.DataFrame(data=vector_valid, index=valid.index, columns=columns)
            X_valid.append(vector_valid)
        filepath = os.path.join(DIR_FEATURES, f'{fe.__class__.__name__}{filename_suffix}.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(fe, f)

    # 得られた特徴量を保存する
    with timer('Save features', logger):
        X_train = pd.concat(X_train, axis=1)
        X_train.to_csv(os.path.join(DIR_FEATURES, f'features_train{filename_suffix}.csv'))
        X_test = pd.concat(X_test, axis=1)
        X_test.to_csv(os.path.join(DIR_FEATURES, f'features_test{filename_suffix}.csv'))
        if valid.shape[0] > 0:
            X_valid = pd.concat(X_valid, axis=1)
            X_valid.to_csv(os.path.join(DIR_FEATURES, f'features_valid{filename_suffix}.csv'))
    return None


def run(split_date: str, random_state: int, n_jobs: int, use_user_col: bool) -> None:
    """コンペティションデータに対して前処理を行い結果をファイルに出力する。
    """
    os.chdir(WORK_DIR)
    logger = get_logger(__name__, os.path.join(
        DIR_LOG, datetime.now().strftime('preprocessing_%Y%m%d_%H%M%S.log')))
    logger.info('Start')

    logger.info('`split_date` is {}'.format(split_date))
    logger.info('`random_state` is {}'.format(random_state))
    logger.info('`n_jobs` is {}'.format(n_jobs))
    logger.info('`use_user_col` is {}'.format(use_user_col))

    '''データの読み込み'''
    train, test = load_train_test(
        os.path.join(DIR_DATA, 'train.csv'),
        os.path.join(DIR_DATA, 'test.csv'),
    )

    log_class_volume(train, logger)

    '''特徴量の生成とデータクレンジング'''
    out_train, out_test = preprocess_all(train, test, logger)
    train.drop(columns=out_train.columns, inplace=True)
    train = pd.concat([train, out_train], axis=1)
    test.drop(columns=out_test.columns, inplace=True)
    test = pd.concat([test, out_test], axis=1)
    del out_train, out_test
    gc.collect()

    overwrapped_user = train[train[USER_COL].isin(test[USER_COL])][USER_COL].unique().tolist()

    '''全てのデータを使い特徴量を抽出する'''
    mask = (train[TIMESTAMP_COL] <= split_date).values  # True: 訓練データ, False: 検証データ
    train_idx = train.loc[mask].index
    valid_idx = train.loc[~mask].index
    logger.info('Time series split) Number of training data: {}'.format(len(train_idx)))
    log_class_volume(train.loc[train_idx], logger)
    logger.info('Time series split) Number of validation data: {}'.format(len(valid_idx)))
    log_class_volume(train.loc[valid_idx], logger)
    if use_user_col:
        not_overwrapped_user_id = 99999999
        assert(not_overwrapped_user_id not in train[USER_COL].tolist())
        assert(not_overwrapped_user_id not in test[USER_COL].tolist())
        mask = (train[USER_COL].isin(overwrapped_user)).values
        train2 = train.copy()
        train2.loc[~mask, USER_COL] = not_overwrapped_user_id
        mask = (test[USER_COL].isin(overwrapped_user)).values
        test2 = test.copy()
        test2.loc[~mask, USER_COL] = not_overwrapped_user_id
        preprocess_1fold(train2, test2, train_idx, valid_idx, filename_suffix='_ts',
                         use_user_col=use_user_col,
                         logger=logger, random_state=random_state, n_jobs=n_jobs)
    else:
        preprocess_1fold(train, test, train_idx, valid_idx, filename_suffix='_ts',
                         use_user_col=use_user_col,
                         logger=logger, random_state=random_state, n_jobs=n_jobs)

    '''"train.csv", "test.csv" の両方に存在するユーザのデータだけを使い特徴量を抽出する'''
    train = train[train[USER_COL].isin(overwrapped_user)]
    logger.info(
        'Make new dataset from "train.csv"(Extract users appeard both in "train.csv" and "test.csv").')
    logger.info('Number of users remained: {}'.format(len(overwrapped_user)))
    logger.info('Number of dataset remained: {}'.format(train.shape[0]))
    splitter = GroupKFold(N_SPLITS)
    for i, (train_idx, valid_idx) in enumerate(splitter.split(train, train[Y], train[USER_COL])):
        train_idx, valid_idx = train.iloc[train_idx].index, train.iloc[valid_idx].index
        logger.info('Group k-fold) Number of training data: {}'.format(len(train_idx)))
        log_class_volume(train.loc[train_idx], logger)
        logger.info('Group k-fold) Number of validation data: {}'.format(len(valid_idx)))
        log_class_volume(train.loc[valid_idx], logger)
        preprocess_1fold(train, test, train_idx, valid_idx,
                         random_state=random_state, n_jobs=n_jobs,
                         filename_suffix=f'_grp{i + 1}', logger=logger, use_user_col=True)
    logger.info('Complete')


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='''Apply preprocessing to "train.csv" and "test.csv".''')
    parser.add_argument('-s', '--seed',  type=int, default=1, help='Random seed. Default = 1.')
    parser.add_argument(
        '-n', '--n_jobs', type=int, default=-1,
        help='Number of cpu cores used. Default = -1(use all).')
    parser.add_argument(
        '-d', '--date', default='2021-07-12', type=str,
        help='Threshold used for timeseries cv, expected to be "YYYY-MM-DD" format.Default = "2021-07-12".')
    parser.add_argument('-u', '--use_user_col', action='store_true',
                        help=f'Use `{USER_COL}` as feature when time series cv.')
    args = parser.parse_args()
    assert(args.seed >= 0)
    assert(args.n_jobs == -1 or args.n_jobs <= os.cpu_count())
    return args


if __name__ == '__main__':
    args = get_args()
    run(args.date, args.seed, args.n_jobs, args.use_user_col)
