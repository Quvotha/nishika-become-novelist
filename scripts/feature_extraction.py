import datetime
from functools import lru_cache
from typing import Any, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PostHistoryFeatureExtractor(BaseEstimator, TransformerMixin):  # OK!
    """Extract features from history of post by user."""

    PREFIX = 'ph_'

    def __init__(self, categorical_columns: List[str], user_col: str, timestamp_col: str):
        self.categorical_columns = categorical_columns
        self.user_col = user_col
        self.timestamp_col = timestamp_col

    def fit(self, X: pd.DataFrame, y=None) -> object:
        X = X[[self.user_col, self.timestamp_col] + self.categorical_columns]
        assert(X.isnull().sum().sum() == 0)
        # 初投稿日付を記録する
        self.first_post_on_ = {
            user_id: df[self.timestamp_col].min() for (user_id, df) in X.groupby(self.user_col)
        }
        # 全てのカテゴリを記録する
        self.categories_ = {c: X[c].unique().tolist() for c in self.categorical_columns}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract features from given data.

        For each user, extract following features;

        1. year, month, day of first post,
        2. number of days elapsed since last post, 0 for first post
        3. number of days elapsed since first post, 0 for first post
        4. how many novels that user have posted,
        5. frequency of post,
        6. count and share of each category.

        Parameters
        ----------
        X : pd.DataFrame
            [description]

        Returns
        -------
        pd.DataFrame
            [description]
        """

        out_df = pd.concat([
            self.transform_1user(user_id, df) for user_id, df in X.groupby(self.user_col)
        ]).loc[X.index]
        count_columns = [c for c in out_df.columns if '_count_' in c]
        for c in count_columns:
            out_df[c] = out_df[c].fillna(0).astype('int')
        return out_df

    def transform_1user(self, userid: int, df: pd.DataFrame) -> pd.DataFrame:
        assert(df[self.user_col].nunique() == 1)
        df.sort_values(self.timestamp_col, inplace=True)

        # 1. 初投稿日（年、月、日）
        if userid in self.first_post_on_.keys():
            first_post_on1 = self.first_post_on_[userid]
            first_post_on2 = df[self.timestamp_col].min()
            first_post_on = np.min([first_post_on1, first_post_on2])
        else:
            first_post_on = df[self.timestamp_col].min()
        df['first_post_on'] = first_post_on
        df[f'{self.PREFIX}first_post_year'] = df['first_post_on'].dt.year.astype('int')
        df[f'{self.PREFIX}first_post_month'] = df['first_post_on'].dt.month.astype('int')
        df[f'{self.PREFIX}first_post_day'] = df['first_post_on'].dt.day.astype('int')

        # 2. 前回投稿からの日数、初投稿の場合は 0
        df['last_post_on'] = df[self.timestamp_col].shift(1)
        df['days_since_last_post'] = (df[self.timestamp_col] - df['last_post_on']).dt.days
        df['seconds_since_last_post'] = (df[self.timestamp_col] - df['last_post_on']).dt.seconds
        df[f'{self.PREFIX}days_since_last_post'] = df['days_since_last_post'] + \
            (df['seconds_since_last_post'] / 86400)
        df[f'{self.PREFIX}days_since_last_post'] = df[f'{self.PREFIX}days_since_last_post'] \
            .fillna(0.).astype('float32')

        # 3. 初投稿日からの経過日数
        df[f'{self.PREFIX}days_since_first_post'] = df[f'{self.PREFIX}days_since_last_post'].cumsum()

        # 4. 作品数
        df[f'{self.PREFIX}num_posts'] = [i + 1 for i in range(df.shape[0])]
        df[f'{self.PREFIX}num_posts'] = df[f'{self.PREFIX}num_posts'].astype('int')

        # 5. 投稿頻度
        df[f'{self.PREFIX}post_freq'] = (df[f'{self.PREFIX}num_posts'] /
                                         df[f'{self.PREFIX}days_since_first_post']) \
            .replace([np.inf, -np.inf], 1.).astype('float32')

        # 6. カテゴリ型の列毎：値のカウント、シェア
        '''大量の列を1個ずつ pd.DataFrame に追加すると PerformanceWarning が出るので一気に追加する'''
        count_share_ = {}
        for c in self.categorical_columns:
            # One hot 形式、ただし fit 時に見たカテゴリだけが残るように調整する
            count_df = pd.get_dummies(df[c]).cumsum()
            for v in self.categories_[c]:
                if v not in count_df.columns:
                    count_df[v] = 0
            count_df = count_df[self.categories_[c]]
            # 通算登場回数とシェアの計算
            for v in count_df.columns:
                count_share_[f'{self.PREFIX}{c}_count_{v}'] = count_df[v].astype('int')
                count_share_[f'{self.PREFIX}{c}_share_{v}'] = (
                    count_df[v] / df[f'{self.PREFIX}num_posts']).astype('float32')
                # df[f'{self.PREFIX}{c}_count_{v}'] = count_df[v]
                # df[f'{self.PREFIX}{c}_share_{v}'] = count_df[v] / df[f'{self.PREFIX}num_posts']
        count_share_ = pd.DataFrame(data=count_share_, index=df.index)
        df = pd.concat([df, count_share_], axis=1)
        use_cols = [c for c in df.columns if c.startswith(self.PREFIX)]
        return df[use_cols]


class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from text column."""

    PREFIX = 'txt_'

    def __init__(self, text_columns: List[str]):
        """Initializer.

        Parameters
        ----------
        text_columns : List[str]
            Names of categorical columns to be encoded.
        """
        self.text_columns = text_columns

    def fit(self, X=None, y=None) -> object:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract feature from given dataframe.

        For each text column, extract followings;

        1. Text length,
        2. Number of lines,
        3. Number of blank lines,
        4. Share of blank lines,
        5. Average length of lines, except for blank lines.

        Parameters
        ----------
        X : pd.DataFrame
            [description]

        Returns
        -------
        pd.DataFrame
            [description]
        """
        out_df = pd.DataFrame(index=X.index)
        X = X[self.text_columns]
        # Process text columns 1 by 1
        for column in self.text_columns:
            # 1. Text length
            out_df[f'{self.PREFIX}len_{column}'] = X[column].str.len().astype('int')
            # 2. Number of lines
            out_df[f'{self.PREFIX}nlines_{column}'] = X[column].apply(
                lambda x: len(x.splitlines())).astype('int')
            # 3. Number of blank lines
            out_df[f'{self.PREFIX}nblines_{column}'] = X[column].apply(
                lambda x: len([l for l in x.splitlines() if not l])
            ).astype('int')
            # 4. Share of blank lines
            out_df[f'{self.PREFIX}sblines_{column}'] = (
                out_df[f'{self.PREFIX}nblines_{column}'] / out_df[f'{self.PREFIX}nlines_{column}']
            ).replace([np.inf, -np.inf], np.nan).astype('float32')
            # 5. Average length of lines, except for blank lines
            out_df[f'{self.PREFIX}avelen_{column}'] = X[column].apply(
                TextFeatureExtractor.calc_average_length).astype('float32')
        return out_df

    @staticmethod
    def calc_average_length(text: str) -> float:
        lines = [l for l in text.splitlines() if l]
        if len(lines) < 1:
            return 0.
        else:
            line_lengths = [len(l) for l in lines]
            return float(sum(line_lengths) / len(lines))


class TimeSeriesTargetAggregator(BaseEstimator, TransformerMixin):  # Todo: Docstring, code
    """ユーザ毎に教師ラベルを時間軸を考慮して集計し特徴量とする。

    集計軸はユーザ×投稿日付（≠タイムスタンプ）とする。
    集計項目は投稿日付より過去の教師ラベルとする。ただし生の教師ラベルを集計対象とするのではなく
    一日に複数の投稿作品がある場合はそれらの教師ラベルの最大値を当該日付における代表値とみなし、代表値を集計対象とする。
    集計方法は最小値、最大値、平均値、中央値の4種類とする。

    Parameters
    ----------
    BaseEstimator : [type]
        [description]
    TransformerMixin : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    PREFIX = 'ts_agg_'
    AGG_NAMES_ = ('min', 'max', 'mean', 'median')

    def __init__(
            self, user_col: str, timestamp_col: str, target_col: str, fill_value: int = -1,
            window_size: int = 10, min_periods: int = 1):
        self.user_col = user_col
        self.timestamp_col = timestamp_col
        self.target_col = target_col
        self.fill_value = fill_value
        self.window_size = window_size
        self.min_periods = min_periods

    def get_feature_names(self) -> List[str]:
        return [f'{self.PREFIX}{_}' for _ in self.AGG_NAMES_]

    def fit(self, X: pd.DataFrame, y=None) -> object:
        X = X[[self.user_col, self.timestamp_col, self.target_col]].copy()
        assert(X.isnull().sum().sum() == 0)
        assert(X[self.target_col].min() > self.fill_value)
        '''
        教師ラベルを投稿日付毎に集計する。一日に複数の作品が投稿されている場合は最大値を代表値として採用する。
        '''
        X['date'] = X[self.timestamp_col].dt.date
        self.agg_df_ = {}  # key: user_id, value: 過去の教師データの集計値, pd.DataFrame
        for (user_id, user_df) in X.groupby(self.user_col):
            # 投稿日付順にソートすること
            date_df = user_df.groupby('date')[self.target_col].max().sort_index()
            rolling = date_df.rolling(window=self.window_size, min_periods=self.min_periods)
            agg_df = pd.DataFrame(index=date_df.index)
            # 当日の教師ラベルは集計対象にしないこと
            for agg_name in self.AGG_NAMES_:
                agg_df[agg_name] = getattr(rolling, agg_name)().shift(1)
                # agg_df['min'] = rolling.min().shift(1)
                # agg_df['max'] = rolling.max().shift(1)
                # agg_df['mean'] = rolling.mean().shift(1)
                # agg_df['median'] = rolling.median().shift(1)
            self.agg_df_[user_id] = agg_df
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[[self.user_col, self.timestamp_col]].copy()
        assert(X.isnull().sum().sum() == 0)
        X['date'] = X[self.timestamp_col].dt.date
        values = []
        for (user_id, date) in zip(X[self.user_col].to_numpy(), X['date'].to_numpy()):
            value = self.query(user_id, date)
            values.append(value)
        feature_names = self.get_feature_names()
        out_df = pd.DataFrame(data=values, index=X.index, columns=feature_names)
        return out_df

    def query(self, user_id: int, date: datetime.date) -> np.ndarray:
        defualt_value = np.array([self.fill_value] * len(self.get_feature_names()),
                                 dtype=np.float32)
        if user_id in self.agg_df_.keys():
            latest_date_df = self.agg_df_[user_id].loc[:date]
            if latest_date_df.shape[0] > 0:
                # 最も直近の日付の集計値を特徴量とする
                # latest_date_df.fillna(self.fill_value, inplace=True)
                return latest_date_df.iloc[-1].fillna(self.fill_value).to_numpy(dtype=np.float32)
            else:  # 過去日付の投稿が無い場合
                return defualt_value
        else:  # `fit` の時に見たことが無いユーザの場合
            return defualt_value


class TimeSeriesFrequencyEncoder(BaseEstimator, TransformerMixin):
    """Frequency encoder for categorical features, considering `general_firstup`."""

    SUFFIX = 'ts_freq_'

    def __init__(self, categorical_columns: List[str], timestamp_col: str):
        """Initializer.

        Parameters
        ----------
        categorica_columns: List[str]
            Names of categorical columns to be encoded.
        timestamp_col: str
            Name of timestamp column.
        """
        self.categorical_columns = categorical_columns
        self.timestamp_col = timestamp_col

    def fit(self, X: pd.DataFrame, y=None) -> object:
        X = X[[self.timestamp_col] + self.categorical_columns].copy()
        X['date'] = X[self.timestamp_col].dt.date
        self.history_ = {}
        for col in self.categorical_columns:
            self.history_[col] = {}
            for c, df in X.groupby(col):
                # 当日の値は含めない
                self.history_[col][c] = df.groupby('date')[col] \
                                          .count() \
                                          .sort_index() \
                                          .cumsum() \
                                          .shift(1) \
                                          .fillna(0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # カテゴリ毎に登場回数、シェアを計算する
        out_df = pd.DataFrame(index=X.index)
        X = X[[self.timestamp_col] + self.categorical_columns].copy()
        dates = X[self.timestamp_col].dt.date.tolist()
        for col in self.history_.keys():
            values = [self.encode(d, c, col) for (d, c) in zip(dates, X[col].tolist())]
            out_df[f'{self.SUFFIX}{col}'] = values
            out_df[f'{self.SUFFIX}{col}'] = out_df[f'{self.SUFFIX}{col}'].astype('int')
        return out_df

    @lru_cache(maxsize=5096)
    def encode(self, date: datetime.date, category: Any, column: str) -> int:
        """Encode category according to how many times it appeared until yesterday of `general_firstup`.

        Parameters
        ----------
        date : datetime.date
            YYYYMMDD part of `general_firstup`.
        category : Any
            Category to be encoded.
        column : str
            Column name of category.

        Returns
        -------
        encode: int
        """
        if category not in self.history_[column].keys():  # fit の時に見たことがないカテゴリ
            return 0
        history = self.history_[column][category].loc[:date]
        if history.shape[0] < 1:  # 過去のデータが存在しない
            return 0
        else:
            return history.iloc[-1]


class TimeSeriesTargetEncoder(BaseEstimator, TransformerMixin):
    """時間軸を考慮してカテゴリ項目をターゲットエンコーディングする。

    カテゴリ項目は当該カテゴリのターゲットの平均値としてエンコーディングする。
    時間軸を考慮するため、レコード日付の過去日のレコードのみを用いてターゲットの平均値を計算する。
    """

    SUFFIX = 'ts_te_'

    def __init__(
            self, categorical_columns: List[str],
            target: str, timestamp_col: str, fill_value: int = -1):
        """Initializer.

        Parameters
        ----------
        categorical_columns : List[str]
            Names of categorical columns to be encoded.
        target : str
            Label name.
        timestamp_col: str
            Name of timestamp column.
        """
        self.categorical_columns = categorical_columns
        self.target = target
        self.timestamp_col = timestamp_col
        self.fill_value = fill_value

    def fit(self, X: pd.DataFrame, y=None) -> object:
        X = X[[self.timestamp_col, self.target] + self.categorical_columns].copy()
        assert(X[self.target].min() > self.fill_value)
        X['date'] = X[self.timestamp_col].dt.date
        self.history_ = {}
        for col in self.categorical_columns:
            self.history_[col] = {}
            for c, df in X.groupby(col):
                history = df.groupby('date')[self.target] \
                            .agg(['sum', 'count']) \
                            .sort_index() \
                            .shift(1)
                self.history_[col][c] = history
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out_df = pd.DataFrame(index=X.index)
        X = X[[self.timestamp_col] + self.categorical_columns].copy()
        dates = X[self.timestamp_col].dt.date.tolist()
        for col in self.history_.keys():
            values = [self.encode(d, c, col) for (d, c) in zip(dates, X[col].tolist())]
            out_df[f'{self.SUFFIX}{col}'] = values
            out_df[f'{self.SUFFIX}{col}'] = out_df[f'{self.SUFFIX}{col}'].astype('float32')
        return out_df

    @ lru_cache(maxsize=5096)
    def encode(self, date: datetime.date, category: Any, column: str) -> float:
        """指定したカテゴリの指定日付次点におけるターゲットエンコーディングを計算する。

        Parameters
        ----------
        date : datetime.date
            日付。
        category : Any
            カテゴリ。
        column : str
            カテゴリ項目。

        Returns
        -------
        target_encoding: float
            エンコーディング結果。
        """
        if category not in self.history_[column].keys():  # fit の時に見たことがないカテゴリ
            return self.fill_value
        history = self.history_[column][category].loc[:date]
        if history.shape[0] < 1:  # 過去のデータが存在しない
            return self.fill_value
        else:
            # 初日は欠損しているので埋める
            latest = history.iloc[-1]
            encoding = latest['sum'] / latest['count']
            return self.fill_value if np.isnan(encoding) else encoding
