'''

GBDT の各種モデルで Classifier を作成する
全てのクラスを使うのか、マイノリティを正例とし他を負例とするのか。
'''

import argparse
from datetime import datetime
import gc
import logging
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_sample_weight, compute_class_weight
from xgboost import XGBClassifier

from dataset import load_df, Y, NOVEL_ID_COL
from preprocessing import N_SPLITS
from utils import get_logger, timer

WORK_DIR = os.environ['ROOT']  # ここで作業する
assert(os.path.isdir(WORK_DIR))
DIR_DATA, DIR_FEATURES, DIR_LOG, DIR_MODEL, DIR_INFERENCE = 'data', 'features', 'log', 'models', 'inference'
assert(os.path.isdir(DIR_DATA))
assert(os.path.isdir(DIR_FEATURES))
assert(os.path.isdir(DIR_MODEL))
assert(os.path.isdir(DIR_LOG))
assert(os.path.isdir(DIR_INFERENCE))


SUBMISSON_FORMAT = ['ncode', 'proba_0', 'proba_1', 'proba_2', 'proba_3', 'proba_4']
MINORITY_CLASSES = [2, 3, 4]
# LR = 0.1
# N_ESTIMATORS = 100


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='''Train classifiers.''')
    parser.add_argument('-t', '--trees', type=int, default=100,
                        help='Number of trees build by GBDT models.')
    parser.add_argument('-r', '--lr', type=float, default=0.1,
                        help='Learning rate used in training GBDT models.')
    parser.add_argument('-s', '--seed',  type=int, default=1, help='Random seed. Default = 1.')
    parser.add_argument('-n', '--n_jobs', type=int, default=-1,
                        help='Number of cpu cores used. Default = -1(use all).')
    parser.add_argument('-e', '--evaluate', action='store_true')
    args = parser.parse_args()
    assert(args.seed >= 0)
    assert(args.n_jobs == -1 or args.n_jobs <= os.cpu_count())
    return args


def get_classifiers(random_state: int, n_jobs: int, class_weight: dict, learning_rate: float,
                    n_estimators: int) -> dict:
    classifiers = {
        'catboost': CatBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            class_weights=class_weight,
            train_dir=DIR_LOG,
            verbose=False),
        'lgbm': LGBMClassifier(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            bagging_fraction=0.8,
            bagging_freq=2,
            feature_fraction=0.8,
            n_jobs=n_jobs,
            random_state=random_state,
            class_weight=class_weight,
            importance_type='gain'),
        # 'xgb': XGBClassifier(
        #     n_estimators=n_estimators,
        #     learning_rate=learning_rate,
        #     random_state=random_state,
        #     n_jobs=n_jobs,
        #     use_label_encoder=False,
        #     eval_metric='mlogloss'),
    }
    return classifiers


def process_1fold(train: pd.DataFrame,
                  X_train_test: pd.DataFrame,
                  filepath_train_features: str,
                  filepath_valid_features: str,
                  filepath_test_features: str,
                  learning_rate: float,
                  n_estimators: int,
                  logger: logging.Logger,
                  evaluate: bool,
                  prefix: str,
                  random_state: int,
                  n_jobs: int) -> None:
    with timer('Load features', logger):
        # 訓練データ
        X_train_ = pd.read_csv(filepath_train_features, index_col=NOVEL_ID_COL)
        X_train = X_train_.join(X_train_test)
        y_train = train.loc[X_train.index, Y].values
        # テストデータ
        X_test_ = pd.read_csv(filepath_test_features, index_col=NOVEL_ID_COL)
        X_test = X_test_.join(X_train_test)
        del X_train_, X_test_
        if evaluate:
            # 評価データ
            X_valid_ = pd.read_csv(filepath_valid_features, index_col=NOVEL_ID_COL)
            X_valid = X_valid_.join(X_train_test)
            y_valid = train.loc[X_valid.index, Y].values
            del X_valid_
        gc.collect()

    classes = np.sort(np.unique(y_train))
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight = {c: w for c, w in zip(classes, weights)}
    classifiers = get_classifiers(random_state, n_jobs, class_weight, learning_rate, n_estimators)
    X_train_, X_eval, y_train_, y_eval = train_test_split(
        X_train, y_train, test_size=0.15, random_state=random_state, stratify=y_train)
    for name, classifier in classifiers.items():
        with timer(f'Train {name}', logger):
            if name == 'xgb':
                sample_weight = compute_sample_weight('balanced', y_train_)
                classifier.fit(X_train_, y_train_, sample_weight=sample_weight,
                               eval_set=[(X_eval, y_eval)], early_stopping_rounds=50)
            elif name == 'lgbm':
                classifier.fit(X_train_, y_train_, eval_set=[(X_eval, y_eval)],
                               early_stopping_rounds=50)
            else:
                classifier.fit(X_train_, y_train_, eval_set=(X_eval, y_eval),
                               early_stopping_rounds=50)
        with timer('Inference', logger):
            proba_train = classifier.predict_proba(X_train)
            proba_test = classifier.predict_proba(X_test)
            if evaluate:
                proba_valid = classifier.predict_proba(X_valid)

        with timer('Evaluation', logger):
            loss_train = log_loss(y_train, proba_train)
            logger.info('Training loss is {:.5f}'.format(loss_train))
            if evaluate:
                loss_valid = log_loss(y_valid, proba_valid)
                logger.info('Validation loss is {:.5f}'.format(loss_valid))

        with timer('Save result', logger):
            with open(os.path.join(DIR_MODEL, f'{prefix}{name}classifier.pkl'), 'wb') as f:
                pickle.dump(classifier, f)

            columns = [f'proba_{c}' for c in classes]
            proba_train = pd.DataFrame(proba_train, columns=columns, index=X_train.index)
            proba_train.to_csv(
                os.path.join(DIR_INFERENCE, f'{prefix}proba_train_{name}.csv'),
                float_format='%.18f')
            proba_test = pd.DataFrame(proba_test, columns=columns, index=X_test.index)
            proba_test.to_csv(
                os.path.join(DIR_INFERENCE, f'{prefix}proba_test_{name}.csv'),
                float_format='%.18f')
            if evaluate:
                proba_valid = pd.DataFrame(proba_valid, columns=columns, index=X_valid.index)
                proba_valid.to_csv(
                    os.path.join(DIR_INFERENCE, f'{prefix}proba_valid_{name}.csv'),
                    float_format='%.18f')
    return None


def train_models(train: pd.DataFrame,
                 logger: logging.Logger,
                 evaluate: bool,
                 random_state: int,
                 n_jobs: int,
                 learning_rate: float,
                 n_estimators: int,
                 positive_class: Optional[int] = None):
    assert(isinstance(positive_class, int) or positive_class is None)
    if positive_class is not None:
        assert(positive_class in MINORITY_CLASSES)
        logger.info('Use {} as positive class others as negative'.format(positive_class))
        mask = (train[Y] == positive_class).values  # True: 1, False: 0 に置換
        train.loc[mask, Y] = 1
        train.loc[~mask, Y] = 0
        prefix = f'minority{positive_class}_'
    else:
        prefix = ''

    with timer('Load common features', logger):
        X_train_test = pd.read_csv(
            os.path.join(DIR_FEATURES, 'features_train_test.csv'),
            index_col=NOVEL_ID_COL)

    # Time series split
    logger.info('Time series cv')
    prefix_ts = prefix + 'ts_'
    process_1fold(
        train=train,
        X_train_test=X_train_test,
        filepath_train_features=os.path.join(DIR_FEATURES, 'features_train_ts.csv'),
        filepath_valid_features=os.path.join(DIR_FEATURES, 'features_valid_ts.csv'),
        filepath_test_features=os.path.join(DIR_FEATURES, 'features_test_ts.csv'),
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        logger=logger,
        evaluate=evaluate,
        prefix=prefix_ts,
        random_state=random_state,
        n_jobs=n_jobs
    )
    logger.info('Group k-fold cv')
    for i in range(N_SPLITS):
        prefix_grp = prefix + f'grp{1 + i}_'
        process_1fold(
            train=train,
            X_train_test=X_train_test,
            filepath_train_features=os.path.join(DIR_FEATURES, f'features_train_grp{i + 1}.csv'),
            filepath_valid_features=os.path.join(DIR_FEATURES, f'features_valid_grp{i + 1}.csv'),
            filepath_test_features=os.path.join(DIR_FEATURES, f'features_test_grp{i + 1}.csv'),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            logger=logger,
            evaluate=evaluate,
            prefix=prefix_grp,
            random_state=random_state,
            n_jobs=n_jobs
        )


def run(learning_rate: float, n_estimators: int, random_state: int, n_jobs: int, evaluate: bool) -> None:
    os.chdir(WORK_DIR)
    logger = get_logger(__name__, os.path.join(
        DIR_LOG, datetime.now().strftime('phase1_%Y%m%d_%H%M%S.log')))
    logger.info('Start')
    logger.info('`learning_rate` is {:.5f}'.format(learning_rate))
    logger.info('`n_estimators` is {}'.format(n_estimators))
    logger.info('`random_state` is {}'.format(random_state))
    logger.info('`n_jobs` is {}'.format(n_jobs))
    logger.info('`evaluate` is {}'.format(evaluate))
    train = load_df(os.path.join(DIR_DATA, 'train.csv'), test=False)
    train_models(train=train.copy(),
                 logger=logger,
                 evaluate=evaluate,
                 random_state=random_state,
                 n_jobs=n_jobs,
                 learning_rate=learning_rate,
                 n_estimators=n_estimators,
                 positive_class=None)
    return None


if __name__ == '__main__':
    args = get_args()
    run(learning_rate=args.lr, n_estimators=args.trees,
        random_state=args.seed, n_jobs=args.n_jobs, evaluate=args.evaluate)
