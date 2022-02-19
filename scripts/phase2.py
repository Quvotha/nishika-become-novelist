from datetime import datetime
import glob
import logging
import os
from typing import List


from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.metrics import log_loss

from dataset import load_train_test, Y, NOVEL_ID_COL
from preprocessing import N_SPLITS
from utils import get_logger, timer

WORK_DIR = os.environ['ROOT']  # ここで作業する
assert(os.path.isdir(WORK_DIR))
DIR_DATA, DIR_LOG, DIR_MODEL, DIR_INFERENCE = 'data', 'log', 'models', 'inference'
assert(os.path.isdir(DIR_DATA))
assert(os.path.isdir(DIR_MODEL))
assert(os.path.isdir(DIR_LOG))
assert(os.path.isdir(DIR_INFERENCE))

SUBMIT_COLUMNS = ['proba_0', 'proba_1', 'proba_2', 'proba_3', 'proba_4']


def load_inference_dfs(filepaths: List[str]) -> pd.DataFrame:
    infer_dfs = []
    for f in filepaths:
        df = pd.read_csv(f, index_col=NOVEL_ID_COL)
        suffix = os.path.basename(f)
        df.columns = [f'{c}_{suffix}' for c in df.columns]
        infer_dfs.append(df)
    infer_df = pd.concat(infer_dfs, axis=1)
    assert(infer_df.isnull().sum().sum() == 0)
    return infer_df


def infer(classifier, filepaths: List[str]) -> pd.DataFrame:
    infer_df = load_inference_dfs(filepaths)
    proba_df = pd.DataFrame(data=classifier.predict_proba(infer_df.values),
                            index=infer_df.index,
                            columns=SUBMIT_COLUMNS)
    return proba_df


def evaluate_model(
        classifier, filepaths: List[str],
        train: pd.DataFrame, logger: logging.Logger) -> None:
    infer_df = load_inference_dfs(filepaths)
    y_valid = train.loc[infer_df.index, Y].values
    loss = log_loss(y_valid, classifier.predict_proba(infer_df.values))
    logger.info('Validation loass is {:.5f}'.format(loss))
    return None


def train_model(filepaths: List[str], train: pd.DataFrame, logger: logging.Logger,
                random_state: int, n_jobs: int):
    infer_df = load_inference_dfs(filepaths)
    y_train = train.loc[infer_df.index, Y].values
    classifier = LGBMClassifier(
        learning_rate=0.1,
        n_estimators=100,
        bagging_fraction=0.8,
        bagging_freq=2,
        feature_fraction=0.8,
        n_jobs=n_jobs,
        random_state=random_state,
        class_weight='balanced',
        importance_type='gain')
    classifier.fit(infer_df.values, y_train)
    loss = log_loss(y_train, classifier.predict_proba(infer_df))
    logger.info('Training loass is {:.5f}'.format(loss))
    return classifier


def run(random_state: int, n_jobs: int):
    os.chdir(WORK_DIR)
    logger = get_logger(__name__, os.path.join(
        DIR_LOG, datetime.now().strftime('phase2_%Y%m%d_%H%M%S.log')))
    logger.info('Start')
    logger.info('`random_state` is {}'.format(random_state))
    logger.info('`n_jobs` is {}'.format(n_jobs))

    train, test = load_train_test(
        os.path.join(DIR_DATA, 'train.csv'),
        os.path.join(DIR_DATA, 'test.csv'),
    )
    # Time series split
    filepaths_train = sorted(glob.glob(os.path.join(DIR_INFERENCE, '*ts_proba_train_*.csv')))
    logger.debug('Training filepaths: {}'.format(', '.join(filepaths_train)))
    classifier = train_model(filepaths_train, train, logger, random_state, n_jobs)
    filepaths_valid = sorted(glob.glob(os.path.join(DIR_INFERENCE, '*ts_proba_valid_*.csv')))
    if len(filepaths_valid) > 0:
        logger.debug('Validation filepaths: {}'.format(', '.join(filepaths_valid)))
        evaluate_model(classifier, filepaths_valid, train, logger)
    filepaths_test = sorted(glob.glob(os.path.join(DIR_INFERENCE, '*ts_proba_test_*.csv')))
    logger.debug('Test filepaths: {}'.format(', '.join(filepaths_test)))
    proba_ts = infer(classifier, filepaths_test)
    proba_ts.to_csv(os.path.join(DIR_INFERENCE, 'submission.ts.csv'), float_format='%.18f')

    # Group K Foold
    proba_dfs = []
    for i in range(N_SPLITS):
        filepaths_train = sorted(glob.glob(os.path.join(
            DIR_INFERENCE, f'*grp{i + 1}_proba_train_*.csv')))
        logger.debug('Training filepaths: {}'.format(', '.join(filepaths_train)))
        classifier = train_model(filepaths_train, train, logger, random_state, n_jobs)
        filepaths_valid = sorted(glob.glob(os.path.join(
            DIR_INFERENCE, f'*grp{i + 1}_proba_valid_*.csv')))
        if len(filepaths_valid) > 0:
            logger.debug('Validation filepaths: {}'.format(', '.join(filepaths_valid)))
            evaluate_model(classifier, filepaths_valid, train, logger)
        filepaths_test = sorted(glob.glob(os.path.join(
            DIR_INFERENCE, f'*grp{i + 1}_proba_test_*.csv')))
        logger.debug('Test filepaths: {}'.format(', '.join(filepaths_test)))
        proba = infer(classifier, filepaths_test)
        proba_dfs.append(proba)
    proba_grp = pd.concat(proba_dfs).reset_index()
    proba_grp = proba_grp.groupby(NOVEL_ID_COL)[SUBMIT_COLUMNS].mean()
    proba_grp['sum'] = proba_grp.sum(axis=1)
    for c in SUBMIT_COLUMNS:
        proba_grp[c] = proba_grp[c] / proba_grp['sum']
    proba_grp[SUBMIT_COLUMNS].to_csv(
        os.path.join(DIR_INFERENCE, 'submission_grp.csv'),
        float_format='%.18f')

    proba = proba_ts.copy()
    for c in SUBMIT_COLUMNS:
        proba.loc[proba_grp.index, c] = proba_grp[c]
    proba.to_csv(
        os.path.join(DIR_INFERENCE, 'submission.csv'),
        float_format='%.18f')


if __name__ == '__main__':
    run(random_state=1, n_jobs=-1)
