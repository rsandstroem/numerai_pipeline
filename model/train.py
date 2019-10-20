#! /home/rikard/anaconda3/envs/numerai_kazutsugi python
# coding: utf-8
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import linear_model
from common import *


def train_lgb(X_train, y_train):
    print('Training model...')
    params = {}
    # params['application'] = 'binary'
    params['boosting_type'] = 'gbdt'
    # params['n_estimators'] = 10000
    params['n_estimators'] = 65
    params['num_threads'] = 4
    params['num_leaves'] = 11
    params['min_child_samples'] = 4  # 5
    params['max_depth'] = 7
    params['learning_rate'] = 0.03  # 0.01
    # params['early_stopping_rounds'] = 2
    params['early_stopping_rounds'] = None
    params['reg_lambda'] = 21.1  # 1
    params['reg_alpha'] = 0.648
    params['metric'] = 'binary_logloss'

    d_train = lgb.Dataset(X_train, label=y_train)
    clf = lgb.train(params, d_train)
    return clf


def train_linear(X_train, y_train):
    linear = linear_model.LinearRegression()
    linear.fit(X_train, y_train)
    print(f'linear model training score: {linear.score(X_train, y_train)}')
    joblib.dump(linear, PROJECT_PATH / 'models' / 'linear.pkl')
    return linear


def main():
    """
    [summary]
    """
    print("# Loading data...")
    data_folder = PROJECT_PATH / 'data'
    # The training data is used to train your model how to predict the targets.
    training_data = pd.read_csv(
        data_folder / 'numerai_training_data.csv').set_index("id")

    feature_names = [
        f for f in training_data.columns if f.startswith("feature")]
    print(f"Loaded {len(feature_names)} features")

    model = train_linear(
        training_data[feature_names].values,
        training_data[TARGET_NAME].values)

    print("Generating predictions")
    training_data[PREDICTION_NAME] = model.predict(
        training_data[feature_names].values)

    # Check the per-era correlations on the training set
    train_correlations = training_data.groupby("era").apply(score)
    print(
        f"On training the correlation = {train_correlations.mean()} +- {train_correlations.std()}")
    print(
        f"On training the average per-era payout is {payout(train_correlations).mean()}")
    print('...done!')

if __name__ == '__main__':
    main()
