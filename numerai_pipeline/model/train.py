#!/usr/bin/env python
# coding: utf-8
from pathlib import Path
import sys
import getopt
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import linear_model
from xgboost import XGBRegressor
from numerai_pipeline import common


def train_lgb(X_train, y_train):
    print('Training LightGBM model')
    params = {
        'num_boost_round': 10000,
        'num_threads': -1,
        'objective': 'regression',
        'metric': 'regression',
        'verbose': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 128,
        'feature_fraction': 0.9,
        'min_data_in_leaf': 5,
        'two_round': True,
        'seed_value': 0,
    }

    d_train = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, d_train)
    joblib.dump(model, common.PROJECT_PATH / 'models' / 'lgb.pkl')
    return model


def train_linear(X_train, y_train):
    print('Training linear model')
    linear = linear_model.LinearRegression()
    linear.fit(X_train, y_train)
    print(f'linear model training score: {linear.score(X_train, y_train)}')
    joblib.dump(linear, common.PROJECT_PATH / 'models' / 'linear.pkl')
    return linear


def train_xgb(X_train, y_train):
    print('Training XGB model')
    # For faster experimentation you can decrease n_estimators to 200, for better performance increase to 20,000
    model = XGBRegressor(max_depth=5, learning_rate=0.01,
                         n_estimators=2000, n_jobs=-1, colsample_bytree=0.1)
    model.fit(X_train, y_train)
    print(f'xgb model training score: {model.score(X_train, y_train)}')
    joblib.dump(model, common.PROJECT_PATH / 'models' / 'xgb.pkl')
    return model


def train(model_name):
    """Train the model and writes the trained model to the model folder. 

    Arguments:
        model_name {str} -- Name of the model to train. Must be a valid option implemented in the code.
    """
    print(f'Model is {model_name}')
    print("Loading data...")
    data_folder = common.PROJECT_PATH / 'data'
    # The training data is used to train your model how to predict the targets.
    training_data = pd.read_csv(
        data_folder / 'numerai_training_data.csv').set_index("id")

    feature_names = [
        f for f in training_data.columns if f.startswith("feature")]
    print(f"Loaded {len(feature_names)} features")

    if model_name == 'linear':
        model = train_linear(
            training_data[feature_names].values,
            training_data[common.TARGET_NAME].values)
    elif model_name == 'xgb':
        model = train_xgb(
            training_data[feature_names].values,
            training_data[common.TARGET_NAME].values)
    elif model_name == 'lgb':
        model = train_lgb(
            training_data[feature_names].values,
            training_data[common.TARGET_NAME].values)
    else:
        print(f'Specified model {model_name} not supported')
        exit(2)

    print('Generating predictions:')
    training_data[common.PREDICTION_NAME] = model.predict(
        training_data[feature_names].values)

    # Check the per-era correlations on the training set
    train_correlations = training_data.groupby("era").apply(common.score)
    print(
        f"On training the correlation = {train_correlations.mean()} +- {train_correlations.std()}")
    print(
        f"On training the average per-era payout is {common.payout(train_correlations).mean()}")
    print('...done!')


def main(argv):
    """
    [summary]
    """
    model_name = ''
    try:
        opts, args = getopt.getopt(argv, "hm:", ["model="])
    except getopt.GetoptError:
        print('train.py -m <model_name>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('train.py -m <model_name>')
            sys.exit()
        elif opt in ("-m", "--model"):
            model_name = arg
    train(model_name)


if __name__ == '__main__':
    main(sys.argv[1:])
