#!/usr/bin/env python
# coding: utf-8
from pathlib import Path
import sys
import getopt
import joblib
import pandas as pd
from numerai_pipeline import common


def predict(model_name):
    """Predicts target for the tournament data using the specified model.

    Arguments:
        model_name {str} -- Name of the trained model. 
    """
    print(f'Model is {model_name}')
    print("Loading data...")
    data_folder = common.PROJECT_PATH / 'data'
    # The tournament data is the data that Numerai uses to evaluate your model.
    tournament_data = pd.read_csv(
        data_folder / 'numerai_tournament_data.csv').set_index('id')
    feature_names = [
        f for f in tournament_data.columns if f.startswith('feature')]
    print(f'Loaded {len(feature_names)} features')

    try:
        model = joblib.load(common.PROJECT_PATH /
                            'models' / f'{model_name}.pkl')
    except:
        print(f'Model {model_name}.pkl not found')
        exit(2)

    print("Generating predictions")
    tournament_data[common.PREDICTION_NAME] = model.predict(
        tournament_data[feature_names].values)

    # Check the per-era correlations on the validation set
    validation_data = tournament_data[tournament_data.data_type == "validation"]
    validation_correlations = validation_data.groupby(
        "era").apply(common.score)
    print(
        f"On validation the correlation = {validation_correlations.mean()} +- {validation_correlations.std()}")
    print(
        f"On validation the average per-era payout is {common.payout(validation_correlations).mean()}")

    tournament_data[common.PREDICTION_NAME].to_csv(
        data_folder /
        f'{common.TOURNAMENT_NAME}_{model_name}_submission.csv',
        header=True)
    # Now you can upload these predictions on https://numer.ai
    print('...done!')


def main(argv):
    """
    [summary]
    """
    model_name = ''
    try:
        opts, args = getopt.getopt(argv, "hm:", ["model="])
    except getopt.GetoptError:
        print('predict.py -m <model_name>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('predict.py -m <model_name>')
            sys.exit()
        elif opt in ("-m", "--model"):
            model_name = arg
    predict(model_name)


if __name__ == '__main__':
    main(sys.argv[1:])
