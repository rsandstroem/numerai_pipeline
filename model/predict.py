#! /home/rikard/anaconda3/envs/numerai_kazutsugi python
# coding: utf-8
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from common import *


def main():
    """
    [summary]
    """
    print("# Loading data...")
    data_folder = PROJECT_PATH / 'data'
    # The tournament data is the data that Numerai uses to evaluate your model.
    tournament_data = pd.read_csv(
        data_folder / 'numerai_tournament_data.csv').set_index('id')
    feature_names = [
        f for f in tournament_data.columns if f.startswith('feature')]
    print(f'Loaded {len(feature_names)} features')

    # TODO: do not hardcode the model
    model = joblib.load(PROJECT_PATH / 'models' / 'linear.pkl')

    print("Generating predictions")
    tournament_data[PREDICTION_NAME] = model.predict(
        tournament_data[feature_names].values)

    # Check the per-era correlations on the validation set
    validation_data = tournament_data[tournament_data.data_type == "validation"]
    validation_correlations = validation_data.groupby(
        "era").apply(score)
    print(
        f"On validation the correlation = {validation_correlations.mean()} +- {validation_correlations.std()}")
    print(
        f"On validation the average per-era payout is {payout(validation_correlations).mean()}")

    tournament_data[PREDICTION_NAME].to_csv(
        data_folder /
        (TOURNAMENT_NAME + "_submission.csv"),
        header=True)
    # Now you can upload these predictions on https://numer.ai
    print('...done!')

if __name__ == '__main__':
    main()
