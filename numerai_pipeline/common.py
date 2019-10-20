#! /home/rikard/anaconda3/envs/numerai_kazutsugi python
# coding: utf-8
import numpy as np
import pandas as pd
from pathlib import Path

TOURNAMENT_NAME = "kazutsugi"
TARGET_NAME = f"target_{TOURNAMENT_NAME}"
PREDICTION_NAME = f"prediction_{TOURNAMENT_NAME}"
PROJECT_PATH = Path("/home/rikard/WORK/numerai_kazutsugi")

BENCHMARK = 0.002
BAND = 0.04


# Submissions are scored by spearman correlation
def score(df):
    # method="first" breaks ties based on order in array
    return np.corrcoef(
        df[TARGET_NAME],
        df[PREDICTION_NAME].rank(pct=True, method="first")
    )[0, 1]


# The payout function
def payout(scores):
    return ((scores - BENCHMARK)/BAND).clip(lower=-1, upper=1)
