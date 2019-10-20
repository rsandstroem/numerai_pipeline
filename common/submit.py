#! /home/rikard/anaconda3/envs/numerai_kazutsugi python
# coding: utf-8
from pathlib import Path
import json
import numerox as nx
import common


def main():
    user = 'rsai'
    with open('authentication.json', "r") as credentials_file:
        credentials = json.load(credentials_file)[user]

    filename = common.TOURNAMENT_NAME + '_submission.csv'
    print(f'Submitting {filename} for user {user}')
    nx.upload(
        filename,
        tournament=common.TOURNAMENT_NAME,
        public_id=credentials['id'],
        secret_key=credentials['key']
    )
    print('...done!')

if __name__ == '__main__':
    main()
