# coding: utf-8
from pathlib import Path
import json
import numerox as nx
from numerai_pipeline import common


def main():
    user = 'rsai'
    with open('authentication.json', "r") as credentials_file:
        credentials = json.load(credentials_file)[user]

    data_folder = common.PROJECT_PATH / 'data'
    filename = common.TOURNAMENT_NAME + '_submission.csv'
    print(f'Submitting {filename} for user {user}')
    nx.upload(
        data_folder /
        (common.TOURNAMENT_NAME + "_submission.csv"),
        tournament=common.TOURNAMENT_NAME,
        public_id=credentials['id'],
        secret_key=credentials['key']
    )
    print('...done!')


if __name__ == '__main__':
    main()
