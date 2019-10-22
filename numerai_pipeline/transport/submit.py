# coding: utf-8
from pathlib import Path
import sys
import getopt
import json
import numerox as nx
from numerai_pipeline import common


def submit(model_name, user):
    """Submits the results to the competition server.

    Arguments:
        model_name {str} -- Name of the model used for prediction.
        user {str} -- User name, used for looking up submission credentials.
    """
    print(f'Model is {model_name}')
    print(f'User is {user}')
    with open('authentication.json', "r") as credentials_file:
        credentials = json.load(credentials_file)[user]

    data_folder = common.PROJECT_PATH / 'data'
    filename = f'{common.TOURNAMENT_NAME}_{model_name}_submission.csv'
    print(f'Submitting {filename} for user {user}')
    nx.upload(
        data_folder / filename,
        tournament=common.TOURNAMENT_NAME,
        public_id=credentials['id'],
        secret_key=credentials['key']
    )
    print('...done!')


def main(argv):
    """
    [summary]
    """
    model_name = ''
    user = ''
    try:
        opts, args = getopt.getopt(argv, "hm:u:", ["model=", "user="])
    except getopt.GetoptError:
        print('submit.py -m <model_name> -u <user>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('submit.py -m <model_name> -u <user>')
            sys.exit()
        elif opt in ("-m", "--model"):
            model_name = arg
        elif opt in ("-u", "--user"):
            user = arg
    submit(model_name, user)


if __name__ == '__main__':
    main(sys.argv[1:])
