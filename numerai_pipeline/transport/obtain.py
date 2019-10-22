# coding: utf-8
from pathlib import Path
import zipfile
import numerox as nx
from numerai_pipeline import common


def main():
    """
    Download and unzip the latest dataset from Numerai.
    This produces csv files in the data folder.
    """
    print('Obtaining data...')

    data_folder = common.PROJECT_PATH / 'data'
    filename = 'numerai_dataset.zip'

    # download dataset from numerai
    nx.download(data_folder / filename, load=False)

    # unzip content
    with zipfile.ZipFile(data_folder / filename, 'r') as zip_ref:
        zip_ref.extractall(data_folder)

    print('...done!')


if __name__ == '__main__':
    main()
