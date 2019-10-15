# -*- coding: utf-8 -*-
"""
Functions for downloading data set.

"""
import tempfile
import urllib.request

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import numpy as np
import scipy.io as sio
from skluc.utils.osutils import read_matfile, download_file

from skluc.utils import logger


def load_svhn_data():
    data_root_url = "http://ufldl.stanford.edu/housenumbers/"
    data_leaf_values = {
            "train": "train_32x32.mat",
            "test": "test_32x32.mat",
    }
    data_arrays = {}

    with tempfile.TemporaryDirectory() as d_tmp:
        for leaf_name, leaf in data_leaf_values.items():
            leaf_url = data_root_url + leaf
            matfile_path = download_file(leaf_url, d_tmp, leaf)
            data_arrays[leaf_name] = read_matfile(matfile_path)

    return data_arrays["train"], data_arrays["test"]

def _download_single_dataset(output_dirpath, dataname):
    if MAP_NAME_CLASSES_PRESENCE[dataname]:
        (x_train, y_train), (x_test, y_test) = MAP_NAME_DATASET[dataname]()
        map_savez = {"x_train": x_train,
                     "y_train": y_train,
                     "x_test": x_test,
                     "y_test": y_test
                     }
    else:
        X = MAP_NAME_DATASET[dataname]()
        map_savez = {"x_train": X}

    output_path = project_dir / output_dirpath / dataname
    logger.info(f"Save {dataname} to {output_path}")
    np.savez(output_path, **map_savez)


@click.command()
@click.argument('dataset', default="all")
@click.argument('output_dirpath', type=click.Path())
def main(output_dirpath, dataset):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    _download_single_dataset(output_dirpath, dataset)


MAP_NAME_DATASET = {
    "svhn": load_svhn_data,
}

MAP_NAME_CLASSES_PRESENCE = {
    "svhn": True,
}


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
