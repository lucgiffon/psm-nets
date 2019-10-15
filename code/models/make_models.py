"""
Function for downloading pretrained models
"""
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import keras
from skluc.utils import logger

from palmnet.utils import root_dir
from skluc.utils.osutils import download_file, check_file_md5

MAP_DATA_MODEL_URL = {
    "mnist_lenet": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/mnist_lenet_1570207294.h5",
}

MAP_DATA_MODEL_MD5SUM = {
    "mnist_lenet": "26d44827c84d44a9fc8f4e021b7fe4d2"
}

def _download_single_model(output_dirpath, model):
    download_path = download_file(MAP_DATA_MODEL_URL[model], root_dir / output_dirpath)
    logger.info(f"Save {model} to {download_path}")
    check_file_md5(download_path, MAP_DATA_MODEL_MD5SUM[model])


@click.command()
@click.argument('model', default="all")
@click.argument('output_dirpath', type=click.Path())
def main(output_dirpath, model):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    _download_single_model(output_dirpath, model)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
