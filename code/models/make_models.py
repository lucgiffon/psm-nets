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
    "cifar10_vgg19_4096x4096": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/cifar10_vgg19_4096x4096_1570693209.h5",
    "cifar100_vgg19_4096x4096": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/cifar100_vgg19_4096x4096_1570789868.h5",
    "svhn_vgg19_4096x4096": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/svhn_vgg19_4096x4096_1570786657.h5",
    "cifar10_vgg19_2048x2048": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/cifar10_vgg19_2048x2048_1572303047.h5",
    "cifar100_vgg19_2048x2048": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/cifar100_vgg19_2048x2048_1572278802.h5",
    "svhn_vgg19_2048x2048": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/svhn_vgg19_2048x2048_1572278831.h5",
}

MAP_DATA_MODEL_MD5SUM = {
    "mnist_lenet": "26d44827c84d44a9fc8f4e021b7fe4d2",
    "cifar10_vgg19_4096x4096": "a3ece534a8e02d17453dffc095048f65",
    "cifar100_vgg19_4096x4096": "cb1bd8558f385030c6c68808023918e0",
    "svhn_vgg19_4096x4096": "204e41afbc84d1806822a60a9558ea52",
    "cifar10_vgg19_2048x2048": "98cece5432051adc2330699a40940dfd",
    "cifar100_vgg19_2048x2048": "57d6bf6434428a81e702271367eac4d1",
    "svhn_vgg19_2048x2048": "d5697042804bcc646bf9882a45dedd9e"
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
