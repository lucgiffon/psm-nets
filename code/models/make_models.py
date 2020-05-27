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
    "mnist_lenet-1": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/mnist_lenet_1_1586853546.h5",
    "mnist_lenet-2": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/mnist_lenet_2_1586853549.h5",
    "mnist_lenet-3": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/mnist_lenet_3_1586854101.h5",
    "cifar10_vgg19_4096x4096": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/cifar10_vgg19_4096x4096_1570693209.h5",
    "cifar100_vgg19_4096x4096": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/cifar100_vgg19_4096x4096_1570789868.h5",
    "svhn_vgg19_4096x4096": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/svhn_vgg19_4096x4096_1570786657.h5",
    "cifar10_vgg19_2048x2048": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/cifar10_vgg19_2048x2048_1572303047.h5",
    "cifar10_vgg19_2048x2048-1": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/cifar10_vgg19_2048x2048_1_1586857195.h5",
    "cifar10_vgg19_2048x2048-2": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/cifar10_vgg19_2048x2048_2_1586849939.h5",
    "cifar10_vgg19_2048x2048-3": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/cifar10_vgg19_2048x2048_3_1586849939.h5",
    "cifar100_vgg19_2048x2048": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/cifar100_vgg19_2048x2048_1572278802.h5",
    "cifar100_vgg19_2048x2048-1": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/cifar100_vgg19_2048x2048_1_1586850015.h5",
    "cifar100_vgg19_2048x2048-2": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/cifar100_vgg19_2048x2048_2_1586850015.h5",
    "cifar100_vgg19_2048x2048-3": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/cifar100_vgg19_2048x2048_3_1586850015.h5",
    "svhn_vgg19_2048x2048": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/svhn_vgg19_2048x2048_1572278831.h5",
    "svhn_vgg19_2048x2048-1": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/svhn_vgg19_2048x2048_1_1586873524.h5",
    "svhn_vgg19_2048x2048-2": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/svhn_vgg19_2048x2048_2_1586877914.h5",
    "svhn_vgg19_2048x2048-3": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/svhn_vgg19_2048x2048_3_1586878915.h5",
    "mnist_500": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/mnist_500.h5",
    "cifar100_resnet20": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/resnet_20_cifar100.h5",
    "cifar100_resnet50": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/resnet_50_cifar100.h5",
    "cifar100_resnet50_new": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/resnet_resnet50_cifar100_1587927534.h5",
    "cifar100_resnet50_new-1": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/resnet_resnet50_cifar100_1_1588162548.h5",
    "cifar100_resnet50_new-2": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/resnet_resnet50_cifar100_2_1588107732.h5",
    "cifar100_resnet50_new-3": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/resnet_resnet50_cifar100_3_1588102661.h5",
    "cifar100_resnet20_new": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/resnet_resnet20_cifar100_1588012286.h5",
    "cifar100_resnet20_new-1": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/resnet_resnet20_cifar100_1_1588096045.h5",
    "cifar100_resnet20_new-2": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/resnet_resnet20_cifar100_2_1588101554.h5",
    "cifar100_resnet20_new-3": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/resnet_resnet20_cifar100_3_1588090286.h5",
    "cifar10_tensor_train_base": "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/cifar10_tensor_train_base_1585409008.h5",
}

MAP_DATA_MODEL_MD5SUM = {
    "mnist_lenet": "26d44827c84d44a9fc8f4e021b7fe4d2",
    "mnist_lenet-1": "2e14e504dbe4be3881f77a485034437a",
    "mnist_lenet-2": "f87245aa886f6597be01bc78caa9b9e3",
    "mnist_lenet-3": "1d294b8320c9aa21f7486135cc2559e6",
    "cifar10_vgg19_4096x4096": "a3ece534a8e02d17453dffc095048f65",
    "cifar100_vgg19_4096x4096": "cb1bd8558f385030c6c68808023918e0",
    "svhn_vgg19_4096x4096": "204e41afbc84d1806822a60a9558ea52",
    "cifar10_vgg19_2048x2048": "98cece5432051adc2330699a40940dfd",
    "cifar10_vgg19_2048x2048-1": "631dc91c11e39ae10bdaec11f5ea1d7b",
    "cifar10_vgg19_2048x2048-2": "8093a871da80919a42f52e3cfdb0d157",
    "cifar10_vgg19_2048x2048-3": "dfc3213fc831cad6e87def1693bdf290",
    "cifar100_vgg19_2048x2048": "57d6bf6434428a81e702271367eac4d1",
    "cifar100_vgg19_2048x2048-1": "eac0b735cd3715286b9e3e6f9cace515",
    "cifar100_vgg19_2048x2048-2": "6d710e8b849d31b662497f9582f99a94",
    "cifar100_vgg19_2048x2048-3": "00ac0bbe7078e96faef71055b92cf01e",
    "svhn_vgg19_2048x2048": "d5697042804bcc646bf9882a45dedd9e",
    "svhn_vgg19_2048x2048-1": "837ea81028f8a5c15c315bd5c2d75496",
    "svhn_vgg19_2048x2048-2": "96f3a7ebb9236b248e624cb6cb4bb113",
    "svhn_vgg19_2048x2048-3": "7ab6fac40acdc6effe2e2dc1b3c3ef22",
    "mnist_500": "1b023b05a01f24a99ac9a460488068f8",
    "cifar100_resnet20": "4845ec6461c5923fc77f42a157b6d0c1",
    "cifar100_resnet50": "d76774eb6f871b1192c144f0dc29340e",
    "cifar100_resnet50_new": "11b49f1b1422e9d03d6cf2ae2c0249b1",
    "cifar100_resnet50_new-1": "aeb56f5553ee0146040f6acf6e0984ce",
    "cifar100_resnet50_new-2": "31339656a2f72f17e3ebe232b365f47b",
    "cifar100_resnet50_new-3": "4ee7b2188ed02180218a221418ba9ee2",
    "cifar100_resnet20_new": "20ea1f7a9d73aaafae311c1effe06410",
    "cifar100_resnet20_new-1": "f6667d674de2321f175b8138d5e7c523",
    "cifar100_resnet20_new-2": "beab1c51b7c33a8cf74b2c9c9618e3c2",
    "cifar100_resnet20_new-3": "a5a3b0e92ae895d842d9ed1e54c06a5c",
    "cifar10_tensor_train_base": "e985fbe4ade6893b7fb92655be1b846f"
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
