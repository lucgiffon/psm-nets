import os
import zlib
import pandas as pd

import pathlib
import random

import numpy as np
from pathlib import Path

import time
from skluc.utils import logger

from palmnet.data import Mnist, Cifar10, Cifar100, Svhn, Test
from palmnet.visualization.utils import get_df


class ParameterManager(dict):
    def __init__(self, dct_params, **kwargs):
        super().__init__(**dct_params, **kwargs)
        self.__init_identifier()
        self.__init_output_file()

    def __init_identifier(self):
        job_id = os.environ.get('OAR_JOB_ID')  # in case it is running with oarsub job scheduler
        if job_id is None:
            job_id = str(int(time.time()))
            job_id = int(job_id) + random.randint(0, 10 ** len(job_id))
        else:
            job_id = int(job_id)

        self["identifier"] = str(job_id)

    def __init_output_file(self):
        self["output_file_resprinter"] = Path(self["identifier"] + "_results.csv")
        self["output_file_modelprinter"] = Path(self["identifier"] + "_model_layers.pckle")

    def __init_seed(self):
        if self["--seed"] is not None:
            self["--seed"] = int(self["--seed"])
            np.random.seed(self["--seed"])
        else:
            self["--seed"] = int(self["--seed"])  # should break

    def get_dataset(self):
        """
        Return dataset in shape n x d.

        n: number of observations.
        d: dimensionality of observations.

        :return:
        """
        if self["--mnist"]:
            # (x_train, y_train), (x_test, y_test) =  Mnist.load_data()
            # return (x_train, y_train), (x_test, y_test)
            return Mnist
        elif self["--cifar10"]:
            # (x_train, y_train), (x_test, y_test) = Cifar10.load_data()
            # return (x_train, y_train), (x_test, y_test)
            return Cifar10
        elif self["--cifar100"]:
            # (x_train, y_train), (x_test, y_test) = Cifar100.load_data()
            # return (x_train, y_train), (x_test, y_test)
            return Cifar100
        elif self["--svhn"]:
            # (x_train, y_train), (x_test, y_test) = Svhn.load_data()
            # return (x_train, y_train), (x_test, y_test)
            return Svhn
        elif self["--test-data"]:
            # (x_train, y_train), (x_test, y_test) = Test.load_data()
            # return (x_train, y_train), (x_test, y_test)
            return Test

        else:
            raise NotImplementedError("No dataset specified.")

    def get_model(self):
        if self["--mnist-lenet"]:
            return Mnist.load_model() # for now there is only one model for each dataset... it may change
        elif self["--cifar10-vgg19"]:
            return Cifar10.load_model()
        elif self["--cifar100-vgg19"]:
            return Cifar100.load_model()
        elif self["--svhn-vgg19"]:
            return Svhn.load_model()
        elif self["--test-model"]:
            return Test.load_model()
        elif self["--mnist-500"]:
            return Mnist.load_model("mnist-500")
        elif self["--cifar100-resnet50"]:
            return Cifar100.load_model("cifar100-resnet50")
        elif self["--cifar100-resnet20"]:
            return Cifar100.load_model("cifar100-resnet20")
        else:
            raise NotImplementedError("No dataset specified.")

class ParameterManagerPalminize(ParameterManager):
    def __init__(self, dct_params, **kwargs):
        super().__init__(self, **dct_params, **kwargs)
        self["--sparsity-factor"] = int(self["--sparsity-factor"]) if self["--sparsity-factor"] is not None else None
        self["--nb-iteration-palm"] = int(self["--nb-iteration-palm"]) if self["--nb-iteration-palm"] is not None else None
        self["--delta-threshold"] = float(self["--delta-threshold"]) if self["--delta-threshold"] is not None else None
        self["--nb-factor"] = int(self["--nb-factor"]) if self["--nb-factor"] is not None else None


class ParameterManagerTensotrainAndTuckerDecomposition(ParameterManager):
    def __init__(self, dct_params, **kwargs):
        super().__init__(self, **dct_params, **kwargs)
        self["--min-lr"] = float(self["--min-lr"]) if self["--min-lr"] is not None else None
        self["--max-lr"] = float(self["--max-lr"]) if self["--max-lr"] is not None else None
        self["--lr"] = float(self["--lr"]) if self["--lr"] is not None else None
        self["--nb-epoch"] = int(self["--nb-epoch"]) if self["--nb-epoch"] is not None else None
        self["--epoch-step-size"] = int(self["--epoch-step-size"]) if self["--epoch-step-size"] is not None else None

        # tensortrain parameters
        self["--rank-value"] = int(self["--rank-value"]) if self["--rank-value"] is not None else None
        self["--order"] = int(self["--order"]) if self["--order"] is not None else None

        self.__init_hash_expe()
        self.__init_output_file()

    def __init_output_file(self):
        self["output_file_resprinter"] = Path(self["hash"] + "_results.csv")
        self["output_file_modelprinter"] = Path(self["hash"] + "_model.h5")
        self["output_file_notfinishedprinter"] = Path(self["hash"] + ".notfinished")
        self["output_file_csvcbprinter"] = Path(self["hash"] + "_history.csv")

    def __init_hash_expe(self):
        lst_elem_to_remove_for_hash = [
            'identifier',
            '-v',
            '--help',
            "output_file_resprinter",
            "output_file_modelprinter",
            "output_file_notfinishedprinter",
            "output_file_csvcbprinter"
        ]

        keys_expe = sorted(self.keys())
        any(keys_expe.remove(item) for item in lst_elem_to_remove_for_hash if item in keys_expe)
        val_expe = [self[k] for k in keys_expe]
        str_expe = [str(val) for pair in zip(keys_expe, val_expe) for val in pair]
        self["hash"] = hex(zlib.crc32(str.encode("".join(str_expe))))

class ParameterManagerPalminizeFinetune(ParameterManagerPalminize):
    def __init__(self, dct_params, **kwargs):
        super().__init__(dct_params, **kwargs)
        self.__init_hash_expe()

        self.__init_seed()
        self["--input-dir"] = pathlib.Path(self["--input-dir"])
        self["--min-lr"] = float(self["--min-lr"]) if self["--min-lr"] is not None else None
        self["--max-lr"] = float(self["--max-lr"]) if self["--max-lr"] is not None else None
        self["--lr"] = float(self["--lr"]) if self["--lr"] is not None else None
        self["--nb-epoch"] = int(self["--nb-epoch"]) if self["--nb-epoch"] is not None else None
        self["--epoch-step-size"] = int(self["--epoch-step-size"]) if self["--epoch-step-size"] is not None else None

        if self["--use-clr"] not in ["triangular", "triangular2"] and self["--use-clr"] is not None:
            if type(self["--use-clr"]) is bool:
                pass
            else:
                raise ValueError(f"CLR policy should be triangular or triangular2. {self['--use-clr']}")

        if "--train-val-split" in self.keys() and self["--train-val-split"] is not None:
            self["--train-val-split"] = float(self["--train-val-split"]) if self["--train-val-split"] is not None else None
            assert 0 <= self["--train-val-split"] <= 1, f"Train-val split should be comprise between 0 and 1. {self['--train-val-split']}"

        self.__init_model_path()
        self.__init_output_file()

    def __init_seed(self):
        if not "--seed" in self.keys():
            self["--seed"] = np.random.randint(0, 2**32 - 2)

        if self["--seed"] is not None:
            self["--seed"] = int(self["--seed"])
            np.random.seed(self["--seed"])

    def __init_hash_expe(self):
        lst_elem_to_remove_for_hash = [
            'output_file_modelprinter',
            'identifier',
            'output_file_resprinter',
            '-v',
            '--help',
            '--input-dir',
        ]
        keys_expe = sorted(self.keys())
        any(keys_expe.remove(item) for item in lst_elem_to_remove_for_hash)
        val_expe = [self[k] for k in keys_expe]
        str_expe = [str(val) for pair in zip(keys_expe, val_expe) for val in pair]
        self["hash"] = hex(zlib.crc32(str.encode("".join(str_expe))))

    def __init_output_file(self):
        self["output_file_resprinter"] = Path(self["hash"] + "_results.csv")
        self["output_file_modelprinter"] = Path(self["hash"] + "_model.h5")
        self["output_file_notfinishedprinter"] = Path(self["hash"] + ".notfinished")
        self["output_file_finishedprinter"] = Path(self["hash"] + ".finished")
        self["output_file_tensorboardprinter"] = Path(self["hash"] + ".tb")
        self["output_file_csvcbprinter"] = Path(self["hash"] + "_history.csv")

    def __init_model_path(self):
        df = get_df(self["--input-dir"])
        keys_of_interest = ['--cifar10',
                            '--cifar10-vgg19',
                            '--cifar100',
                            '--cifar100-vgg19',
                            '--delta-threshold',
                            '--hierarchical',
                            '--mnist',
                            '--mnist-lenet',
                            '--nb-iteration-palm',
                            '--sparsity-factor',
                            '--svhn',
                            '--svhn-vgg19',
                            '--test-data',
                            '--test-model',
                            "--nb-factor"
                            ]
        if self["--cifar100-resnet50"] or self["--cifar100-resnet20"]:
            keys_of_interest.extend([
                '--cifar100-resnet50',
                '--cifar100-resnet20',
            ])
        # queries = []
        # for k in keys_of_interest:
        #     logger.debug("{}, {}, {}".format(self[k], type(self[k]), k))
        #     if self[k] is None:
        #         str_k = "'None'"
        #     else:
        #         str_k = self[k]
        #
        #     query = "(df['{}']=={})".format(k, str_k)
        #     queries.append(query)
        #
        # s_query = " & ".join(queries)
        # s_eval = "df[({})]".format(s_query)
        # line_of_interest = eval(s_eval)
        # logger.debug(line_of_interest)
        # logger.debug(s_eval)
        #
        # assert len(line_of_interest) == 1, "The parameters doesn't allow to discriminate only one pre-trained model in directory"
        line_of_interest = get_line_of_interest(df, keys_of_interest, self)
        self["input_model_path"] = self["--input-dir"] / line_of_interest["output_file_modelprinter"][0]


class ParameterManagerRandomSparseFacto(ParameterManager):
    def __init__(self, dct_params, **kwargs):
        super().__init__(dct_params, **kwargs)
        if self["--seed"] is not None:
            np.random.seed(int(self["--seed"]))

        self.__init_hash_expe()

        self["--walltime"] = int(self["--walltime"])
        self["--nb-factor"] = int(self["--nb-factor"]) if (self["--nb-factor"] != "None" and self["--nb-factor"] is not None) else None
        self["--sparsity-factor"] = int(self["--sparsity-factor"]) if self["--sparsity-factor"] is not None else None

        self.__init_output_file()

    def __init_hash_expe(self):
        lst_elem_to_remove_for_hash = [
            'output_file_modelprinter',
            'identifier',
            'output_file_resprinter',
            '-v',
            '--help',
            "--walltime",
        ]
        keys_expe = sorted(self.keys())
        any(keys_expe.remove(item) for item in lst_elem_to_remove_for_hash)
        val_expe = [self[k] for k in keys_expe]
        str_expe = [str(val) for pair in zip(keys_expe, val_expe) for val in pair]
        self["hash"] = hex(zlib.crc32(str.encode("".join(str_expe))))

    def __init_output_file(self):
        self["output_file_resprinter"] = Path(self["hash"] + "_results.csv")
        self["output_file_modelprinter"] = Path(self["hash"] + "_model.h5")
        self["output_file_notfinishedprinter"] = Path(self["hash"] + ".notfinished")
        self["output_file_finishedprinter"] = Path(self["hash"] + ".finished")
        self["output_file_tensorboardprinter"] = Path(self["hash"] + ".tb")
        self["output_file_csvcbprinter"] = Path(self["hash"] + "_history.csv")

class ParameterManagerEntropyRegularization(ParameterManager):
    def __init__(self, dct_params, **kwargs):
        super().__init__(dct_params, **kwargs)
        self.__init_hash_expe()

        self["--walltime"] = int(self["--walltime"])
        self["--nb-factor"] = int(self["--nb-factor"]) if (self["--nb-factor"] != "None" and self["--nb-factor"] is not None) else None
        self["--sparsity-factor"] = int(self["--sparsity-factor"]) if self["--sparsity-factor"] is not None else None
        self["--param-reg-softmax-entropy"] = float(self["--param-reg-softmax-entropy"]) if self["--param-reg-softmax-entropy"] is not None else None
        if self["--seed"] is not None:
            np.random.seed(int(self["--seed"]))
        self.__init_output_file()

    def __init_hash_expe(self):
        lst_elem_to_remove_for_hash = [
            'output_file_modelprinter',
            'identifier',
            'output_file_resprinter',
            '-v',
            '--help',
            "--walltime"
        ]
        keys_expe = sorted(self.keys())
        any(keys_expe.remove(item) for item in lst_elem_to_remove_for_hash)
        val_expe = [self[k] for k in keys_expe]
        str_expe = [str(val) for pair in zip(keys_expe, val_expe) for val in pair]
        self["hash"] = hex(zlib.crc32(str.encode("".join(str_expe))))

    def __init_output_file(self):
        self["output_file_resprinter"] = Path(self["hash"] + "_results.csv")
        self["output_file_modelprinter"] = Path(self["hash"] + "_model.h5")
        self["output_file_notfinishedprinter"] = Path(self["hash"] + ".notfinished")
        self["output_file_finishedprinter"] = Path(self["hash"] + ".finished")
        self["output_file_tensorboardprinter"] = Path(self["hash"] + ".tb")
        self["output_file_csvcbprinter"] = Path(self["hash"] + "_history.csv")

dict_cast_dtypes = {
    "object": str,
    "float64": float,
    "int64": int,
    "bool": bool
}

class ParameterManagerEntropyRegularizationFinetune(ParameterManagerEntropyRegularization):
    def __init__(self, dct_params, **kwargs):
        super().__init__(dct_params, **kwargs)
        self["--input-dir"] = pathlib.Path(self["--input-dir"])
        self["--permutation-threshold"] = float(self["--permutation-threshold"])
        self.__init_model_path()

    def __init_model_path(self):
        FORCE = False
        path_prepared_df = Path(self["--input-dir"]) / "prepared_df.csv"
        if not FORCE and path_prepared_df.exists():
            df = pd.read_csv(path_prepared_df, sep=";")
        else:
            df = get_df(self["--input-dir"])
            df.to_csv(str(path_prepared_df.absolute()), sep=";")

        keys_of_interest = ['--cifar10',
                            '--cifar10-vgg19',
                            '--cifar100',
                            '--cifar100-vgg19',
                            '--mnist',
                            '--mnist-lenet',
                            '--sparsity-factor',
                            '--svhn',
                            '--svhn-vgg19',
                            '--test-data',
                            '--test-model',
                            "--nb-factor",
                            "--nb-units-dense-layer",
                            "--param-reg-softmax-entropy",
                            "--pbp-dense-layers",
                            "--dense-layers",
                            "--seed"
                            ]
        # queries = []
        #
        # for k in keys_of_interest:
        #     logger.debug("{}, {}, {}".format(self[k], type(self[k]), k))
        #     key_type = df.dtypes[k].name
        #     if key_type == "object":
        #         str_k = "'{}'".format(self[k])
        #     else:
        #         str_k = "{}".format(self[k])
        #     # if self[k] is None:
        #     #     str_k = "'None'"
        #     # elif k in ["--nb-units-dense-layer", "--param-reg-softmax-entropy", "--nb-factor", "--sparsity-factor"]:
        #     #     str_k = "'{}'".format(self[k])
        #     # else:
        #     #     str_k = self[k]
        #
        #     query = "(df['{}']=={})".format(k, str_k)
        #     queries.append(query)
        #
        # s_query = " & ".join(queries)
        # s_eval = "df[({})]".format(s_query)
        # line_of_interest = eval(s_eval)
        # line_of_interest.drop_duplicates(keys_of_interest, inplace=True)
        # logger.debug(line_of_interest)
        # logger.debug(s_eval)
        #
        # assert len(line_of_interest) == 1, "The parameters doesn't allow to discriminate only one pre-trained model in directory"

        line_of_interest = get_line_of_interest(df, keys_of_interest, self)

        self["input_model_path"] = self["--input-dir"] / line_of_interest["output_file_modelprinter"].iloc[0]

def get_line_of_interest(df, keys_of_interest, dct_values):
    queries = []

    for k in keys_of_interest:
        logger.debug("{}, {}, {}".format(dct_values[k], type(dct_values[k]), k))
        try:
            key_type = df.dtypes[k].name

            if key_type == "object" or dct_values[k] is None or np.isnan(dct_values[k]):
                df[k] = df[k].astype(str)
                str_k = "'{}'".format(dct_values[k])
            else:
                str_k = "{}".format(dct_values[k])
        except KeyError:
            logger.warning("key {} not present in input palminized results".format(k) )
            keys_of_interest.remove(k)
            continue
        # if self[k] is None:
        #     str_k = "'None'"
        # elif k in ["--nb-units-dense-layer", "--param-reg-softmax-entropy", "--nb-factor", "--sparsity-factor"]:
        #     str_k = "'{}'".format(self[k])
        # else:
        #     str_k = self[k]

        query = "df_of_interest['{}']=={}".format(k, str_k)
        queries.append(query)

    # s_query = " & ".join(queries)
    df_of_interest = df
    for query in queries:
        s_eval = "df_of_interest[{}]".format(query)
        logger.debug(s_eval)
        df_of_interest = eval(s_eval)
        # try:
        assert not len(df_of_interest) < 1, "No corresponding pretrained model found in directory. Query {} discarded all".format(query)
        # except:
        #     pass

    line_of_interest = df_of_interest
    line_of_interest.drop_duplicates(keys_of_interest, inplace=True)
    logger.debug(line_of_interest)

    assert not len(line_of_interest) > 1, "The parameters doesn't allow to discriminate only one pre-trained model in directory. There are multiple"
    assert not len(line_of_interest) < 1, "No corresponding pretrained model found in directory"

    return line_of_interest

class ResultPrinter:
    """
    Class that handles 1-level dictionnaries and is able to print/write their values in a csv like format.
    """
    def __init__(self, *args, header=True, output_file=None):
        """
        :param args: the dictionnaries objects you want to print.
        :param header: tells if you want to print the header
        :param output_file: path to the outputfile. If None, no outputfile is written on ResultPrinter.print()
        """
        self.__dict = dict()
        self.__header = header
        self.__output_file = output_file

    def add(self, d):
        """
        Add dictionnary after initialisation.

        :param d: the dictionnary object you want to add.
        :return:
        """
        self.__dict.update(d)

    def _get_ordered_items(self):
        all_keys, all_values = zip(*self.__dict.items())
        arr_keys, arr_values = np.array(all_keys), np.array(all_values)
        indexes_sort = np.argsort(arr_keys)
        return list(arr_keys[indexes_sort]), list(arr_values[indexes_sort])

    def print(self):
        """
        Call this function whener you want to print/write to file the content of the dictionnaires.
        :return:
        """
        headers, values = self._get_ordered_items()
        headers = [str(h) for h in headers]
        s_headers = ",".join(headers)
        values = [str(v) for v in values]
        s_values = ",".join(values)
        if self.__header:
            print(s_headers)
        print(s_values)
        if self.__output_file is not None:
            with open(self.__output_file, "w+") as out_f:
                if self.__header:
                    out_f.write(s_headers + "\n")
                out_f.write(s_values + "\n")