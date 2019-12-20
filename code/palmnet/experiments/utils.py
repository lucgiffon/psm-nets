import os
import zlib

import pathlib
import random

import numpy as np
from pathlib import Path

import time

from palmnet.data import Mnist, Cifar10, Cifar100, Svhn, Test
from palmnet.utils import get_df

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
        self["--seed"] = int(self["--seed"])
        if self["--seed"] is not None:
            np.random.seed(self["--seed"])
        else:
            self["--seed"] = int(self["--seed"])

    def get_dataset(self):
        """
        Return dataset in shape n x d.

        n: number of observations.
        d: dimensionality of observations.

        :return:
        """
        if self["--mnist"]:
            (x_train, y_train), (x_test, y_test) =  Mnist.load_data()
            return (x_train, y_train), (x_test, y_test)
        elif self["--cifar10"]:
            (x_train, y_train), (x_test, y_test) = Cifar10.load_data()
            return (x_train, y_train), (x_test, y_test)
        elif self["--cifar100"]:
            (x_train, y_train), (x_test, y_test) = Cifar100.load_data()
            return (x_train, y_train), (x_test, y_test)
        elif self["--svhn"]:
            (x_train, y_train), (x_test, y_test) = Svhn.load_data()
            return (x_train, y_train), (x_test, y_test)
        elif self["--test-data"]:
            (x_train, y_train), (x_test, y_test) = Test.load_data()
            return (x_train, y_train), (x_test, y_test)

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
        else:
            raise NotImplementedError("No dataset specified.")

class ParameterManagerPalminize(ParameterManager):
    def __init__(self, dct_params, **kwargs):
        super().__init__(self, **dct_params, **kwargs)
        self["--sparsity-factor"] = int(self["--sparsity-factor"]) if self["--sparsity-factor"] is not None else None
        self["--nb-iteration-palm"] = int(self["--nb-iteration-palm"]) if self["--nb-iteration-palm"] is not None else None
        self["--delta-threshold"] = float(self["--delta-threshold"]) if self["--delta-threshold"] is not None else None

class ParameterManagerPalminizeFinetune(ParameterManagerPalminize):
    def __init__(self, dct_params, **kwargs):
        super().__init__(dct_params, **kwargs)
        self.__init_hash_expe()

        self["--input-dir"] = pathlib.Path(self["--input-dir"])
        self["--walltime"] = int(self["--walltime"])

        self.__init_model_path()
        self.__init_output_file()

    def __init_hash_expe(self):
        lst_elem_to_remove_for_hash = [
            'output_file_modelprinter',
            'identifier',
            'output_file_resprinter',
            '-v',
            '--help',
            '--input-dir',
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
                            ]
        queries = []
        for k in keys_of_interest:
            query = "(df['{}']=={})".format(k, self[k])
            queries.append(query)

        s_query = " & ".join(queries)
        s_eval = "df[({})]".format(s_query)
        line_of_interest = eval(s_eval)

        assert len(line_of_interest) == 1, "The parameters doesn't allow to discriminate only one pre-trained model in directory"

        self["input_model_path"] = self["--input-dir"] / line_of_interest["output_file_modelprinter"][0]


class ParameterManagerRandomSparseFacto(ParameterManager):
    def __init__(self, dct_params, **kwargs):
        super().__init__(dct_params, **kwargs)
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