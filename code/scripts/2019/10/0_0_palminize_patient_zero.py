"""
This script is the experiment script for palminizing models.

Usage:
    script.py [-h] [-v|-vv] (--mnist|--svhn|--cifar10|--cifar100) --sparsity-factor=int [--nb-iteration-palm=int] [--delta-threshold=float]

Options:
  -h --help                             Show this screen.
  -vv                                   Set verbosity to debug.
  -v                                    Set verbosity to info.

Dataset:
  --mnist                               Use blobs dataset from sklearn. Formatting is size-dimension-nbcluster
  --svhn                                Use blobs dataset from sklearn with few data for testing purposes.
  --cifar10                             Use blobs dataset from sklearn with few data for testing purposes.
  --cifar100                            Use blobs dataset from sklearn with few data for testing purposes.

Palm-Specifc options:
  --sparsity-factor=int                 Integer coefficient from which is computed the number of value in each factor.
  --nb-iteration-palm=int               Number of iterations in the inner palm4msa calls. [default: 300]
  --delta-threshold=float               Threshold value before stopping palm iterations. [default: 1e-6]
"""
import logging
import sys
import time

import docopt

from palmnet.core.palminize import Palminizer, Palminizable
from palmnet.utils import ParameterManager, ResultPrinter
from skluc.utils import logger, log_memory_usage

lst_results_header = [
    "palminization_time",
    "nb_param_model_layers_conv_dense",
]

def main():
    base_model = paraman.get_model()
    palminizer = Palminizer(sparsity_fac=paraman["--sparsity-factor"],
                            nb_iter=paraman["--nb-iteration-palm"],
                            delta_threshold_palm=paraman["--delta-threshold"])
    palminizable = Palminizable(base_model, palminizer)
    start_palminize = time.time()
    palminizable.palminize()
    stop_palminize = time.time()

    #
    # todo compute nb param in base model and palminized model
    # todo compute nb flops for one example pass in base model and palminized model
    # todo verifier test accuracy
    # todo print model


if __name__ == "__main__":
    logger.info("Command line: " + " ".join(sys.argv))
    log_memory_usage("Memory at startup")
    arguments = docopt.docopt(__doc__)
    paraman = ParameterManager(arguments)
    initialized_results = dict((v, None) for v in lst_results_header)
    resprinter = ResultPrinter(output_file=paraman["output_file_resprinter"])
    resprinter.add(initialized_results)
    resprinter.add(paraman)
    if paraman["-v"] >= 2:
        logger.setLevel(level=logging.DEBUG)
    elif paraman["-v"] >= 1:
        logger.setLevel(level=logging.INFO)
    else:
        logger.setLevel(level=logging.WARNING)

    logger.warning("Verbosity set to warning")
    logger.info("Verbosity set to info")
    logger.debug("Verbosity set to debug")

    has_failed = False
    try:
        main()
    except Exception as e:
        has_failed = True
        raise e

    finally:
        failure_dict = {
            "failure": has_failed
        }

        resprinter.add(failure_dict)
        resprinter.print()