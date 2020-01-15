import pickle

import pathlib

import re
from io import StringIO
import pandas as pd
from skluc.utils import logger

output_file_end_re = {
    "results": r"_results.csv",
    "palminized_model": r"_model_layers.pckle"
}

def display_cmd_lines_from_root_name_list(root_names_list, src_results_dir, find_in_stderr=False):
    cmd_lines = []

    if not find_in_stderr:
        for root_name in root_names_list:
            stdout_file = src_results_dir / (root_name + ".stdout")
            with open(stdout_file, 'r') as stdoutfile:
                lines = stdoutfile.readlines()
                i_line = -1
                for i_line, lin in enumerate(lines):
                    if lin[:2] == "--":
                        break
                if i_line == -1:
                    continue
                data = "".join(lines[i_line:i_line + 2])

                io_data = StringIO(data)
                df = pd.read_csv(io_data)

            # print("".join(lines))

        # print("\n")

            cmd_line = ""
            cmd_line += "qmeans" if df["qmeans"][0] else "kmeans"
            cmd_line += " --sparsity-factor" + " " + str(df["--sparsity-factor"][0])
            cmd_line += " --seed" + " " + str(df["--seed"][0])
            cmd_line += " --nystrom" if df["--nystrom"][0] else ""
            cmd_line += " --assignation-time" if df["--assignation-time"][0] else ""
            cmd_line += " --1-nn" if df["--1-nn"][0] else ""
            cmd_line += " --initialization" + " " + str(df["--initialization"][0])
            cmd_line += " --nb-cluster" + " " + str(df["--nb-cluster"][0])
            cmd_line += " --blobs" if df["--blobs"][0] else ""
            cmd_line += " --census" if df["--census"][0] else ""
            cmd_line += " --kddcup" if df["--kddcup"][0] else ""
            cmd_line += " --mnist" if df["--mnist"][0] else ""
            cmd_line += " --fashion-mnist" if df["--fashion-mnist"][0] else ""

            cmd_lines.append(cmd_line)

    else:
        regex_cmd_line = re.compile(r".+Command line: (.+)")
        for root_name in root_names_list:
            stderr_file = src_results_dir / (root_name + ".stderr")
            with open(stderr_file, 'r') as stderrfile:
                lines = stderrfile.readlines()
                for lin in lines:
                    match = regex_cmd_line.match(lin)
                    if match:
                        cmd_lines.append(" ".join(match.group(1).split()[1:]) + "\t" + root_name) # with root name
                        # cmd_lines.append(" ".join(match.group(1).split()[1:])) # without root name
                        break

            # print("".join(lines))
            # print("\n")
    return cmd_lines


def get_dct_result_files_by_root(src_results_dir, old_filename_objective=False):
    """
    From a directory with the result of oarjobs give the dictionnary of results file for each experiment. Files are:

    * OAR.`jobid`.stderr
    * OAR.`jobid`.stdout

    * `idexpe`_model_layers.pckle contains the palminized model;
    * `idexpe`_results.csv contains the parameters of the experiments and the various metric measures;

    where:

     * `jobid` correspond to oar's own job identifier;
     * `nameobjective` correspond to the name of the objective function being printed;
     * `idexpe` correspond to the name

    The returned dictionnary gives:

    {
        "OAR.`jobid`": {
            "palminized_model": "`idexpe`_model_layers.pckle",
            "results": "`idexpe`_results.csv",
        }
    }

    :param src_results_dir: path to
    :return:
    """
    files = src_results_dir.glob('**/*')
    files = [x for x in files if x.is_file()]
    lst_str_filenames = [file.name for file in files]

    dct_output_files_by_root = {}
    count_complete = 0
    count_has_printed_results = 0
    count_total = 0

    for pth_file in files:
        if pth_file.suffix != '.stdout':
            continue

        count_total += 1

        with open(pth_file, 'r') as stdoutfile:
            lines = stdoutfile.readlines()
            for i_line, lin in enumerate(lines):
                if lin[:2] == "--":
                    break
            else:
                logger.warning("file {} didn't contain anything".format(pth_file.name))
                dct_output_files_by_root[pth_file.stem] = {}
                continue
            count_has_printed_results += 1

            data = "".join(lines[i_line:i_line + 2])

        io_data = StringIO(data)
        df = pd.read_csv(io_data)

        try:
            root_name = df["output_file_resprinter"][0].split("_")[0]
        except KeyError:
            logger.warning("no key for resprinter in {}".format(pth_file.name))

        dct_files = {}
        complete = True

        used_output_file_end_re = output_file_end_re

        for type_file, root_re in used_output_file_end_re.items():
            forged_re_compiled = re.compile(r"{}".format(root_name) + root_re)
            try:
                dct_files[type_file] = list(filter(forged_re_compiled.match, lst_str_filenames))[0]
            except IndexError:
                logger.warning("{} not found for root name {}".format(type_file, root_name))
                complete = False

        if complete:
            count_complete += 1

        dct_output_files_by_root[pth_file.stem] = dct_files

    return dct_output_files_by_root

def build_df(path_results_dir, dct_output_files_by_root, col_to_delete=[]):
    lst_df_results = []
    for root_name, dct_results in dct_output_files_by_root.items():
        try:
            result_file = path_results_dir / dct_results["results"]
            df_expe = pd.read_csv(result_file)
            df_expe["oar_id"] = root_name
            lst_df_results.append(df_expe)
        except KeyError:
            logger.warning("No 'results' entry for root name {}".format(root_name))
        except FileNotFoundError:
            logger.warning("File '{}' not found".format(root_name))

    df_results = pd.concat(lst_df_results)

    for c in col_to_delete:
        df_results = df_results.drop([c], axis=1)
    return df_results


def get_palminized_model_and_df(path):
    src_result_dir = pathlib.Path(path)
    dct_output_files_by_root = get_dct_result_files_by_root(src_results_dir=src_result_dir, old_filename_objective=True)

    col_to_delete = []

    dct_oarid_palminized_model = {}
    for root_name, job_files in dct_output_files_by_root.items():
        objective_file_path = src_result_dir / job_files["palminized_model"]
        loaded_model = pickle.load(open(objective_file_path, 'rb'))
        dct_oarid_palminized_model[root_name] = loaded_model

    df_results = build_df(src_result_dir, dct_output_files_by_root, col_to_delete)
    return dct_oarid_palminized_model, df_results


def get_df(path):
    src_result_dir = pathlib.Path(path)
    dct_output_files_by_root = get_dct_result_files_by_root(src_results_dir=src_result_dir, old_filename_objective=True)

    col_to_delete = []

    df_results = build_df(src_result_dir, dct_output_files_by_root, col_to_delete)
    return df_results