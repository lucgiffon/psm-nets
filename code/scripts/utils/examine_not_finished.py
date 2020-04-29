"""
Find jobs which have a .notfinished file still existing and get their associated parameter line.

Usage:
    examine_not_finished.py --input-dir path [--output-dir path]

Options:
  -h --help                             Show this screen.
  --input-dir path                      Set the input dir to look experiments from.
  --output-dir path                     Set the output dir to save the experiments to if specified. Doesn't write anything by default.
"""
from docopt import docopt
from pathlib import Path
import os
import re
import pandas as pd

if __name__ == "__main__":
    paraman = docopt(__doc__)
    path_input_dir = Path(paraman["--input-dir"])

    regex_results_csv = re.compile(r'.+results.csv')
    regex_command_line = re.compile(r'INFO\s+root: Command line: .+?\.py (.+)\n')

    lst_params_to_resub = list()


    for root, dir, files in os.walk(path_input_dir):
        root_path = Path(root)
        for file in files:
            # read only csv results files
            if not regex_results_csv.match(file):
                continue

            path_file = root_path / file
            df_results = pd.read_csv(path_file)

            file_not_finished = df_results["output_file_notfinishedprinter"].values[0]
            path_not_finished = root_path / file_not_finished

            if not path_not_finished.exists():
                continue  # then the job has properly finished

            for col in df_results.columns:
                print(str(col), df_results[col].values[0])
            print(path_not_finished)
            OAR_identifier = df_results["identifier"].values[0]
            print("OAR_identifier", OAR_identifier)
            stderr_name = f"OAR.{OAR_identifier}.stderr"
            path_stderr = root_path / stderr_name
            assert path_stderr.exists(), "no OAR stderr file exists for that experiment"

            # look for the command line used for the experiment that did not finish
            with open(path_stderr, 'r') as std_errfile:
                lines = std_errfile.readlines()
                # just look at the obtained error
                for lin in reversed(lines):
                    striped = lin.strip()
                    if striped == "":
                        continue
                    print(striped)
                    break
                # get the parameters for this command line
                for lin in lines:
                    match = regex_command_line.search(lin)
                    if match:
                        lst_params_to_resub.append(match.group(1))
                        print(match.group(1))
                        break

    if paraman["--output-dir"] is not None:
        path_output_dir = Path(paraman["--output-dir"])
        output_file_name = f"{path_input_dir.name}_errors.txt"
        path_output_file = path_output_dir / output_file_name
        print()
        with open(path_output_file, 'w') as ofile_w:
            for param in lst_params_to_resub:
                print(param)
                ofile_w.write(f'{param}\n')