"""
Find Jobd Killed prematurely and print info about them

Usage:
    resurector.py [--input-dir path]

Options:
  -h --help                             Show this screen.
  -vv                                   Set verbosity to debug.
  -v                                    Set verbosity to info.
  --input-dir path                      Set the input dir to look experiments from.
"""

import pathlib
import docopt
from subprocess import check_output

import re
import os
from pprint import pprint
import yaml

regex_killed = re.compile(r"##.+? (\d+) KILLED ##")
oarstat_command = "oarstat -f -j {} -Y"

def main():
    dct_params = docopt.docopt(__doc__)

    if dct_params["--input-dir"] is None:
        input_dir_path = pathlib.Path(os.getcwd())
    else:
        input_dir_path = pathlib.Path(dct_params["--input-dir"])

    for root, dir, files in os.walk(input_dir_path):
        root_path_to_file = pathlib.Path(root)
        for file in files:
            if not (file.split(".")[-1] == "stderr"):
                # if this is not an OAR stderr file, pass
                continue

            path_to_file = root_path_to_file / file

            file_job_id = file.split(".")[-2]
            oarstat_command_job = oarstat_command.format(file_job_id)
            output_oarstat = yaml.safe_load(check_output(oarstat_command_job.split()).decode("utf-8"))
            job_output = output_oarstat[int(file_job_id)]
            exit_code = job_output["exit_code"]
            command = job_output["command"]
            if exit_code != 0:  # error or interruption by OAR
                print(file, exit_code, command)

            #
            # with open(path_to_file, 'r') as of:
            #     str_of = of.read()
            # search = regex_killed.search(str_of)
            # if search is not None:
            #     pprint(job_output)
if __name__ == "__main__":
    main()