"""
Find Jobd Killed prematurely and print info about them

Usage:
    resurector.py [--input-dir path] [--output-dir path]

Options:
  -h --help                             Show this screen.
  -vv                                   Set verbosity to debug.
  -v                                    Set verbosity to info.
  --input-dir path                      Set the input dir to look experiments from.
  --output-dir path                     Set the output dir to save the experiments to if specified. Doesn't write anything by default.
"""

import pathlib
import docopt
from subprocess import check_output

import re
import os
from pprint import pprint
import yaml

# regex_killed = re.compile(r"##.+? (\d+) KILLED ##")
oarstat_command = "oarstat -f -j {} -Y"

regex_command_line = re.compile(r'.+?\.py (.+)')

def main():
    paraman = docopt.docopt(__doc__)

    if paraman["--input-dir"] is None:
        input_dir_path = pathlib.Path(os.getcwd())
    else:
        input_dir_path = pathlib.Path(paraman["--input-dir"])

    lst_param_to_relaunch = list()

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
                match = regex_command_line.match(command)
                print(match.group(1))
                lst_param_to_relaunch.append(match.group(1))

    if paraman["--output-dir"] is not None:
        path_output_dir = pathlib.Path(paraman["--output-dir"])
        output_file_name = f"{input_dir_path.name}_killed.txt"
        path_output_file = path_output_dir / output_file_name
        print()
        with open(path_output_file, 'w') as ofile_w:
            for param in lst_param_to_relaunch:
                print(param)
                ofile_w.write(f'{param}\n')


if __name__ == "__main__":
    main()