import sys
import re
import os
import pathlib
from pprint import pprint
from shutil import copyfile

keras_progress_bar_re = r"\d+/\d+ \[=*>\.\.+\] - ETA:.+?\n"
stdout_re = r".+\.stdout"
stderr_re = r".+\.stderr"

if __name__ == "__main__":
    dir_path = sys.argv[1].rstrip("/")
    last_dir = dir_path.split("/")[-1]
    # out_dir_path = last_dir + "_after_reduce"
    regex_progress_bar = re.compile(keras_progress_bar_re)
    regex_stdout = re.compile(stdout_re)
    regex_stderr = re.compile(stderr_re)
    for root, dirs, files in os.walk(dir_path):
        print("Exploring directory {}".format(root))
        root_path = pathlib.Path(root)
        for file in files:
            if regex_stdout.match(file):
                print("Stdout file found: {}".format(file))
                with open(root_path / file, 'r') as f:
                    str_file = f.read()
                new_str = regex_progress_bar.sub('', str_file)
                with open(root_path/file, 'w') as f:
                    f.write(new_str)