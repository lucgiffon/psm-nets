import re
import subprocess

regex_job_1000_1000 = r"Job_Id: (\d+).+?command = .+? --nb-units-dense-layer 1000-1000"
regex_obj = re.compile(regex_job_1000_1000, re.DOTALL)
if __name__ == "__main__":
    with open("jobs_luc.txt", 'r') as f:
        str_file = f.read()
        search = regex_obj.findall(str_file)
        del_str_pattern = "oardel {}"
        for s in search:
            del_str = del_str_pattern.format(s)
            print(del_str.split())
            subprocess.run(del_str.split())

