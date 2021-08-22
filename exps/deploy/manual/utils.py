"""
1. create machines using the same machine image
2. generate configs, save them as commands in files, one file for each machine
3. rsync code to machines
4. manually run the right file in each machine

"""
import json
import os
import pdb
import stat
import subprocess
import threading
from typing import Dict, List

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def _to_str(v):
    if isinstance(v, list):
        v = ",".join([str(x).strip() for x in v])
    else:
        v = str(v)
    return v


def config_to_command(config: Dict, prefix: str) -> str:
    strings = [prefix]
    for k, v in config.items():
        strings.append(f"--{k}={_to_str(v)}")
    return " ".join(strings)


def generate_files(configs: List[Dict], n_files: int, folder_path: str, prefix: str):
    n = len(configs)
    cmds = [config_to_command(c, prefix=prefix) for c in configs]
    n_configs_per_file = int(np.ceil(n / n_files))
    for i in range(n_files):
        cmds_i = cmds[i * n_configs_per_file : (i + 1) * n_configs_per_file]
        file_path = os.path.join(folder_path, f"{i + 1}.sh")
        generate_bash_file(file_path, cmds_i)


def generate_bash_file(file_path: str, cmd_strs: List[str], mode: str = "w"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode) as f:
        for s in cmd_strs:
            f.write(s + "\n")
    make_executable(file_path)


def make_executable(file_path: str):
    st = os.stat(file_path)
    os.chmod(file_path, st.st_mode | stat.S_IEXEC)


def print_output(std):
    while True:
        try:
            string = next(std)
            string.rstrip("\n")
            print(string)
        except StopIteration:
            return


class printThread(threading.Thread):
    def __init__(self, std):
        threading.Thread.__init__(self)
        self.std = std

    def run(self):
        print_output(self.std)


def run_command(cmds: List[str], buffer=False):
    print(f"Running command:\n {''.join(cmds)}")
    p = subprocess.Popen(
        cmds,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        bufsize=1,
    )

    if buffer:
        stdout_iter = iter(p.stdout.readline, "")
        stderr_iter = iter(p.stderr.readline, "")
        threads = [printThread(stdout_iter), printThread(stderr_iter)]
        for t in threads:
            t.start()

        for t in threads:
            t.join()
        print("No output anymore, waiting for program to finish...")
        p.stdout.close()
        p.stderr.close()

    stdout, stderr = p.communicate()
    return_code = p.returncode
    if return_code != 0:
        print(f"failed, return code = {p.returncode}")
    else:
        print("success")
    return stdout, stderr, return_code


def robust_run_command(cmd: str):
    rerun_cond = lambda s: "CUDA_ERROR_ILLEGAL_ADDRESS" in s
    while True:
        stdout, stderr, return_code = run_command(cmd.split())
        if return_code == 0:
            break
        print(
            f"Failed with the following reason\nstdout=\n\n{stdout}\n\n\nstderr=\n\n{stderr}"
        )
        if not rerun_cond(stdout) and not rerun_cond(stderr):
            print("Give up")
            break
        print("Rerunning")


def run_config_file(file_path):
    with open(file_path, "r") as f:
        cmd = f.readline()
        print(f"Running command {cmd}...")
        robust_run_command(cmd)


def get_info_of_vm(name):
    cmd = f"gcloud compute instances list --format=json {name}"
    output = subprocess.check_output(cmd, shell=True)
    info = json.loads(output)
    assert len(info) == 1, f"{info}"
    info = info[0]
    return {
        "ip": info["networkInterfaces"][0]["accessConfigs"][0]["natIP"],
        "zone": info["zone"].split("/")[-1],
        "name": name,
    }


def ssh_remote_cmd(cmd, ip, private_key_file="~/.ssh/google_compute_engine"):
    return f"ssh -o 'IdentitiesOnly=yes' -i {private_key_file} {ip} {cmd}"


def allow_ssh():
    run_command(
        "gcloud compute firewall-rules create default-allow-ssh --allow tcp:22".split()
    )


def rsync_code(
    vm_name=None,
    ip=None,
    local_path=PROJECT_ROOT,
    remote_path="/home/qfeng/workspace",
    make_allow_ssh=True,
):
    if make_allow_ssh:
        allow_ssh()
    if ip is None:
        ip = get_info_of_vm(vm_name)["ip"]
    remote_path = f"{ip}:{remote_path}"
    cmd = f"""
        rsync -avhud -zz --copy-links --delete
        --filter 'protect *.egg-info'
        -e "ssh -o 'IdentitiesOnly=yes' -i ~/.ssh/google_compute_engine"
        -r {local_path} {remote_path}
    """
    cmd = cmd.strip().strip("\n").replace("\\", "").replace("\n", "")
    print(f"running command: {cmd}")
    print(subprocess.check_output(cmd, shell=True).decode())
