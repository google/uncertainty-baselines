import os
import pdb
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from exps.deploy.manual import utils
from absl import app, flags


flags.DEFINE_string(
    "run_script",
    os.path.join(utils.PROJECT_ROOT, "exps/deploy/manual/run_file.py"),
    "Path to the run_script",
)
flags.DEFINE_string(
    "job_folder",
    os.path.join(utils.PROJECT_ROOT, "exps/random/jobs"),
    "Path to the run_script",
)

FLAGS = flags.FLAGS


def main(argv):
    del argv
    vms = [f"qf-{i}" for i in range(1, 9)]

    vm_infos = {vm: utils.get_info_of_vm(vm) for vm in vms}

    # synchronise code
    for vm in tqdm(vms, desc="synchronise code.."):
        utils.rsync_code(ip=vm_infos[vm]["ip"], make_allow_ssh=True)

    # launch script
    for i, vm in tqdm(enumerate(vms), desc="launch run script..."):
        cmd = f"python {FLAGS.run_script} --file_path {os.path.join(FLAGS.job_folder, str(i + 1))}"
        ssh_cmd = utils.ssh_remote_cmd(cmd, ip=vm_infos[vm]["ip"])
        pdb.set_trace()
        # utils.run_command(ssh_cmd.split())


if __name__ == "__main__":
    app.run(main)
