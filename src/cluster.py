import numpy as np
import time
from asynchronous import make_experiment_path
import os

possible_values = \
    ["n_parameter_servers",
     "n_workers",
     "description",
     "tensorboard",
     "flush_every",
     "n_trajectories",
     "episode_length",
     "update_per_episode",
     "restore_from",
     "model_learning_rate",
     "critic_learning_rate",
     "actor_learning_rate",
     "entropy_reg",
     "entropy_reg_decay"]

class ClusterQueue:

    def __init__(self, algo_params, cluster_params):
        self.algo_params = algo_params
        self.cluster_params = cluster_params
        # Make experiment path and create folders
        experiment_path = make_experiment_path(
            mlr=algo_params["model_learning_rate"] if "model_learning_rate" in algo_params else None,
            clr=algo_params["critic_learning_rate"] if "critic_learning_rate" in algo_params else None,
            description=algo_params["description"].replace(" ", "_"))
        print(experiment_path)
        os.mkdir(experiment_path)
        os.mkdir(experiment_path + "/log")
        # Build console command for cluster execution
        print("\n")
        self.cmd = "sbatch --output {}/log/%N_%j.log".format(experiment_path)
        for k, v in cluster_params.items():
            if k in ["description"]:
                continue
            flag = self._key_to_flag(k)
            arg = self._to_arg(flag, v)
            print("[DEBUG] Adding command: {}".format(arg))
            self.cmd += arg
        # TODO: Move this outside of the class
        self.cmd += " -LXserver"
        if "description" in cluster_params:
            self.cmd += " --job-name {}".format(cluster_params["description"].replace(" ", "_"))
        print("\n")
        self.cmd += " cluster.sh"
        for k, v in algo_params.items():
            flag = self._key_to_flag(k)
            arg = self._to_arg(flag, v)
            print("[DEBUG] Adding command: {}".format(arg))
            self.cmd += arg
        self.cmd += self._to_arg("--experiment-path", experiment_path)
        # Print command line command
        print("\n", "[DEBUG] ", self.cmd, "\n")

    def _key_to_flag(self, key):
        return "--" + str(key).replace("_", "-")

    def _to_arg(self, flag, v):
        return " {} {}".format(flag, str(v).replace(" ", "_"))

    def run(self):
        print("\n", "[+] Launching: ", self.cmd, "\n")
        os.system(self.cmd)
        time.sleep(10)

# General parameters
description = "Basic Description With Spaces"

# Define cluster specs here
cluster_params = {
    "partition":"sleuths",
    "gres":'gpu:3',
    "mincpus":40,
    "mem":90000,
    "description":description
}
# Define algorithm specs here
algo_params = {
    "n_episodes":20000,
    "flush_every":5000,
    "n_workers":40,
    "description":description,
    "critic_learning_rate":1e-4,
    "model_learning_rate":1e-4,
    "discount_factor":0.0,
    "update_factor":10,
    "episode_length":10,
    "epsilon":0.2,
}

cq = ClusterQueue(algo_params, cluster_params)
cq.run()
