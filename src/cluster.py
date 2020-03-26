import numpy as np
import time
from train import make_experiment_path
import os
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
    """Initiate folders, path, parameters for execution on cluster"""
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
            logging.debug("Adding command: {}".format(arg))
            self.cmd += arg
        # FIXIT: Move this outside of the class
        self.cmd += " -LXserver"
        if "description" in cluster_params:
            self.cmd += " --job-name {}".format(cluster_params["description"].replace(" ", "_"))
        print("\n")
        self.cmd += " cluster.sh"
        for k, v in algo_params.items():
            flag = self._key_to_flag(k)
            arg = self._to_arg(flag, v)
            logging.debug("Adding command: {}".format(arg))
            self.cmd += arg
        self.cmd += self._to_arg("--experiment-path", experiment_path)
        # Print command line command
        logging.debug(self.cmd)

    def _key_to_flag(self, key):
        return "--" + str(key).replace("_", "-")

    def _to_arg(self, flag, v):
        return " {} {}".format(flag, str(v))

    def run(self):
        print("\n", "[+] Launching: ", self.cmd, "\n")
        os.system(self.cmd)
        time.sleep(10)


# General parameters
description = "pan_tilt_vergence_2_frames_reward_scaling_factor_600_net_dim_200_200_after_fix_typo_batch_norm_0.99_low_compression"

# Define cluster specs here
cluster_params = {
    "partition": "sleuths",
    "gres": 'gpu:1',
    "mincpus": 46, #40
    "mem": 90000, #90_000
    "description": description,
    "reservation": "triesch-shared"
}

# Define algorithm specs here
algo_params = {
    "n_episodes": 400000, #200_000
    "n_workers": 40, #40
    "description": description,
    "critic_learning_rate": 5e-5,
    "model_learning_rate": 5e-5,
    "discount_factor": 0.001,
    "episode_length": 10,
    "reward_scaling_factor": 600,
    "epsilon": 0.05,
    "batch_size": 200,
    # "restore_from": "../experiments/2020_03_25-16.32.32_mlr5.00e-05_clr5.00e-05__pan_tilt_vergence_2_frames_reward_scaling_factor_600_net_dim_200_200_after_fix_typo_batch_norm_0.99_low_compression/checkpoints/00100238/"
}

cq = ClusterQueue(algo_params, cluster_params)
cq.run()
