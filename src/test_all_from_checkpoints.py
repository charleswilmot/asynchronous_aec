import time
from tempfile import TemporaryDirectory
from algorithm.experiment import Experiment
from algorithm.conf import Conf
import pickle
import os


if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser()

    parser.add_argument(
        'path', metavar="PATH",
        type=str,
        action='store',
        help="Path to the checkpoint dir."
    )
    parser.add_argument(
        'test_conf_path', metavar="TEST_CONF_PATH",
        type=str,
        action='store',
        help="Path to the test config file."
    )

    args = parser.parse_args()

    n_parameter_servers = 1
    n_workers = 1

    with open(args.path + "/../conf/worker_conf.pkl", "rb") as f:
        worker_conf = pickle.load(f)

    with TemporaryDirectory() as d:
        with Experiment(n_parameter_servers, n_workers, d + "/dummy/", worker_conf, worker0_display=False) as exp:
            for checkpoint in os.listdir(args.path):
                exp.restore_model(args.path + '/' + checkpoint)
                exp.test(args.test_conf_path, outpath=args.path + "/../test_data/")
