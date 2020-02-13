import time
from tempfile import TemporaryDirectory
from asynchronous import Experiment, Conf
import pickle


if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser()

    parser.add_argument(
        'path', metavar="PATH",
        type=str,
        action='store',
        help="Path to the checkpoint."
    )
    parser.add_argument(
        'test_conf_path', metavar="TEST_CONF_PATH",
        type=str,
        action='store',
        help="Path to the test config file."
    )

    args = parser.parse_args()
    experiment_dir = args.path

    n_parameter_servers = 1
    n_workers = 4

    with open(experiment_dir + "/../../helper_classes/worker_conf.pkl", "rb") as f:
        worker_conf = pickle.load(f)

    with TemporaryDirectory() as d:
        with Experiment(n_parameter_servers, n_workers, d + "/dummy/", worker_conf, worker0_display=False) as exp:
            exp.restore_model(experiment_dir)
            exp.test(args.test_conf_path, outpath=args.path + "/../../test_data/")
