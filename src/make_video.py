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
        '-n', "--name",
        type=str,
        action='store',
        default="default_name",
        help="Path to the checkpoint."
    )

    args = parser.parse_args()
    experiment_dir = args.path

    n_parameter_servers = 1
    n_workers = 1

    with open(experiment_dir + "/../../conf/worker_conf.pkl", "rb") as f:
        worker_conf = pickle.load(f)

    with TemporaryDirectory() as d:
        with Experiment(n_parameter_servers, n_workers, d + "/dummy/", worker_conf, worker0_display=False) as exp:
            exp.restore_model(experiment_dir)
            exp.save_video(args.name, 50, outpath=args.path + "/../../video/")
