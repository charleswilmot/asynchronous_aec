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

    args = parser.parse_args()
    experiment_dir = args.path

    n_parameter_servers = 1
    n_workers = 1
    # experiment_dir = "../experiments/2019_06_12-12:20:04_mlr1.00e-04_clr1.00e-04_alr1.00e-05_entropy2.00e-01__tune_temperature_1.0/checkpoints/01000000/"

    with open(experiment_dir + "/../../conf/worker_conf.pkl", "rb") as f:
        worker_conf = pickle.load(f)

    with TemporaryDirectory() as d:
        with Experiment(n_parameter_servers, n_workers, d + "/dummy/", worker_conf, worker0_display=True) as exp:
            exp.restore_model(experiment_dir)
            exp.playback(50, True)
            exp.playback(10, False)
