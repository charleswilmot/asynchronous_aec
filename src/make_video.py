import time
from tempfile import TemporaryDirectory
from algorithm.experiment import Experiment
from algorithm.conf import Conf
from helper.generate_test_conf import TestConf, test_case
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
        '-tcp', "--test_conf_path",
        type=str,
        action='store',
        default=None,
        help="Path to the test config file."
    )
    parser.add_argument(
        '-n', "--name",
        type=str,
        action='store',
        default="default_name",
        help="Name of the video."
    )

    args = parser.parse_args()
    experiment_dir = args.path
    test_conf_path = args.test_conf_path

    n_parameter_servers = 1
    n_workers = 1

    test_cases = None
    "(Stimulus ID, Initial depth position, Initial vergence error, Initial speed error;Pan/Tilt speed, Depth speed, Episode iterations)"
    # test_cases = [
    #     test_case(0, 2, 0, [0., 0.], -0.03, 20),
    #     test_case(0, 3, 0, [0., 0.], -0.00, 20),
    #     test_case(4, 2.5, 0, [0., 0.], 0.03, 20),
    #     test_case(0, 4, 0, [1., 0.], -0.01, 20),
    #     test_case(6, 5, 0, [0., 1.], -0.01, 20),
    # ]

    with open(experiment_dir + "/../../conf/worker_conf.pkl", "rb") as f:
        worker_conf = pickle.load(f)

    with TemporaryDirectory() as d:
        with Experiment(n_parameter_servers, n_workers, d + "/dummy/", worker_conf, test_conf_path=args.test_conf_path,
                        worker0_display=False) as exp:
            exp.restore_all(experiment_dir)
            # Makes a normal video based on the experiment parameters if no test conf file if given via command line or
            # makes a video of specific test cases if test conf file is given via command line. If a test conf file is
            # given, test_conf.data["test_cases_policy_dependent"] test cases are used (with a limit of 100, can be
            # ajusted) if test_cases in this file is None. If test_cases in this file is NOT None, the test cases of
            # this file are being used.
            if test_conf_path:
                exp.make_video_test_cases(args.name, outpath=args.path + "/../../video/", list_of_test_cases=test_cases)
            else:
                exp.make_video(args.name, 20, outpath=args.path + "/../../video/")
