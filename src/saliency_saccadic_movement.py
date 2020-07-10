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
        '-n', "--name",
        type=str,
        action='store',
        default="default_name",
        help="Name of the video."
    )

    args = parser.parse_args()
    experiment_dir = args.path

    n_parameter_servers = 1
    n_workers = 1

    #test_cases = None
    # "(Stimulus ID, Initial depth position, Initial vergence error, Initial speed error;Pan/Tilt speed, Depth speed, Episode iterations)"
    test_cases = [
        # -------------Vergence----------------
        # test_case(9, 2.5, 0, [0, 0.], 0.00, 1),
        # test_case(9, 4, -1.5, [0, 0.], 0.00, 1),
        # test_case(9, 5, -2.5, [0, 0.], 0.00, 1),
        # test_case(9, 6, -3.5, [0, 0.], 0.00, 1),
        # test_case(9, 8, -5.5, [0, 0.], 0.00, 1),
        # test_case(9, 2.5, 0, [0, 0.], 0.00, 1),
        # test_case(9, 2.5, 1.5, [0, 0.], 0.00, 1),
        # test_case(9, 2.5, 2.5, [0, 0.], 0.00, 1),
        # test_case(9, 2.5, 3.5, [0, 0.], 0.00, 1),
        # test_case(9, 2.5, 5.5, [0, 0.], 0.00, 1),
        # -------------Speed----------------
        # test_case(15, 2, 2, [0.6, 1.2], 0.00, 40),
        # test_case(16, 2, 2, [0, 1.2], 0.00, 40),
        # test_case(17, 2, 2, [0, 0], 0.00, 40),
        # test_case(11, 4, 0, [0, 0.6], 0.00, 40),
        # test_case(11, 4, 0, [0, 0.6], 0.00, 40),
        # test_case(12, 4, 0, [0, 0.9], 0.00, 40),
        # test_case(12, 4, 0, [0.6, 0.9], 0.00, 40),
        # test_case(13, 5, 0, [0, 1.2], 0.00, 40),
        # test_case(13, 5, 0, [0.6, 1.2], 0.00, 40),
        # test_case(13, 7, 0, [0, 1.2], 0.00, 40),
        # test_case(14, 7, 0, [0.6, 1.2], 0.00, 40),
        # test_case(14, 7, 0, [0, 1.2], 0.00, 40),
        # test_case(15, 8, 2, [0.6, 1.2], 0.00, 40),
        # test_case(16, 8, 2, [0, 1.2], 0.00, 40),
        # test_case(17, 8, 2, [0, 0], 0.00, 40),
        #
        #------------------
        test_case(1, 4, 2, [0.6, 0.6], 0.00, 80),
        test_case(2, 4, 2, [0.6, -0.6], 0.00, 80),
        test_case(3, 3, 2, [-0.6, -0.6], 0.00, 80),
        test_case(4, 4, 0, [-0.6, 0.6], 0.00, 80),
        test_case(5, 4, 3, [0, 0.6], 0.00, 80),
        test_case(6, 5, 0, [0, 0.9], 0.00, 80),
        test_case(7, 5, 0, [0.6, 0.6], 0.20, 80),
        # test_case(8, 6, 0, [0, 1.2], -0.20, 40),
        # test_case(9, 6, 0, [0.6, 1.2], 0.00, 40),
        # test_case(10, 7, 2, [0, 1.2], 0.00, 40),
        # test_case(11, 7, 0, [0.6, 0.6], 0.00, 40),
        #test_case(12, 7, 0, [0, 1.2], 0.00, 40),
        # test_case(15, 8, 2, [0.6, 1.2], 0.00, 40),
        # test_case(16, 8, 2, [0, 1.2], 0.00, 40),
        # test_case(17, 8, 2, [0, 0], 0.00, 40),

        #------------------

        # test_case(11, 4, 0, [0, 2.1], 0.00, 1),
        # test_case(11, 4, 0, [0, 2.4], 0.00, 1),
        # test_case(10, 2.5, 0, [0, 0.], -0.01, 20),
        # test_case(10, 4, 0, [0, 0.], -0.01, 20),
        # test_case(10, 5, 0, [0, 0.], -0.01, 20),
        # test_case(10, 6, 0, [0, 0.], -0.01, 20),
        # test_case(10, 8, 0, [0, 0.], -0.01, 20),
        # test_case(15, 2.5, 0, [0, 0.], -0.01, 20),
        # test_case(15, 4, 0, [0, 0.], -0.01, 20),
        # test_case(15, 5, 0, [0, 0.], -0.01, 20),
        # test_case(15, 6, 0, [0, 0.], -0.01, 20),
        # test_case(15, 8, 0, [0, 0.], -0.01, 20),
    ]

    with open(experiment_dir + "/../../conf/worker_conf.pkl", "rb") as f:
        worker_conf = pickle.load(f)

    with TemporaryDirectory() as d:
        with Experiment(n_parameter_servers, n_workers, d + "/dummy/", worker_conf, worker0_display=False, mt_exp=True) as exp:
            exp.restore_all(experiment_dir)
            # Makes a normal video based on the experiment parameters if no test conf file if given via command line or
            # makes a video of specific test cases if test conf file is given via command line. If a test conf file is
            # given, test_conf.data["test_cases_policy_dependent"] test cases are used (with a limit of 100, can be
            # ajusted) if test_cases in this file is None. If test_cases in this file is NOT None, the test cases of
            # this file are being used.
            exp.saliency_saccadic_movement(args.name, 20, outpath=args.path + "/../../saliency/", list_of_test_cases=test_cases)
