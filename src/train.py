import numpy as np
import time
import os
from PIL import Image
from algorithm.conf import Conf
from algorithm.experiment import Experiment
from helper.generate_test_conf import TestConf


default_mlr = 1e-4
default_clr = 1e-4


def get_textures(path):
    """Reads and return all the textures (ie images) found under path.
    """
    filepaths = [path + "/{}".format(x) for x in os.listdir(path) if x.endswith(".bmp") or x.endswith(".png")]
    return np.array([np.array(Image.open(x)) for x in filepaths])


def make_experiment_path(date=None, mlr=None, clr=None, description=None):
    date = date if date else time.strftime("%Y_%m_%d-%H.%M.%S", time.localtime())
    mlr = mlr if mlr else default_mlr
    clr = clr if clr else default_clr
    description = ("__" + description) if description else ""
    experiment_dir = "../experiments/{}_mlr{:.2e}_clr{:.2e}{}".format(
        date, mlr, clr, description)
    return experiment_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-ep', '--experiment-path',
        type=str,
        action='store',
        default="",
        help="Path to the experiment directory. Results are stored here."
    )

    parser.add_argument(
        '-np', '--n-parameter-servers',
        type=int,
        help="Number of parameter servers.",
        default=1
    )

    parser.add_argument(
        '-nw', '--n-workers',
        type=int,
        help="Number of workers.",
        default=8
    )

    parser.add_argument(
        '-d', '--description',
        type=str,
        help="Short description of the experiment.",
        default=""
    )

    parser.add_argument(
        '-t', '--tensorboard',
        action="store_true",
        help="Start TensorBoard."
    )

    parser.add_argument(
        '-ra', '--ratios',
        type=int,
        nargs="+",
        help="Downscaling ratios.",
        default=[1, 3]
    )

    parser.add_argument(
        '-b', '--buffer-size',
        type=int,
        help="Size of the replay buffer.",
        default=2000
    )

    parser.add_argument(
        '-4f', '--turn-2-frames-vergence-off',
        help="Self expl.",
        action="store_true",
        default=False
    )

    parser.add_argument(
        '-ne', '--n-episodes',
        type=int,
        help="Number of episodes to be simulated.",
        default=100000
    )

    parser.add_argument(
        '-el', '--episode-length',
        type=int,
        help="Length of an episode.",
        default=10
    )

    parser.add_argument(
        '-u', '--update-factor',
        type=int,
        help="Number of updates per simulated episode.",
        default=10
    )

    parser.add_argument(
        '-bs', '--batch-size',
        type=int,
        help="Size of one batch.",
        default=200
    )

    parser.add_argument(
        '-r', '--restore-from',
        type=str,
        default="none",
        help="Checkpoint to restore from."
    )

    parser.add_argument(
        '-rm', '--restore-model-from',
        type=str,
        default="none",
        help="Checkpoint to restore from (encoder / decoder only)"
    )

    parser.add_argument(
        '-mlr', '--model-learning-rate',
        type=float,
        default=default_mlr,
        help="model learning rate."
    )

    parser.add_argument(
        '-clr', '--critic-learning-rate',
        type=float,
        default=default_clr,
        help="critic learning rate."
    )

    parser.add_argument(
        '-df', '--discount-factor',
        type=float,
        default=0.001,
        help="Discount factor."
    )

    parser.add_argument(
        '-eps', '--epsilon',
        type=float,
        default=0.05,
        help="Initial value for epsilon."
    )

    parser.add_argument(
        '-epsd', '--epsilon-decay',
        type=float,
        default=1.0,
        help="Decay for epsilon."
    )

    parser.add_argument(
        '-bnd', '--batch-norm-decay',
        type=float,
        default=0.99,
        help="Decay the batch normalization estimators."
    )

    parser.add_argument(
        '-rsf', '--reward-scaling-factor',
        type=float,
        default=600,
        help="Reward scaling by a constant."
    )

    parser.add_argument(
        '-nv', '--no-video',
        action='store_true',
        help="Don't record a video if present"
    )

    parser.add_argument(
        '-tcp', '--test-conf-path',
        type=str,
        default="../test_conf/test_pan_tilt_vergence_fast.pkl",
        help="Path to the test config file"
    )

    args = parser.parse_args()

    if not args.experiment_path:
        experiment_dir = make_experiment_path(
            mlr=args.model_learning_rate,
            clr=args.critic_learning_rate,
            description=args.description)
    else:
        experiment_dir = args.experiment_path

    worker_conf = Conf(args)

    test_at = [0, 5000,
        10000, 15000, 20000,
        25000, 50000, 75000, 100000,
        125000, 150000, 175000, 200000,
        225000, 250000, 275000, 300000,
        325000, 350000, 375000, 400000,
        425000, 450000, 475000, 500000,
        525000, 550000, 575000, 600000,
        500000, 600000
    ]
    test_at = [x for x in test_at if x < args.n_episodes] + [np.inf]
    save_at = test_at[8::4]

    with Experiment(args.n_parameter_servers, args.n_workers, experiment_dir, worker_conf, args.test_conf_path) as exp:
        if args.restore_from != "none":
            exp.restore_all(args.restore_from)
        if args.restore_model_from != "none":
            exp.restore_model(args.restore_model_from)
        if args.tensorboard:
            exp.start_tensorboard()
        last_test = 0
        for test in test_at:
            if test < args.n_episodes:
                exp.train(test - last_test)
                last_test = test
            else:
                exp.train(args.n_episodes - last_test)
            if test in save_at:
                exp.save_model()
            exp.test()
            exp.generate_saliency_map()
        if test not in save_at:
            exp.save_model()
        if not args.no_video:
            exp.make_video("final", 100)
