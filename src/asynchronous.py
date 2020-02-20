import numpy as np
import time
import os
from PIL import Image
from algorithm.conf import Conf
from algorithm.experiment import Experiment


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
        '-fe', '--flush-every',
        type=int,
        help="Flush every N simulated episode.",
        default=5000
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
        '-r', '--restore-from',
        type=str,
        default="none",
        help="Checkpoint to restore from."
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
        default=0.1,
        help="Discount factor."
    )

    parser.add_argument(
        '-eps', '--epsilon',
        type=float,
        default=0.2,
        help="Initial value for epsilon."
    )

    parser.add_argument(
        '-epsd', '--epsilon-decay',
        type=float,
        default=1.0,
        help="Decay for epsilon."
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

    test_at = np.array([5000, 10000, 20000, 30000, 50000, 75000, 100000, 150000]) // args.flush_every

    with Experiment(args.n_parameter_servers, args.n_workers, experiment_dir, worker_conf) as exp:
        if args.restore_from != "none":
            exp.restore_model(args.restore_from)
        if args.tensorboard:
            exp.start_tensorboard()
        for i in range(args.n_episodes // args.flush_every):
            if i in test_at:
                exp.save_model()
                # exp.test("../test_conf/vergence_trajectory_4_distances.pkl")
            exp.train(args.flush_every)
            exp.flush_data()
        exp.save_model()
        # exp.test("../test_conf/vergence_trajectory_4_distances.pkl")
        exp.make_video("final", 100)
