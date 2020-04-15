import os
import argparse
from plot.write_data import FigureManager
from plot.read_data import get_experiment_metadata
from plot.plot_testing import plot_test
from plot.plot_training import plot_train
from helper.generate_test_conf import TestConf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'path', metavar="PATH",
        type=str,
        action='store',
        help="Path to the experiment directory."
    )
    parser.add_argument(
        '-o', '--overwrite',
        action='store_true',
        help="Overwrite existing files"
    )
    args = parser.parse_args()

    experiment_metadata = get_experiment_metadata(args.path)
    os.makedirs(experiment_metadata["merge_path"], exist_ok=True)
    plot_train(experiment_metadata, save=True, overwrite=args.overwrite)
    plot_test(experiment_metadata, save=True, overwrite=args.overwrite)

    os.system("eog {}/*/*".format(experiment_metadata["plot_training_path"]))
