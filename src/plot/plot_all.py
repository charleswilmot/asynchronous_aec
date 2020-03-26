import os
import argparse
from plot.write_data import FigureManager
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
        '-c', '--test-conf-path',
        type=str,
        action='store',
        default="../test_conf/test_pan_tilt_vergence_fast.pkl",
        help="Path to the test config file."
    )
    parser.add_argument(
        '-o', '--overwrite',
        action='store_true',
        help="Overwrite existing files"
    )
    # parser.add_argument(
    #     '-s', '--save',
    #     action='store_true',
    #     help="Save plots"
    # )
    args = parser.parse_args()

    test_data_path = args.path + "/test_data/"
    test_plots_path = args.path + "/test_plots/"
    train_data_path = args.path + "/data/training.data"
    merge_path = args.path + "/merge/"
    os.makedirs(merge_path, exist_ok=True)

    plot_train(train_data_path, True, args.overwrite)

    last_test = -1
    for test in os.listdir(test_data_path):
        print(test)
        test_episode = int(test.split("_")[0])
        last_test = max(last_test, test_episode)
        plot_test(test_data_path + test, args.test_conf_path, True, args.overwrite)
        path = test_plots_path + test.split(".")[0]
        for plot_filename in os.listdir(path):
            source = path + "/" + plot_filename
            destination = merge_path + plot_filename.split(".")[0] + "_{:07d}.png".format(test_episode)
            if not os.path.exists(destination):
                os.link(source, destination)

    os.system("eog {}/test_plots/{}*/*".format(args.path, last_test))
    # os.system("eog {}/train_plots/*/*".format(args.path))
