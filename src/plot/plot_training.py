import os
import argparse
from read_data import read_training_data, read_all_abs_testing_performance
import numpy as np
import matplotlib.pyplot as plt
from helper.utils import vergence_error


TRAINING_DATA_RECORD_FREQ = 10


def plot_joint_errors(fig, data, abs_errors, win_size=500, stddev=200):
    ax = fig.add_subplot(111)
    # speed_errors_tilt = data["eyes_speed"][:, -1, 0] + data["object_speed"][:, -1, 0]
    # speed_errors = data["eyes_speed"][:, -1] - data["object_speed"][:, -1]
    # speed_errors_tilt = speed_error[:, 0]
    # speed_errors_pan = speed_error[:, 1]
    speed_errors_tilt = data["eyes_speed"][:, -1, 0] - data["object_speed"][:, -1, 0]
    speed_errors_pan = data["eyes_speed"][:, -1, 1] - data["object_speed"][:, -1, 1]
    vergence_errors = vergence_error(data["eyes_position"][:, -1], data["object_distance"][:, -1])
    kernel = np.exp(-(np.arange(-win_size / 2, win_size / 2) ** 2) / stddev ** 2)
    kernel /= np.sum(kernel)
    a = np.convolve(np.abs(speed_errors_tilt), kernel, mode="valid")
    b = np.convolve(np.abs(speed_errors_pan), kernel, mode="valid")
    c = np.convolve(np.abs(vergence_errors), kernel, mode="valid")
    x = np.arange(win_size / 2, a.shape[0] + win_size / 2) * TRAINING_DATA_RECORD_FREQ
    line, = ax.plot(x, a / 90 * 320, label="tilt")
    test_x, test_y = abs_errors["tilt"]
    ax.plot(test_x, test_y / 90 * 320, color=line.get_color(), linestyle="--")
    line, = ax.plot(x, b / 90 * 320, label="pan")
    test_x, test_y = abs_errors["pan"]
    ax.plot(test_x, test_y / 90 * 320, color=line.get_color(), linestyle="--")
    line, = ax.plot(x, c / 90 * 320, label="vergence")
    test_x, test_y = abs_errors["vergence"]
    ax.plot(test_x, test_y / 90 * 320, color=line.get_color(), linestyle="--")
    ax.axhline(1, color="k", linestyle="--")
    ax.legend()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Speed error in pixels/it")
    ax.set_ylim([-0.2, 8])
    ax.set_title("Speed error wrt train time")


def plot(data, abs_errors):
    with FigureManager("joint_errors.png") as fig:
        plot_joint_errors(fig, data, abs_errors, 500, 200)









if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'path', metavar="PATH",
        type=str,
        action='store',
        help="Path to the train_data."
    )
    parser.add_argument(
        '-o', '--overwrite',
        action='store_true',
        help="Overwrite existing files"
    )
    parser.add_argument(
        '-s', '--save',
        action='store_true',
        help="Save plots"
    )
    args = parser.parse_args()
    train_plots_dir = os.path.dirname(os.path.abspath(args.path)) + "/../train_plots/"
    data = read_training_data(args.path)
    test_data_path, _ = os.path.split(args.path)
    test_data_path, _ = os.path.split(test_data_path)
    test_data_path += '/test_data/'
    abs_errors = read_all_abs_testing_performance(test_data_path)
    n_episodes = data.shape[0]
    plots_dir = train_plots_dir + "{:08d}".format(n_episodes) + "/"
    if args.save:
        os.makedirs(train_plots_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=args.overwrite)


    class FigureManager:
        def __init__(self, filename):
            self._fig = plt.figure(dpi=200)
            self._path = plots_dir
            self._filename = filename

        def __enter__(self):
            return self._fig

        def __exit__(self, exc_type, exc_value, exc_traceback):
            if args.save:
                print("saving plot {}  ...  ".format(self._filename), end="")
                self._fig.savefig(self._path + self._filename)
                print("done")
                plt.close(self._fig)
            else:
                plt.show()

    plot(data, abs_errors)
