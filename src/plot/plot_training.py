import os
import argparse
from read_data import read_training_data, read_all_abs_testing_performance, get_experiment_metadata
import numpy as np
import matplotlib.pyplot as plt
from helper.utils import vergence_error, to_angle, to_distance
from helper.generate_test_conf import TestConf
from plot.write_data import FigureManager


TRAINING_DATA_RECORD_FREQ = 10


def plot_n_vergence_performance(fig, data, n, m):
    ax = fig.add_subplot(111)
    vergence_errors = vergence_error(data["eyes_position"][-n:-m], data["object_distance"][-n:-m])
    num_iterations = range(len(data["eyes_position"][0]))
    for i, vergence_error_list in enumerate(vergence_errors):
        ax.plot(num_iterations, vergence_error_list / 90 * 320, color="green", linestyle="--")
        # ax.fill_between(num_iterations, data["eyes_position"][-i][..., -1], to_angle(data["object_distance"][-i]),
        #                 where=to_angle(data["object_distance"][-i]) >= data["eyes_position"][-i][..., -1], facecolor="blue")
        # ax.fill_between(num_iterations, data["eyes_position"][-i][..., -1], to_angle(data["object_distance"][-i]),
        #                 where=to_angle(data["object_distance"][-i]) <= data["eyes_position"][-i][..., -1], facecolor="red")
        #ax.plot(num_iterations, data["eyes_position"][-i][..., -1])
        #ax.plot(num_iterations, to_angle(data["object_distance"][-i]))
        ax.plot(num_iterations, to_distance(data["eyes_position"][-i-n][..., -1]), color="blue", linestyle="--")
        ax.plot(num_iterations, data["object_distance"][-i-n], color="red", linestyle="--")
    ax.axhline(0, color="k", linestyle="--")
    ax.legend()
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Speed error in px it-1\nVergence error in px")
    ax.set_ylim([-3, 6])
    ax.set_title("Training vergence error last 10 test episodes")


def plot_joint_errors(fig, data, abs_errors, win_size=500, stddev=200):
    """
    Plots the speed error of pan, tilt and vergence with respect to training time
    :param fig:
    :param data:
    :param abs_errors:
    :param win_size:
    :param stddev:
    :return:
    """
    ax = fig.add_subplot(111)
    # speed_errors_tilt = data["eyes_speed"][:, -1, 0] + data["object_speed"][:, -1, 0]
    # speed_errors = data["eyes_speed"][:, -1] - data["object_speed"][:, -1]
    # speed_errors_tilt = speed_error[:, 0]
    # speed_errors_pan = speed_error[:, 1]
    speed_errors_tilt = data["eyes_speed"][:, -1, 0] - data["object_speed"][:, -1, 0]
    speed_errors_pan = data["eyes_speed"][:, -1, 1] - data["object_speed"][:, -1, 1]
    vergence_errors = vergence_error(data["eyes_position"][:, -1], data["object_distance"][:, -1])
    kernel = np.exp(-(np.arange(-win_size / 2, win_size / 2) ** 2) / stddev ** 2) #RBF Kernel
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
    ax.set_ylabel("Speed error in px it-1\nVergence error in px")
    ax.set_ylim([-0.2, 6])
    ax.set_title("Speed error wrt train time")


def plot(data, abs_errors=None):
    # with FigureManager("joint_errors.png") as fig:
    #     plot_joint_errors(fig, data, abs_errors, 500, 200)
    with FigureManager("vergence_with_depth.png") as fig:
        plot_n_vergence_performance(fig, data, 20, 11)


def plot_train(experiment_metadata, save, overwrite):
    data = read_training_data(experiment_metadata["train_data_path"])
    abs_errors = read_all_abs_testing_performance(experiment_metadata["test_data_path"])
    n_episodes = data.shape[0]
    plots_dir = experiment_metadata["plot_training_path"] + "/{:08d}".format(n_episodes) + "/"
    FigureManager._path = plots_dir
    FigureManager._save = save
    if save:
        os.makedirs(experiment_metadata["plot_training_path"], exist_ok=True)
        try:
            os.makedirs(plots_dir, exist_ok=overwrite)
        except Exception as e:
            print(plots_dir, " : file exists")
            return
    plot(data, abs_errors)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'path', metavar="PATH",
        type=str,
        action='store',
        help="Path to the experiment."
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
    experiment_metadata = get_experiment_metadata(args.path)
    plot_train(experiment_metadata, save=True, overwrite=args.overwrite)
