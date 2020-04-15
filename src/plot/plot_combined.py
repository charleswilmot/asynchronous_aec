import os
import argparse
from plot.read_data import read_training_data, read_all_abs_testing_performance, get_experiment_metadata
import numpy as np
import matplotlib.pyplot as plt
from helper.utils import vergence_error
from helper.generate_test_conf import TestConf
from plot.write_data import FigureManager
from scipy.signal import convolve2d


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
    ax.set_ylabel("Speed error in px it-1\nVergence error in px")
    ax.set_ylim([-0.2, 6])
    ax.set_title("Speed error wrt train time")


def plot(data, abs_errors):
    with FigureManager("joint_errors.png") as fig:
        plot_joint_errors(fig, data, abs_errors, 500, 200)


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


def plot_mean_absolute_error_ax(ax, testing_x, testing_data, training_data, joint_name, set_ylabel=False, ylim=[-0.2, 4], legend=False, yscale="linear"):
    winsize = 500
    TRAINING_DATA_RECORD_FREQ = 10
    kernel = np.full((1, winsize), fill_value=1 / winsize)
    training_data = convolve2d(training_data, kernel, mode="valid")

    X = np.arange(winsize / 2, training_data.shape[1] + winsize / 2) * TRAINING_DATA_RECORD_FREQ
    training_data_mean = np.mean(training_data, axis=0)
    # training_data_std = np.std(training_data, axis=0)

    testing_data_mean = np.mean(testing_data, axis=0)
    testing_data_std = np.std(testing_data, axis=0)

    # ax.fill_between(X, training_data_mean - training_data_std, training_data_mean + training_data_std, alpha=0.5)
    ax.set_yscale(yscale)
    ax.set_ylim(ylim)
    ax.plot(X, training_data_mean, label="training")
    ax.errorbar(testing_x, testing_data_mean, yerr=testing_data_std, label="testing")
    ax.axhline(1, color="grey", linestyle="--")
    if set_ylabel:
        ax.set_ylabel("Mean absolute joint error in px (vergence) or px.it-1 (pan, tilt)")
    else:
        ax.set_yticks([])
    if legend:
        ax.legend()
    ax.set_xlabel("# Episode ({})".format(joint_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'paths', metavar="PATHS",
        type=str,
        action='store',
        nargs="+",
        help="Path to the experiments."
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
    paths = args.paths
    # paths = [
    #     "../experiments/2020_04_07-15.47.11_mlr5.00e-04_clr5.00e-04__ICDL_error_bars/",
    #     "../experiments/2020_04_07-15.47.43_mlr5.00e-04_clr5.00e-04__ICDL_error_bars/",
    #     "../experiments/2020_04_07-15.47.59_mlr5.00e-04_clr5.00e-04__ICDL_error_bars/",
    #     "../experiments/2020_04_07-15.48.31_mlr5.00e-04_clr5.00e-04__ICDL_error_bars/",
    #     "../experiments/2020_04_07-15.48.46_mlr5.00e-04_clr5.00e-04__ICDL_error_bars/"
    #     ]
    paths = [
        "../experiments/2020_04_10-12.43.39_mlr5.00e-04_clr5.00e-04__ICDL_error_bars",
        "../experiments/2020_04_10-12.43.56_mlr5.00e-04_clr5.00e-04__ICDL_error_bars",
        "../experiments/2020_04_10-12.44.28_mlr5.00e-04_clr5.00e-04__ICDL_error_bars",
        "../experiments/2020_04_10-12.44.44_mlr5.00e-04_clr5.00e-04__ICDL_error_bars",
        "../experiments/2020_04_10-12.45.00_mlr5.00e-04_clr5.00e-04__ICDL_error_bars"
    ]
    experiment_metadatas = []
    speed_errors_tilt = []
    speed_errors_pan = []
    vergence_errors = []
    testing_tilt = []
    testing_pan = []
    testing_vergence = []
    for path in paths:
        experiment_metadatas.append(get_experiment_metadata(path))
    for experiment_metadata in experiment_metadatas:
        testing = read_all_abs_testing_performance(experiment_metadata["test_data_path"])
        testing_tilt.append(320 / 90 * testing["tilt"][1])
        testing_pan.append(320 / 90 * testing["pan"][1])
        testing_vergence.append(320 / 90 * testing["vergence"][1])
        # testing = {"pan": [[test_indices], [test_values]], "tilt": ..., "vergence": ...}
        training = read_training_data(experiment_metadata["train_data_path"])
        speed_errors_tilt.append(320 / 90 * np.abs(training["eyes_speed"][:, -1, 0] - training["object_speed"][:, -1, 0]))
        speed_errors_pan.append(320 / 90 * np.abs(training["eyes_speed"][:, -1, 1] - training["object_speed"][:, -1, 1]))
        vergence_errors.append(320 / 90 * np.abs(vergence_error(training["eyes_position"][:, -1], training["object_distance"][:, -1])))

    min_length = min(map(len, speed_errors_tilt))
    print("min length:", min_length)
    speed_errors_tilt = np.array([x[:min_length] for x in speed_errors_tilt])
    speed_errors_pan = np.array([x[:min_length] for x in speed_errors_pan])
    vergence_errors = np.array([x[:min_length] for x in vergence_errors])

    # winsize = 500
    # TRAINING_DATA_RECORD_FREQ = 10
    # kernel = np.full((1, winsize), fill_value=1 / winsize)
    # speed_errors_tilt = convolve2d(speed_errors_tilt, kernel, mode="valid")
    # speed_errors_pan = convolve2d(speed_errors_pan, kernel, mode="valid")
    # vergence_errors = convolve2d(vergence_errors, kernel, mode="valid")

    # print(speed_errors_tilt.shape, speed_errors_pan.shape, vergence_errors.shape)

    # X = np.arange(winsize / 2, speed_errors_tilt.shape[1] + winsize / 2) * TRAINING_DATA_RECORD_FREQ
    # speed_errors_tilt_mean = np.mean(speed_errors_tilt, axis=0)
    # speed_errors_pan_mean = np.mean(speed_errors_pan, axis=0)
    # vergence_errors_mean = np.mean(vergence_errors, axis=0)
    # speed_errors_tilt_std = np.std(speed_errors_tilt, axis=0)
    # speed_errors_pan_std = np.std(speed_errors_pan, axis=0)
    # vergence_errors_std = np.std(vergence_errors, axis=0)


    testing_x = testing["vergence"][0]
    testing_tilt = np.array(testing_tilt)
    testing_pan = np.array(testing_pan)
    testing_vergence = np.array(testing_vergence)
    # testing_tilt_mean = np.mean(testing_tilt, axis=0)
    # testing_pan_mean = np.mean(testing_pan, axis=0)
    # testing_vergence_mean = np.mean(testing_vergence, axis=0)
    # testing_tilt_std = np.std(testing_tilt, axis=0)
    # testing_pan_std = np.std(testing_pan, axis=0)
    # testing_vergence_std = np.std(testing_vergence, axis=0)

    #plot_train(experiment_metadata, save=True, overwrite=args.overwrite)
    FigureManager._path = "./"
    FigureManager._save = args.save

    with FigureManager("joint_errors_combined.png") as fig:
        ax = fig.add_subplot(131)
        plot_mean_absolute_error_ax(ax, testing_x, testing_pan, speed_errors_pan, "pan", set_ylabel=True)
        ax = fig.add_subplot(132)
        plot_mean_absolute_error_ax(ax, testing_x, testing_tilt, speed_errors_tilt, "tilt")
        ax = fig.add_subplot(133)
        plot_mean_absolute_error_ax(ax, testing_x, testing_vergence, vergence_errors, "vergence", legend=True)
