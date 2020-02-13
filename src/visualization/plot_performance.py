import matplotlib.pyplot as plt
import numpy as np
from visualization.plot_log_data import get_data, group_by_episode, vergence_error
import pickle
import os
import argparse


def performances(train_data, test_data, save=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # TRAIN
    train_data = group_by_episode(train_data)
    train_data = {k: v[:, -1] for k, v in train_data.items()}
    episode_numbers = train_data["episode_number"]
    vergence_errors = vergence_error(train_data["eyes_position"], train_data["object_distance"])
    abs_vergence_errors = np.array(np.abs(vergence_errors))
    args = np.argsort(episode_numbers)
    episode_numbers = episode_numbers[args]
    abs_vergence_errors = abs_vergence_errors[args]
    win = 10000
    # mean_abs_vergence_errors = [np.mean(abs_vergence_errors[np.where(np.logical_and(episode_numbers < x + win, episode_numbers > x - win))]) for x in episode_numbers]
    # smooth = savgol_filter(mean_abs_vergence_errors, 21, 3)
    # ax.plot(episode_numbers, smooth, "-", lw=2, label="train")
    P75 = [np.percentile(abs_vergence_errors[np.where(np.logical_and(episode_numbers < x + win, episode_numbers > x - win))], 75) for x in episode_numbers]
    P50 = [np.percentile(abs_vergence_errors[np.where(np.logical_and(episode_numbers < x + win, episode_numbers > x - win))], 50) for x in episode_numbers]
    P25 = [np.percentile(abs_vergence_errors[np.where(np.logical_and(episode_numbers < x + win, episode_numbers > x - win))], 25) for x in episode_numbers]
    ax.fill_between(episode_numbers, P25, P75, color="grey", label="training IQR")
    ax.plot(episode_numbers, P50, color="r", label="training median")
    # TEST
    episodes = []
    abs_vergence_errors = []
    # vergence_error_std = []
    for episode, episode_test_data in test_data:
        episodes.append(episode)
        print(episode)
        # print(episode_test_data[0][1]["vergence_error"][-1])
        # print("\n")
        # print(episode_test_data[0][1]["vergence_error"].shape)
        episode_test_data = [np.abs(b["vergence_error"][-1]) for a, b in episode_test_data] # if a["object_distance"] > 2]
        abs_vergence_errors.append(episode_test_data)
        # abs_vergence_errors.append([np.percentile(np.abs(episode_test_data), p) for p in [50, 80]])
    # abs_vergence_errors = np.array(abs_vergence_errors)
    # ax.plot(episodes, abs_vergence_errors[:, 0], "k-", alpha=1.0, label="test_50")
    # ax.plot(episodes, abs_vergence_errors[:, 1], "k-", alpha=0.5, label="test_80")
    ax.boxplot(abs_vergence_errors, positions=episodes, showfliers=False, widths=5000)
    # OTHER
    ax.axhline(90 / 320, color="k", linestyle="--")
    ax.legend()
    ax.set_xlabel("episode")
    ax.set_ylabel("abs vergence error in degrees")
    ax.set_ylim([0.0, 1.5])
    ax.set_xlim([0.0, np.max(episode_numbers) + 5000])
    ax.set_title("vergence error wrt time")
    if save:
        fig.savefig(plotpath + "/performances.png")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'path', metavar="PATH",
        type=str,
        action='store',
        help="Path to the experiment dir."
    )
    parser.add_argument(
        'test_conf_path', metavar="TEST_CONF_PATH",
        type=str,
        action='store',
        help="Path to the test config file."
    )
    parser.add_argument(
        '-s', '--save',
        action='store_true',
        help="If turned on, saves the plots."
    )
    args = parser.parse_args()

    test_conf_basename = os.path.basename(args.test_conf_path)
    test_conf_name = os.path.splitext(test_conf_basename)[0]
    print("read train data")
    train_data = get_data(args.path + "/data/")
    print("read test data")
    test_data = []
    filenamelist = os.listdir(args.path + "/test_data/")
    filenamelist = list(filter(lambda x: x.find(test_conf_name) != -1, filenamelist))
    filenamelist.sort(key=lambda x: int(x.split("_")[0]))
    for filename in filenamelist:
        with open(args.path + "/test_data/" + filename, "rb") as f:
            test_data.append((int(filename.split("_")[0]), pickle.load(f)))

    plotpath = args.path
    print("start ploting")
    performances(train_data, test_data, save=args.save)
