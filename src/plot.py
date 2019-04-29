import RESSOURCES
from scipy.stats import linregress, gaussian_kde
from scipy.signal import savgol_filter
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import re
import argparse


keys = [
    "train_step",
    "rewards",
    "actions",
    "critic_values",
    "object_distance",
    "eyes_position",
    "eyes_speed"
]


def get_worker_and_flush_ids(path):
    ids = {}
    for f in os.listdir(path):
        m = re.match("worker_([0-9]+)_flush_([0-9]+).pkl", f)
        if m:
            worker_id = int(m.group(1))
            flush_id = int(m.group(2))
            if worker_id not in ids:
                ids[worker_id] = set()
            ids[worker_id].add(flush_id)
    return ids


def get_data(path, flush_id=None):
    if flush_id is None:
        worker_ids = get_worker_and_flush_ids(path)
        flush_id = list(worker_ids[0])
        flush_id.sort()
        return get_data(path, flush_id)
    elif type(flush_id) is int:
        worker_ids = get_worker_and_flush_ids(path)
        if flush_id < 0:
            flush_ids = list(worker_ids[0])
            flush_ids.sort()
            return get_data(path, flush_ids[flush_id:])
        data = []
        for worker_id in worker_ids:
            with open(path + "/worker_{:04d}_flush_{:04d}.pkl".format(worker_id, flush_id), "rb") as f:
                data += pickle.load(f)
        return data
    else:
        data = []
        for i in flush_id:
            data += get_data(path, i)
        return data


def vergence_error(d):
    vergence = d["eyes_position"][-1][0]
    object_distance = d["object_distance"]
    return vergence + np.degrees(np.arctan2(RESSOURCES.Y_EYES_DISTANCE, 2 * object_distance)) * 2


def reward_wrt_vergence_fill_ax(ax, vergence_errors, reward, object_distances, title=None):
    # poly = np.poly1d(np.polyfit(vergence_errors, reward, 2))
    for sub_vergence_errors, sub_reward, sub_object_distances in zip(vergence_errors, reward, object_distances):
        X = np.linspace(min(sub_vergence_errors), max(sub_vergence_errors), 100)
        indices = [np.where(np.logical_and(x - 0.1 < sub_vergence_errors, sub_vergence_errors < x + 0.1)) for x in X]
        mean = [np.mean(np.array(sub_reward)[idx]) for idx in indices]
        ax.scatter(sub_vergence_errors, sub_reward, s=10, alpha=0.3)
        # ax.plot(X, poly(X), "--")
        ax.plot(X, mean, linewidth=3)
    ax.axvline(0, color="k")
    ax.set_title(title)
    ax.set_xlabel("vergence error")
    ax.set_ylabel("reward")


def reward_wrt_vergence(data, dist_bins=[1, 2, 3, 4], ax_fine=None, ax_coarse=None, ax_both=None):
    bounds = zip([-100] + dist_bins, dist_bins + [100])
    data = [[d for d in data if min_dist < d["object_distance"] < max_dist] for min_dist, max_dist in bounds]
    vergence_errors = [[vergence_error(d) for d in subdata] for subdata in data]
    object_distances = [np.array([d["object_distance"] for d in subdata]) for subdata in data]
    if ax_fine is not None:
        fine_rewards = [[d["rewards"][0][0] for d in subdata] for subdata in data]
        reward_wrt_vergence_fill_ax(ax_fine, vergence_errors, fine_rewards, object_distances,
                                    title="reward wrt vergence error (fine scale)")
    if ax_coarse is not None:
        coarse_rewards = [[d["rewards"][0][1] for d in subdata] for subdata in data]
        reward_wrt_vergence_fill_ax(ax_coarse, vergence_errors, coarse_rewards, object_distances,
                                    title="reward wrt vergence error (coarse scale)")
    if ax_both is not None:
        both_rewards = [[sum(d["rewards"][0]) for d in subdata] for subdata in data]
        reward_wrt_vergence_fill_ax(ax_both, vergence_errors, both_rewards, object_distances,
                                    title="reward wrt vergence error")


def reward_wrt_vergence_better_fill_ax(ax, vergence_errors, rewards, title=None):
    N = 150
    X = np.linspace(min(vergence_errors), max(vergence_errors), N)
    Y = np.linspace(min(rewards), max(rewards), N)
    probs = np.zeros((N, N))
    for i, x in enumerate(X):
        weights = np.exp(-3 * np.abs(vergence_errors - x))
        weights /= np.sum(weights)
        kde = gaussian_kde(rewards, weights=weights)
        probs[:, i] = kde(Y)
    ax.contourf(X, Y, probs)
    mean = np.sum(probs * Y[:, np.newaxis], axis=0) / np.sum(probs, axis=0)
    ax.plot(X, mean, "r-")
    ax.plot(X, Y[np.argmax(probs, axis=0)], "k-")
    ax.axvline(0, c="k")
    ax.set_title(title)
    ax.set_xlabel("vergence error")
    ax.set_ylabel("reward")


def reward_wrt_vergence_better(data, min_dist, max_dist, ax_fine=None, ax_coarse=None, ax_both=None):
    data = [d for d in data if min_dist < d["object_distance"] < max_dist]
    vergence_errors = [vergence_error(d) for d in data]
    if ax_fine is not None:
        fine_rewards = [d["rewards"][0][0] for d in data]
        reward_wrt_vergence_better_fill_ax(ax_fine, vergence_errors, fine_rewards,
                                           title="fine scale")
    if ax_coarse is not None:
        coarse_rewards = [d["rewards"][0][1] for d in data]
        reward_wrt_vergence_better_fill_ax(ax_coarse, vergence_errors, coarse_rewards,
                                           title="coarse scale")
    if ax_both is not None:
        both_rewards = [sum(d["rewards"][0]) for d in data]
        reward_wrt_vergence_better_fill_ax(ax_both, vergence_errors, both_rewards,
                                           title="both scales")


def action_wrt_vergence_fill_ax(ax, vergence_errors, action, title=None):
    ax.plot(vergence_errors, action, ".")
    ax.set_title(title)
    ax.set_xlabel("vergence error")
    ax.set_ylabel("action")


def action_wrt_vergence(data, min_dist, max_dist, ax=None, action_set=None):
    data = [d for d in data if min_dist < d["object_distance"] < max_dist]
    vergence_errors = [vergence_error(d) for d in data]
    actions = [d["actions"][2] for d in data]
    if action_set is not None:
        actions = np.asarray(action_set)[actions]
        ax.plot([action_set[0], action_set[-1]], [action_set[0], action_set[-1]], "k-")
    if ax is not None:
        action_wrt_vergence_fill_ax(ax, vergence_errors, actions, title="action wrt vergence error")


def reward_wrt_critic_value_fill_ax(ax, critic, reward, title=None):
    ax.plot(critic, reward, ".")
    ax.set_title(title)
    ax.set_xlabel("Critic value")
    ax.set_ylabel("Reward")
    ax.axis('scaled')
    slope, intercept, r_value, p_value, std_err = linregress(critic, reward)
    X = np.linspace(min(critic), max(critic))
    Y = slope * X + intercept
    ax.plot(X, Y, "k")
    ax.text(min(critic), max(critic) * slope + intercept, "corr: {:.2f}".format(r_value))


def reward_wrt_critic_value(data, ax_fine=None, ax_coarse=None, ax_both=None):
    critic = [d["critic_values"][0] for d in data]
    rewards = [d["rewards"][0] for d in data]
    if ax_fine is not None:
        reward_wrt_critic_value_fill_ax(ax_fine, [c[0] for c in critic], [r[0] for r in rewards],
                                        title="reward wrt critic value (fine scale)")
    if ax_coarse is not None:
        reward_wrt_critic_value_fill_ax(ax_coarse, [c[1] for c in critic], [r[1] for r in rewards],
                                        title="reward wrt critic value (coarse scale)")
    if ax_both is not None:
        reward_wrt_critic_value_fill_ax(ax_both, [sum(c) for c in critic], [sum(r) for r in rewards],
                                        title="reward wrt critic value")


def vergence_error_wrt_episode(data, ax):
    train_step = [d["train_step"] for d in data]
    vergence_errors = [vergence_error(d) for d in data]
    ax.plot(train_step, vergence_errors, ".", alpha=0.8)
    ax.set_xlabel("episode")
    ax.set_ylabel("vergence error in degrees")
    ax.set_title("vergence error wrt time")


def mean_abs_vergence_error_wrt_episode(data, ax):
    train_step = np.array([d["train_step"] for d in data])
    abs_vergence_errors = np.array([np.abs(vergence_error(d)) for d in data])
    args = np.argsort(train_step)
    train_step = train_step[args]
    abs_vergence_errors = abs_vergence_errors[args]
    for win in [200, 500, 1000]:
        mean_abs_vergence_errors = [np.mean(abs_vergence_errors[np.where(np.logical_and(train_step < x + win, train_step > x - win))]) for x in train_step]
        smooth = savgol_filter(mean_abs_vergence_errors, 21, 3)
        ax.plot(train_step, smooth, "-", lw=2, label="+-{}".format(win))
    ax.legend()
    ax.set_xlabel("episode")
    ax.set_ylabel("abs vergence error in degrees")
    ax.set_title("vergence error wrt time")


def vergence_wrt_episode(data, ax):
    train_step = np.array([d["train_step"] for d in data])
    vergence = np.array([d["eyes_position"][-1][0] for d in data])
    args = np.argsort(train_step)
    train_step = train_step[args]
    vergence = vergence[args]
    ax.scatter(train_step, vergence, alpha=0.1)
    ax.set_xlabel("episode")
    ax.set_ylabel("vergence position in degrees")
    ax.set_title("vergence wrt time")


def vergence_wrt_object_distance(data, ax):
    object_distance = np.array([d["object_distance"] for d in data])
    vergence = np.array([d["eyes_position"][-1][0] for d in data])
    rewards = np.array([sum(d["rewards"][0]) for d in data])
    X = np.linspace(np.min(object_distance), np.max(object_distance), 100)
    correct = -np.degrees(np.arctan2(RESSOURCES.Y_EYES_DISTANCE, 2 * X)) * 2
    ax.scatter(object_distance, vergence, c=rewards, cmap="Greys")
    ax.plot(X, correct, "k-")
    ax.set_xlabel("object position in meters")
    ax.set_ylabel("vergence position in degrees")
    ax.set_title("final vergence wrt object position")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path', metavar="PATH",
        type=str,
        action='store',
        help="Path to the data."
    )

    parser.add_argument(
        '-s', '--save',
        action='store_true',
        help="If turned on, saves the plots."
    )

    args = parser.parse_args()
    path = args.path
    plotpath = path + "/../plots/"
    # path = "../experiments/tmp6/data/"
    # path = "../experiments/conv_no_reward_norm/data/"
    # path = "../experiments/good_fixation_init_small_filters_no_reward_norm_alr_1e-2/data/"

    data = get_data(path, -5)
    if args.save:
        os.mkdir(plotpath)

    fig = plt.figure()
    ax_fine = fig.add_subplot(131)
    ax_coarse = fig.add_subplot(132)
    ax_both = fig.add_subplot(133)
    reward_wrt_vergence_better(data, 1, 4, ax_fine=ax_fine, ax_coarse=ax_coarse, ax_both=ax_both)
    # reward_wrt_vergence(data, ax_fine=ax_fine, ax_coarse=ax_coarse, ax_both=ax_both)
    if args.save:
        fig.savefig(plotpath + "/reward_wrt_vergence.png")
    else:
        plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    vergence_wrt_object_distance(data, ax)
    if args.save:
        fig.savefig(plotpath + "/vergence_wrt_object_distance.png")
    else:
        plt.show()

    fig = plt.figure()
    ax_fine = fig.add_subplot(131)
    ax_coarse = fig.add_subplot(132)
    ax_both = fig.add_subplot(133)
    reward_wrt_critic_value(data, ax_fine=ax_fine, ax_coarse=ax_coarse, ax_both=ax_both)
    if args.save:
        fig.savefig(plotpath + "/reward_wrd_critic.png")
    else:
        plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    n_actions_per_joint = 9
    n = n_actions_per_joint // 2
    mini = 0.28
    maxi = 0.28 * 2 ** (n - 1)
    positive = np.logspace(np.log2(mini), np.log2(maxi), n, base=2)
    negative = -positive[::-1]
    action_set = np.concatenate([negative, [0], positive])
    action_wrt_vergence(data, 0.5, 5, ax=ax, action_set=action_set)
    if args.save:
        fig.savefig(plotpath + "/action_wrt_vergence.png")
    else:
        plt.show()

    data = get_data(path)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    vergence_error_wrt_episode(data, ax=ax)
    if args.save:
        fig.savefig(plotpath + "/vergence_error_wrt_time.png")
    else:
        plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    vergence_wrt_episode(data, ax)
    if args.save:
        fig.savefig(plotpath + "/vergence_wrt_time.png")
    else:
        plt.show()


    fig = plt.figure()
    ax = fig.add_subplot(111)
    mean_abs_vergence_error_wrt_episode(data, ax=ax)
    if args.save:
        fig.savefig(plotpath + "/vergence_error_wrt_time.png")
    else:
        plt.show()
