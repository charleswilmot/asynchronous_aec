import RESSOURCES
from scipy.stats import linregress
from scipy.signal import savgol_filter
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import re


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


def reward_wrt_vergence_fill_ax(ax, vergence_errors, reward, title=None):
    poly = np.poly1d(np.polyfit(vergence_errors, reward, 2))
    X = np.linspace(min(vergence_errors), max(vergence_errors), 100)
    indices = [np.where(np.logical_and(x - 0.28 < vergence_errors, vergence_errors < x + 0.28)) for x in X]
    mean = [np.mean(np.array(reward)[idx]) for idx in indices]
    ax.plot(vergence_errors, reward, ".", X, poly(X), "--")
    ax.plot(X, mean)
    ax.set_title(title)
    ax.set_xlabel("vergence error")
    ax.set_ylabel("reward")


def reward_wrt_vergence(data, min_dist, max_dist, ax_fine=None, ax_coarse=None, ax_both=None):
    data = [d for d in data if min_dist < d["object_distance"] < max_dist]
    vergence_errors = [vergence_error(d) for d in data]
    rewards = [d["rewards"][0] for d in data]
    if ax_fine is not None:
        reward_wrt_vergence_fill_ax(ax_fine, vergence_errors, [r[0] for r in rewards],
                                    title="reward wrt vergence error (fine scale)")
    if ax_coarse is not None:
        reward_wrt_vergence_fill_ax(ax_coarse, vergence_errors, [r[1] for r in rewards],
                                    title="reward wrt vergence error (coarse scale)")
    if ax_both is not None:
        reward_wrt_vergence_fill_ax(ax_both, vergence_errors, [sum(r) for r in rewards],
                                    title="reward wrt vergence error")


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


def mean_abs_vergence_error_wrt_episode(data, ax):
    train_step = np.array([d["train_step"] for d in data])
    abs_vergence_errors = np.array([np.abs(vergence_error(d)) for d in data])
    args = np.argsort(train_step)
    train_step = train_step[args]
    abs_vergence_errors = abs_vergence_errors[args]
    for win in [200, 500, 1000, 2000, 10000]:
        mean_abs_vergence_errors = [np.mean(abs_vergence_errors[np.where(np.logical_and(train_step < x + win, train_step > x - win))]) for x in train_step]
        smooth = savgol_filter(mean_abs_vergence_errors, 21, 3)
        ax.plot(train_step, smooth, "-", lw=2, label="+-{}".format(win))
    ax.legend()


if __name__ == "__main__":
    path = "../experiments/tmp2/data/"
    data = get_data(path, -5)

    fig = plt.figure()
    ax_fine = fig.add_subplot(131)
    ax_coarse = fig.add_subplot(132)
    ax_both = fig.add_subplot(133)
    reward_wrt_vergence(data, 0.5, 5, ax_fine=ax_fine, ax_coarse=ax_coarse, ax_both=ax_both)
    plt.show()

    fig = plt.figure()
    ax_fine = fig.add_subplot(131)
    ax_coarse = fig.add_subplot(132)
    ax_both = fig.add_subplot(133)
    reward_wrt_critic_value(data, ax_fine=ax_fine, ax_coarse=ax_coarse, ax_both=ax_both)
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
    plt.show()

    data = get_data(path)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    vergence_error_wrt_episode(data, ax=ax)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    mean_abs_vergence_error_wrt_episode(data, ax=ax)
    plt.show()
