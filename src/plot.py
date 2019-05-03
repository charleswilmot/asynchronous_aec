from numpy import ma
from matplotlib import cbook
from matplotlib.cm import seismic
from matplotlib.colors import ListedColormap, Normalize
from collections import defaultdict
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


def group_by_train_step(data):
    ret = defaultdict(list)
    for d in data:
        ret[(d["worker"], d["train_step"])].append(d)
    return ret


def vergence_error(d):
    vergence = np.squeeze(d["eyes_position"][-1])
    object_distance = d["object_distance"]
    return vergence + np.degrees(np.arctan2(RESSOURCES.Y_EYES_DISTANCE, 2 * object_distance)) * 2


class MidPointNorm(Normalize):
    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):
        Normalize.__init__(self,vmin, vmax, clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if not (vmin < midpoint < vmax):
            raise ValueError("midpoint must be between maxvalue and minvalue.")
        elif vmin == vmax:
            result.fill(0) # Or should it be all masked? Or 0.5?
        elif vmin > vmax:
            raise ValueError("maxvalue must be bigger than minvalue")
        else:
            vmin = float(vmin)
            vmax = float(vmax)
            if clip:
                mask = ma.getmask(result)
                result = ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                  mask=mask)

            # ma division is very slow; we can take a shortcut
            resdat = result.data

            #First scale to -1 to 1 range, than to from 0 to 1.
            resdat -= midpoint
            resdat[resdat>0] /= abs(vmax - midpoint)
            resdat[resdat<0] /= abs(vmin - midpoint)

            resdat /= 2.
            resdat += 0.5
            result = ma.array(resdat, mask=result.mask, copy=False)

        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if cbook.iterable(value):
            val = ma.asarray(value)
            val = 2 * (val-0.5)
            val[val>0]  *= abs(vmax - midpoint)
            val[val<0] *= abs(vmin - midpoint)
            val += midpoint
            return val
        else:
            val = 2 * (val - 0.5)
            if val < 0:
                return  val*abs(vmin-midpoint) + midpoint
            else:
                return  val*abs(vmax-midpoint) + midpoint


def delta_reward_wrt_delta_vergence(data, save=False):
    gridsize = 100
    data = group_by_train_step(data)
    fig = plt.figure()
    for r in ratios:
        vergence_error_start = []
        vergence_error_end = []
        delta_reward = []
        for key in data:
            vergence_errors = [vergence_error(d) for d in data[key]]
            rewards = [d["rewards"][r][0] for d in data[key]]
            for v0, r0 in zip(vergence_errors, rewards):
                for v1, r1 in zip(vergence_errors, rewards):
                    vergence_error_start.append(v0)
                    vergence_error_end.append(v1)
                    delta_reward.append(r1 - r0)
        ax = fig.add_subplot(2, 4, r, axisbg='grey')
        hexbin = ax.hexbin(vergence_error_start, vergence_error_end, delta_reward, cmap=seismic, norm=MidPointNorm(), gridsize=gridsize, mincnt=10)
        # hexbin = ax.hexbin(vergence_error_start, vergence_error_end, delta_reward, cmap=seismic, norm=None, reduce_C_function=np.std, gridsize=gridsize, mincnt=10)
        cb = fig.colorbar(hexbin, ax=ax)
        cb.set_label("Delta reward")
    if save:
        fig.savefig(plotpath + "/delta_reward_wrt_delta_vergence.png")
    else:
        plt.show()
        plt.close(fig)
    fig = plt.figure()
    for key in data:
        vergence_errors = [vergence_error(d) for d in data[key]]
        rewards = [sum([d["rewards"][r][0] for r in ratios]) for d in data[key]]
        for v0, r0 in zip(vergence_errors, rewards):
            for v1, r1 in zip(vergence_errors, rewards):
                vergence_error_start.append(v0)
                vergence_error_end.append(v1)
                delta_reward.append(r1 - r0)
    ax = fig.add_subplot(111, axisbg='grey')
    hexbin = ax.hexbin(vergence_error_start, vergence_error_end, delta_reward, cmap=seismic, norm=MidPointNorm(), gridsize=gridsize, mincnt=10)
    cb = fig.colorbar(hexbin, ax=ax)
    cb.set_label("Delta reward")
    if save:
        fig.savefig(plotpath + "/delta_reward_wrt_delta_vergence_all_scales.png")
    else:
        plt.show()
        plt.close(fig)


def reward_wrt_vergence_fill_ax(ax, vergence_errors, rewards, iterations, title=None):
    N = 150
    X = np.linspace(min(vergence_errors), max(vergence_errors), N)
    Y = np.linspace(min(rewards), max(rewards), N)
    probs = np.zeros((N, N))
    for i, x in enumerate(X):
        weights = np.exp(-5 * np.abs(vergence_errors - x))
        weights /= np.sum(weights)
        kde = gaussian_kde(rewards, weights=weights)
        probs[:, i] = kde(Y)
    ax.contourf(X, Y, probs)
    mean = np.sum(probs * Y[:, np.newaxis], axis=0) / np.sum(probs, axis=0)
    ax.plot(X, mean, "r-")
    ax.plot(X, Y[np.argmax(probs, axis=0)], "k-")
    ax.axvline(0, c="k")
    # ax.scatter(vergence_errors, rewards, c=iterations, cmap="Greys")  # "white"  , marker=","  , alpha=1   , s=1
    ax.set_title(title)
    ax.set_xlabel("vergence error")
    ax.set_ylabel("reward")


def reward_wrt_vergence(data, min_dist, max_dist, save=False):
    fig = plt.figure()
    data = [d for d in data if min_dist < d["object_distance"] < max_dist]
    vergence_errors = [vergence_error(d) for d in data]
    iterations = np.array([d["iteration"] for d in data]).astype(np.float32)
    # iterations /= np.max(iterations)
    for r in ratios:
        ax = fig.add_subplot(2, 4, r)
        rewards = [d["rewards"][r][0] for d in data]
        reward_wrt_vergence_fill_ax(ax, vergence_errors, rewards, iterations,
                                           title="ratio {}".format(r))
    if save:
        fig.savefig(plotpath + "/reward_wrt_vergence.png")
    else:
        plt.show()
    plt.close(fig)


def reward_wrt_vergence_all_scales(data, min_dist, max_dist, save=False):
    fig = plt.figure()
    data = [d for d in data if min_dist < d["object_distance"] < max_dist]
    vergence_errors = [vergence_error(d) for d in data]
    iterations = np.array([d["iteration"] for d in data]).astype(np.float32)
    # iterations /= np.max(iterations)
    ax = fig.add_subplot(1, 1, 1)
    rewards = [sum([d["rewards"][r][0] for r in ratios]) for d in data]
    reward_wrt_vergence_fill_ax(ax, vergence_errors, rewards, iterations, title="all scales")
    if save:
        fig.savefig(plotpath + "/reward_wrt_vergence_all_scales.png")
    else:
        plt.show()
    plt.close(fig)


def critic_wrt_vergence_fill_ax(ax, vergence_errors, critic, title=None):
    N = 150
    X = np.linspace(min(vergence_errors), max(vergence_errors), N)
    Y = np.linspace(min(critic), max(critic), N)
    probs = np.zeros((N, N))
    for i, x in enumerate(X):
        weights = np.exp(-5 * np.abs(vergence_errors - x))
        weights /= np.sum(weights)
        kde = gaussian_kde(critic, weights=weights)
        probs[:, i] = kde(Y)
    ax.contourf(X, Y, probs)
    mean = np.sum(probs * Y[:, np.newaxis], axis=0) / np.sum(probs, axis=0)
    ax.plot(X, mean, "r-")
    ax.plot(X, Y[np.argmax(probs, axis=0)], "k-")
    ax.axvline(0, c="k")
    ax.scatter(vergence_errors, critic, alpha=1, marker=",", color="white", s=1)
    ax.set_title(title)
    ax.set_xlabel("vergence error")
    ax.set_ylabel("critic value")


def critic_wrt_vergence(data, min_dist, max_dist, save=False):
    fig = plt.figure()
    data = [d for d in data if min_dist < d["object_distance"] < max_dist]
    vergence_errors = [vergence_error(d) for d in data]
    for r in ratios:
        ax = fig.add_subplot(2, 4, r)
        critic = [d["critic_values"][r][0][0] for d in data]
        critic_wrt_vergence_fill_ax(ax, vergence_errors, critic,
                                           title="ratio {}".format(r))
    if save:
        fig.savefig(plotpath + "/critic_wrt_vergence.png")
    else:
        plt.show()
    plt.close(fig)


def critic_wrt_vergence_all_scales(data, min_dist, max_dist, save=False):
    fig = plt.figure()
    data = [d for d in data if min_dist < d["object_distance"] < max_dist]
    vergence_errors = [vergence_error(d) for d in data]
    ax = fig.add_subplot(1, 1, 1)
    critic = [sum([d["critic_values"][r][0][0] for r in ratios]) for d in data]
    critic_wrt_vergence_fill_ax(ax, vergence_errors, critic, title="all scales")
    if save:
        fig.savefig(plotpath + "/critic_wrt_vergence_all_scales.png")
    else:
        plt.show()
    plt.close(fig)


def action_wrt_vergence_fill_ax(ax, vergence_errors, action, title=None):
    ax.hist2d(vergence_errors, action, bins=[25, 25], cmin=1)  # , cmax=50)
    # ax.plot(vergence_errors, action, ".")
    ax.set_title(title)
    ax.set_xlabel("vergence error")
    ax.set_ylabel("action")


def action_wrt_vergence(data, min_dist, max_dist, save=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data = [d for d in data if min_dist < d["object_distance"] < max_dist]
    vergence_errors = [vergence_error(d) for d in data]
    actions = [d["sampled_actions_indices"]["vergence"][0] for d in data]
    # actions = [d["greedy_actions_indices"]["vergence"][0] for d in data]
    actions = np.asarray(action_set)[actions]
    ax.plot([action_set[0], action_set[-1]], [action_set[0], action_set[-1]], "k-")
    action_wrt_vergence_fill_ax(ax, vergence_errors, actions, title="action wrt vergence error")
    if save:
        fig.savefig(plotpath + "/action_wrt_vergence.png")
    else:
        plt.show()
    plt.close(fig)


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


def reward_wrt_critic_value(data, save=False):
    fig = plt.figure()
    critic = [d["critic_values"] for d in data]
    rewards = [d["rewards"] for d in data]
    for r in ratios:
        ax = fig.add_subplot(2, 4, r)
        reward_wrt_critic_value_fill_ax(ax, [c[r][0, 0] for c in critic], [rew[r][0] for rew in rewards],
                                        title="ratio {}".format(r))
    if save:
        fig.savefig(plotpath + "/reward_wrd_critic.png")
    else:
        plt.show()
    plt.close(fig)


def vergence_error_episode_end_wrt_episode(data, save=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sequence_length = max([d["iteration"] for d in data])
    data = [d for d in data if d["iteration"] == sequence_length]
    train_step = [d["train_step"] for d in data]
    vergence_errors = [vergence_error(d) for d in data]
    ax.plot(train_step, vergence_errors, ".", alpha=0.8)
    ax.set_xlabel("episode")
    ax.set_ylabel("vergence error in degrees")
    ax.set_title("vergence error wrt time")
    if save:
        fig.savefig(plotpath + "/vergence_error_wrt_time.png")
    else:
        plt.show()
    plt.close(fig)


def mean_abs_vergence_error_episode_end_wrt_episode(data, save=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sequence_length = max([d["iteration"] for d in data])
    data = [d for d in data if d["iteration"] == sequence_length]
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
    if save:
        fig.savefig(plotpath + "/vergence_error_wrt_time.png")
    else:
        plt.show()
    plt.close(fig)


def vergence_episode_end_wrt_episode(data, save=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sequence_length = max([d["iteration"] for d in data])
    data = [d for d in data if d["iteration"] == sequence_length]
    train_step = np.array([d["train_step"] for d in data])
    vergence = np.array([np.squeeze(d["eyes_position"][-1]) for d in data])
    args = np.argsort(train_step)
    train_step = train_step[args]
    vergence = vergence[args]
    ax.scatter(train_step, vergence, alpha=0.1)
    ax.set_xlabel("episode")
    ax.set_ylabel("vergence position in degrees")
    ax.set_title("vergence wrt time")
    if save:
        fig.savefig(plotpath + "/vergence_wrt_time.png")
    else:
        plt.show()
    plt.close(fig)


def vergence_wrt_object_distance(data, save=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    object_distance = np.array([d["object_distance"] for d in data])
    vergence = np.array([np.squeeze(d["eyes_position"][-1]) for d in data])
    rewards = np.array([sum([d["rewards"][r][0] for r in ratios]) for d in data])
    X = np.linspace(np.min(object_distance), np.max(object_distance), 100)
    correct = -np.degrees(np.arctan2(RESSOURCES.Y_EYES_DISTANCE, 2 * X)) * 2
    ax.scatter(object_distance, vergence, c=rewards, cmap="Greys")
    ax.plot(X, correct, "k-")
    ax.set_xlabel("object position in meters")
    ax.set_ylabel("vergence position in degrees")
    ax.set_title("final vergence wrt object position")
    if save:
        fig.savefig(plotpath + "/vergence_wrt_object_distance.png")
    else:
        plt.show()
    plt.close(fig)


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

    ### CONSTANTS
    plotpath = path + "/../plots/"
    ratios = list(range(1, 9))
    n_actions_per_joint = 9
    n = n_actions_per_joint // 2
    mini = 0.28
    maxi = 0.28 * 2 ** (n - 1)
    positive = np.logspace(np.log2(mini), np.log2(maxi), n, base=2)
    negative = -positive[::-1]
    action_set = np.concatenate([negative, [0], positive])

    data = get_data(path, -1)
    if args.save:
        os.mkdir(plotpath)

    ### reward_wrt_vergence_all_scales(data, 1, 4, args.save)
    ### reward_wrt_vergence(data, 2, 3, args.save)
    ### critic_wrt_vergence_all_scales(data, 1, 4, args.save)
    ### critic_wrt_vergence(data, 2, 3, args.save)
    delta_reward_wrt_delta_vergence(data, args.save)
    vergence_wrt_object_distance(data, args.save)
    reward_wrt_critic_value(data, args.save)
    action_wrt_vergence(data, 0.5, 5, args.save)

    data = get_data(path)
    vergence_error_episode_end_wrt_episode(data, args.save)
    vergence_episode_end_wrt_episode(data, args.save)
    mean_abs_vergence_error_episode_end_wrt_episode(data, args.save)
