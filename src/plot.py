from scipy.stats import binned_statistic
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
import warnings
from asynchronous import Conf


warnings.filterwarnings("ignore")

keys = [
    "trajectory",
    "scale_rewards",
    "actions",
    "scale_critic_values",
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


def merge_flush(flush_list):
    dummy_key = list(flush_list[0].keys())[0]
    total_length = sum(len(f[dummy_key]) for f in flush_list)
    data = {k: np.zeros([total_length] + list(flush_list[0][k].shape[1:]), dtype=flush_list[0][k].dtype) for k in flush_list[0]}
    index = 0
    for f in flush_list:
        index_new = index + len(f[dummy_key])
        for k, v in f.items():
            data[k][index:index_new] = v
        index = index_new
    return data


def get_data(path, flush_id=None):
    return merge_flush(get_data_nonmerged(path, flush_id=flush_id))


def get_data_nonmerged(path, flush_id=None):
    if flush_id is None:
        worker_ids = get_worker_and_flush_ids(path)
        flush_id = list(worker_ids[0])
        flush_id.sort()
        return get_data_nonmerged(path, flush_id)
    elif type(flush_id) is int:
        worker_ids = get_worker_and_flush_ids(path)
        if flush_id < 0:
            flush_ids = list(worker_ids[0])
            flush_ids.sort()
            return get_data_nonmerged(path, flush_ids[flush_id:])
        data = []
        for worker_id in worker_ids:
            filename = "/worker_{:04d}_flush_{:04d}.pkl".format(worker_id, flush_id)
            with open(path + filename, "rb") as f:
                data.append(pickle.load(f))
        return data
    else:
        data = []
        for i in flush_id:
            data += get_data_nonmerged(path, i)
        return data


def group_by_trajectory(data):
    length = len(data["worker"])
    dtype = np.dtype([("worker", data["worker"].dtype), ("trajectory", data["trajectory"].dtype), ("iteration", data["iteration"].dtype)])
    vals = np.empty(length, dtype=dtype)
    vals["worker"] = data["worker"]
    vals["trajectory"] = data["trajectory"]
    vals["iteration"] = data["iteration"]
    args = np.argsort(vals, order=("trajectory", "worker", "iteration"))
    traj_length = np.max(data["iteration"]) + 1
    data = {k: v[args].reshape([-1, traj_length] + list(v.shape[1:])) for k, v in data.items()}
    return data


def vergence_error(eyes_positions, object_distances):
    vergences = eyes_positions[..., -1]
    return - np.degrees(np.arctan2(RESSOURCES.Y_EYES_DISTANCE, 2 * object_distances)) * 2 - vergences


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
    data = group_by_trajectory(data)
    n_trajectories = len(data["iteration"])
    fig = plt.figure()
    vergence_errors = vergence_error(data["eyes_position"], data["object_distance"])
    rewards = data["scale_rewards"]
    for ratio_index, r in enumerate(ratios):
        print("ratio", r)
        vergence_error_start = []
        vergence_error_end = []
        delta_reward = []
        for i in range(n_trajectories):
            for v0, r0 in zip(vergence_errors[i], rewards[i]):
                for v1, r1 in zip(vergence_errors[i], rewards[i]):
                    vergence_error_start.append(v0)
                    vergence_error_end.append(v1)
                    delta_reward.append(r1[ratio_index] - r0[ratio_index])
        print("ratio", r, "done")
        ax = fig.add_subplot(2, 4, ratio_index + 1, facecolor='grey')
        hexbin = ax.hexbin(vergence_error_start, vergence_error_end, delta_reward, cmap=seismic, norm=MidPointNorm(), gridsize=gridsize, mincnt=10)
        # hexbin = ax.hexbin(vergence_error_start, vergence_error_end, delta_reward, cmap=seismic, norm=None, reduce_C_function=np.std, gridsize=gridsize, mincnt=10)
        cb = fig.colorbar(hexbin, ax=ax)
        cb.set_label("Delta reward")
    if save:
        fig.savefig(plotpath + "/delta_reward_wrt_delta_vergence_all_scales.png")
    else:
        plt.show()
        plt.close(fig)
    fig = plt.figure()
    vergence_error_start = []
    vergence_error_end = []
    delta_reward = []
    rewards = data["total_reward"]
    for i in range(n_trajectories):
        for v0, r0 in zip(vergence_errors[i], rewards[i]):
            for v1, r1 in zip(vergence_errors[i], rewards[i]):
                vergence_error_start.append(v0)
                vergence_error_end.append(v1)
                delta_reward.append(r1 - r0)
    ax = fig.add_subplot(111, facecolor='grey')
    hexbin = ax.hexbin(vergence_error_start, vergence_error_end, delta_reward, cmap=seismic, norm=MidPointNorm(), gridsize=gridsize, mincnt=10)
    cb = fig.colorbar(hexbin, ax=ax)
    cb.set_label("Delta reward")
    if save:
        fig.savefig(plotpath + "/delta_reward_wrt_delta_vergence.png")
    else:
        plt.show()
        plt.close(fig)


def reward_wrt_vergence_fill_ax(ax, vergence_errors, rewards, title=None):
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
    ax.set_title(title)
    ax.set_xlabel("vergence error")
    ax.set_ylabel("reward")


def reward_wrt_vergence(data, min_dist, max_dist, save=False):
    fig = plt.figure()
    where = np.where(np.logical_and(data["object_distance"] < max_dist, data["object_distance"] > min_dist))
    vergence_errors = vergence_error(data["eyes_position"][where], data["object_distance"][where])
    for ratio_index, r in enumerate(ratios):
        ax = fig.add_subplot(2, 4, r)
        rewards = data["scale_rewards"][where, ratio_index]
        reward_wrt_vergence_fill_ax(ax, vergence_errors, rewards, title="ratio {}".format(r))
    if save:
        fig.savefig(plotpath + "/reward_wrt_vergence.png")
    else:
        plt.show()
    plt.close(fig)


def reward_wrt_vergence_all_scales(data, min_dist, max_dist, save=False):
    fig = plt.figure()
    where = np.where(np.logical_and(data["object_distance"] < max_dist, data["object_distance"] > min_dist))
    vergence_errors = vergence_error(data["eyes_position"][where], data["object_distance"][where])
    ax = fig.add_subplot(1, 1, 1)
    rewards = data["total_reward"][where]
    reward_wrt_vergence_fill_ax(ax, vergence_errors, rewards, title="all scales")
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
    where = np.where(np.logical_and(data["object_distance"] < max_dist, data["object_distance"] > min_dist))
    vergence_errors = vergence_error(data["eyes_position"][where], data["object_distance"][where])
    for ratio_index, r in enumerate(ratios):
        ax = fig.add_subplot(2, 4, r)
        critic = data["scale_critic_values"][where, ratio_index]
        critic_wrt_vergence_fill_ax(ax, vergence_errors, critic,
                                           title="ratio {}".format(r))
    if save:
        fig.savefig(plotpath + "/critic_wrt_vergence.png")
    else:
        plt.show()
    plt.close(fig)


def critic_wrt_vergence_all_scales(data, min_dist, max_dist, save=False):
    fig = plt.figure()
    where = np.where(np.logical_and(data["object_distance"] < max_dist, data["object_distance"] > min_dist))
    vergence_errors = vergence_error(data["eyes_position"][where], data["object_distance"][where])
    ax = fig.add_subplot(1, 1, 1)
    critic = np.sum(data["scale_critic_values"][where, ratio_index], axis=-1)
    critic_wrt_vergence_fill_ax(ax, vergence_errors, critic, title="all scales")
    if save:
        fig.savefig(plotpath + "/critic_wrt_vergence_all_scales.png")
    else:
        plt.show()
    plt.close(fig)


def action_wrt_vergence_fill_ax(ax, vergence_errors, action, title=None):
    ax.hist2d(vergence_errors, action, bins=[75, 75], cmin=1)  # , cmax=50)
    # ax.plot(vergence_errors, action, ".")
    ax.set_title(title)
    ax.set_xlabel("vergence error")
    ax.set_ylabel("action")


def action_wrt_vergence(data, min_dist, max_dist, greedy=False, save=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    where = np.where(np.logical_and(data["object_distance"] < max_dist, data["object_distance"] > min_dist))
    vergence_errors = vergence_error(data["eyes_position"][where], data["object_distance"][where])
    action_key = "greedy_actions_indices" if greedy else "sampled_actions_indices"
    actions = data[action_key][where, -1][0]
    n_bins = 50
    bins = np.linspace(np.min(vergence_errors), np.max(vergence_errors), n_bins)
    args = np.digitize(vergence_errors, bins)
    probs = []
    for i in range(1, n_bins):
        counts = np.bincount(actions[np.where(args==i)], minlength=n_actions_per_joint)
        # probs.append(counts)
        tmp = counts / np.sum(counts)
        # tmp[4] = 0
        probs.append(tmp)
        # print("\n")
    im = ax.imshow(np.array(probs)[:, ::-1].T, extent=(bins[0], bins[-1], -(n_actions_per_joint // 2), n_actions_per_joint // 2), interpolation="none", vmin=0, vmax=1)
    fig.colorbar(im)
    ax.set_title("Action wrt vergence error")
    ax.set_xlabel("vergence error")
    ax.set_ylabel("action")
    # actions = [d["greedy_actions_indices"]["vergence"][0] for d in data]
    # actions_val = np.asarray(action_set)[actions]
    # ax.plot([action_set[0], action_set[-1]], [action_set[-1], action_set[0]], "k-")
    # action_wrt_vergence_fill_ax(ax, vergence_errors, actions_val, title="action wrt vergence error")
    if save:
        filename = "/action_wrt_vergence_greedy.png" if greedy else "/action_wrt_vergence_sample.png"
        fig.savefig(plotpath + filename)
    else:
        plt.show()
    plt.close(fig)


def return_wrt_critic_value_fill_ax(ax, critic, reward, title=None):
    ax.plot(critic, reward, ".", alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel("Critic value")
    ax.set_ylabel("Reward")
    ax.axis('scaled')
    slope, intercept, r_value, p_value, std_err = linregress(critic, reward)
    X = np.linspace(min(critic), max(critic))
    Y = slope * X + intercept
    ax.plot(X, Y, "k")
    ax.text(1, 1, "corr: {:.2f}".format(r_value), horizontalalignment="right",
        transform=ax.transAxes)


def to_return_1d(rewards, discount_factor, axis_=-1):
    ret = np.zeros_like(rewards)
    prev = 0
    for i, r in reversed(list(enumerate(rewards))):
        prev = prev * discount_factor + r
        ret[i] = prev
    return ret


def to_return(rewards, discount_factor, start=None, axis=-1):
    returns = np.apply_along_axis(to_return_1d, axis=axis, arr=rewards, discount_factor=discount_factor, axis_=axis)
    if start is not None:
        start_shape = np.array(start.shape)
        rewards_shape = np.array(rewards.shape)
        assert(
            rewards.ndim == start.ndim and
            (start_shape[:axis] == rewards_shape[:axis]).all() and
            (start_shape[axis + 1:] == rewards_shape[axis + 1:]).all() and
            start_shape[axis] == 1
        )
        gammas = np.zeros(rewards.shape[axis], dtype=np.float32)
        gammas[:] = discount_factor
        shape = np.array(rewards.shape)
        gammas = np.cumprod(gammas)[::-1]
        shape[:axis] = 1
        shape[axis + 1:] = 1
        gammas = gammas.reshape(shape)
        returns += start * gammas
    return returns


def return_wrt_critic_value(data, discount_factor, save=False):
    data = group_by_trajectory(data)
    critic, returns = [], []
    critic_sum, returns_sum = [], []
    n_trajectories = len(data["iteration"])
    for i in range(n_trajectories):
        critic_trajectory = data["scale_critic_values"][i]
        critic_sum_trajectory = data["critic_values"][i]
        returns_trajectory = to_return(data["scale_rewards"][i], discount_factor, start=critic_trajectory[-1, np.newaxis], axis=0)
        return_total_trajectory = to_return(data["total_reward"][i], discount_factor, start=critic_sum_trajectory[-1, np.newaxis], axis=0)
        for j in range(len(critic_trajectory) - 1):
            critic.append(critic_trajectory[j])
            returns.append(returns_trajectory[j + 1])
            critic_sum.append(critic_sum_trajectory[j])
            returns_sum.append(return_total_trajectory[j + 1])
    fig = plt.figure()
    returns = np.array(returns)
    critic = np.array(critic)
    for ratio_index, r in enumerate(ratios):
        ax = fig.add_subplot(2, 4, r)
        return_wrt_critic_value_fill_ax(ax, critic[:, ratio_index], returns[:, ratio_index],
                                        title="ratio {}".format(r))
    if save:
        fig.savefig(plotpath + "/return_wrt_critic.png")
    else:
        plt.show()
    plt.close(fig)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    return_wrt_critic_value_fill_ax(ax, critic_sum, returns_sum,
                                    title="all scales")
    if save:
        fig.savefig(plotpath + "/return_wrt_critic_all_scales.png")
    else:
        plt.show()


def vergence_error_episode_end_wrt_episode(data, save=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # sequence_length = max([d["iteration"] for d in data])
    # data = [d for d in data if d["iteration"] == sequence_length]
    data = group_by_trajectory(data)
    data = {k: v[:, -1] for k, v in data.items()}
    trajectory = data["trajectory"]
    vergence_errors = vergence_error(data["eyes_position"], data["object_distance"])
    ax.plot(trajectory, vergence_errors, ".", alpha=0.8)
    ax.set_xlabel("episode")
    ax.set_ylabel("vergence error in degrees")
    ax.set_title("vergence error wrt time")
    if save:
        fig.savefig(plotpath + "/vergence_error_wrt_time_2.png")
    else:
        plt.show()
    plt.close(fig)


def mean_abs_vergence_error_episode_end_wrt_episode(data, save=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # sequence_length = max([d["iteration"] for d in data])
    # data = [d for d in data if d["iteration"] == sequence_length]
    data = group_by_trajectory(data)
    data = {k: v[:, -1] for k, v in data.items()}
    trajectory = data["trajectory"]
    vergence_errors = vergence_error(data["eyes_position"], data["object_distance"])
    abs_vergence_errors = np.array(np.abs(vergence_errors))
    args = np.argsort(trajectory)
    trajectory = trajectory[args]
    abs_vergence_errors = abs_vergence_errors[args]
    for win in [200, 500, 1000]:
        mean_abs_vergence_errors = [np.mean(abs_vergence_errors[np.where(np.logical_and(trajectory < x + win, trajectory > x - win))]) for x in trajectory]
        smooth = savgol_filter(mean_abs_vergence_errors, 21, 3)
        ax.plot(trajectory, smooth, "-", lw=2, label="+-{}".format(win))
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
    # sequence_length = max([d["iteration"] for d in data])
    # data = [d for d in data if d["iteration"] == sequence_length]
    data = group_by_trajectory(data)
    data = {k: v[:, -1] for k, v in data.items()}
    trajectory = data["trajectory"]
    vergence = data["eyes_position"][:, -1]
    args = np.argsort(trajectory)
    trajectory = trajectory[args]
    vergence = vergence[args]
    ax.scatter(trajectory, vergence, alpha=0.1)
    ax.set_xlabel("episode")
    ax.set_ylabel("vergence position in degrees")
    ax.set_title("vergence wrt time")
    if save:
        fig.savefig(plotpath + "/vergence_wrt_time.png")
    else:
        plt.show()
    plt.close(fig)


def vergence_wrt_object_distance(data, save=False):
    data = group_by_trajectory(data)
    data = {k: v[:, -1] for k, v in data.items()}
    trajectory = data["trajectory"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    object_distance = data["object_distance"]
    vergence = data["eyes_position"][:, -1]
    rewards = data["total_reward"]
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


quick_and_dirty = 0

def target_wrt_delta_vergence(data, discount_factor, save=False):
    data = group_by_trajectory(data)
    vergence_errors = [[] for i in range(9)]
    target = [[] for i in range(9)]
    n_trajectories = len(data["iteration"])
    for i in range(n_trajectories):
        critic_sum_trajectory = data["critic_values"][i]
        return_total_trajectory = to_return(data["total_reward"][i], discount_factor, start=critic_sum_trajectory[-1, np.newaxis], axis=0)
        for j in range(data["iteration"].shape[1] - 1):
            action = data["sampled_actions_indices"][i][j, -1]
            tmp_target = return_total_trajectory[j + 1] - data["critic_values"][i][j]
            # tmp_target = data["total_reward"][i][j + 1] - data["total_reward"][i][j]
            target[action].append(tmp_target)
            vergence_errors[action].append(vergence_error(data["eyes_position"][i][j], data["object_distance"][i][j]))
    fig = plt.figure()
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    all_bin_means = []
    all_bin_centers = []
    bins = 50
    for i in range(9):
        ax = fig.add_subplot(3, 3, i + 1)
        ax.set_title("Action id {}".format(i))
        if len(vergence_errors[i]) > 0:
            bin_means, bin_edges, binnumber = binned_statistic(vergence_errors[i], target[i], bins=bins)
            bin_stds, _1, _2 = binned_statistic(vergence_errors[i], target[i], bins=bins, statistic=np.std)
            bin_width = (bin_edges[1] - bin_edges[0])
            bin_centers = bin_edges[1:] - bin_width / 2
            all_bin_means.append(bin_means)
            all_bin_centers.append(bin_centers)
            ax.fill_between(bin_centers, bin_means - bin_stds, bin_means + bin_stds, color="b", alpha=0.2)
            ax.plot(bin_centers, bin_means, "b", linewidth=2.5)
            where = np.where(bin_means > 0)
            ax2.plot(bin_centers[where], bin_means[where], linewidth=2.5)
            ax2.set_ylim([-0.1, 0.4])
            ax2.set_xlim([-3, 3])
            ylim = ax.get_ylim()
            ax.scatter(vergence_errors[i], target[i], alpha=1, s=1, c="g")
            ax.axvline(action_set[i], c="r")
            ax.axhline(0, c="k")
            ax.set_ylim([-0.5, 0.5])
    n = 0
    for i in range(9):
        action = action_set[i]
        means = np.array([np.interp(action, bin_centers, bin_means) for bin_centers, bin_means in zip(all_bin_centers, all_bin_means)])
        # print(i, np.argmax(means), np.argmax(means) == i)
        if np.argmax(means) == i:
            n += 1
    print(n)
    if save:
        global quick_and_dirty
        fig.savefig(plotpath + "/target_wrt_delta_vergence.png")
        fig2.savefig(plotpath + "/target_wrt_delta_vergence_2_{:04d}.png".format(quick_and_dirty))
        quick_and_dirty += 1
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path', metavar="PATH",
        type=str,
        nargs="+",
        action='store',
        help="Path to the data."
    )

    parser.add_argument(
        '-s', '--save',
        action='store_true',
        help="If turned on, saves the plots."
    )

    parser.add_argument(
        '-n', '--name',
        type=str,
        help="Outdir name",
        default="default"
    )

    args = parser.parse_args()
    paths = args.path


    for path in paths:
        print("\n", path, "\n")
        ### CONSTANTS
        plotpath = path + "/../plots_{}/".format(args.name)
        with open(path + "/../conf/worker_conf.pkl", "rb") as f:
            conf = pickle.load(f)
            discount_factor = conf.discount_factor
        ratios = list(range(1, 9))
        n_actions_per_joint = 9
        n = n_actions_per_joint // 2
        mini = 0.28
        maxi = 0.28 * 2 ** (n - 1)
        positive = np.logspace(np.log2(mini), np.log2(maxi), n, base=2)
        negative = -positive[::-1]
        action_set = np.concatenate([negative, [0], positive])

        data = get_data(path, -5)
        if args.save:
            os.mkdir(plotpath)

        # reward_wrt_vergence_all_scales(data, 0.5, 5, args.save)
        # reward_wrt_vergence(data, 0.5, 5, args.save)
        # critic_wrt_vergence_all_scales(data, 1, 4, args.save)
        # critic_wrt_vergence(data, 2, 3, args.save)

        # for i in range(100):
        #     data = get_data(path, i)
        #     print("flush  ", i)
        #     target_wrt_delta_vergence(data, args.save)

        print("return_wrt_critic_value:")
        return_wrt_critic_value(data, discount_factor, args.save)
        print("target_wrt_delta_vergence:")
        target_wrt_delta_vergence(data, discount_factor, args.save)
        print("action_wrt_vergence:")
        action_wrt_vergence(data, 0.5, 5, greedy=False, save=args.save)
        print("action_wrt_vergence:")
        action_wrt_vergence(data, 0.5, 5, greedy=True, save=args.save)
        print("vergence_wrt_object_distance:")
        vergence_wrt_object_distance(data, args.save)
        # delta_reward_wrt_delta_vergence(data, args.save)

        data = get_data(path)

        print("vergence_error_episode_end_wrt_episode:")
        vergence_error_episode_end_wrt_episode(data, args.save)
        print("vergence_episode_end_wrt_episode:")
        vergence_episode_end_wrt_episode(data, args.save)
        print("mean_abs_vergence_error_episode_end_wrt_episode:")
        mean_abs_vergence_error_episode_end_wrt_episode(data, args.save)
