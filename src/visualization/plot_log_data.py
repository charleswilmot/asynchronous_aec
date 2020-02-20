from numpy import ma
from matplotlib import cbook
from matplotlib.cm import seismic
from matplotlib.colors import Normalize
from helper.utils import *
from scipy.signal import savgol_filter
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import re
import warnings

warnings.filterwarnings("ignore")

keys = [
    "episode_number",
    "scale_rewards",
    "actions",
    "scale_critic_values",
    "object_distance",
    "object_speed",
    "object_position",
    "eyes_position",
    "eyes_speed",
]


def group_by_episode(data):
    length = len(data["worker"])
    dtype = np.dtype([("worker", data["worker"].dtype), ("episode_number", data["episode_number"].dtype), ("iteration", data["iteration"].dtype)])
    vals = np.empty(length, dtype=dtype)
    vals["worker"] = data["worker"]
    vals["episode_number"] = data["episode_number"]
    vals["iteration"] = data["iteration"]
    args = np.argsort(vals, order=("episode_number", "worker", "iteration"))
    traj_length = np.max(data["iteration"]) + 1
    data = {k: v[args].reshape([-1, traj_length] + list(v.shape[1:])) for k, v in data.items()}
    return data


def vergence_error(eyes_positions, object_distances):
    vergences = eyes_positions[..., -1]
    return to_angle(object_distances) - vergences


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


def check_data(data, save=False):
    data = group_by_episode(data)
    # print(data.keys())
    # print(data['eyes_speed'])


def delta_reward_wrt_delta_vergence(data, save=False):
    gridsize = 100
    data = group_by_episode(data)
    n_trajectories = len(data["iteration"])
    fig = plt.figure()
    vergence_errors = vergence_error(data["eyes_position"], data["object_distance"])
    rewards = data["scale_rewards"]
    # for ratio_index, r in enumerate(ratios):
    #     print("ratio", r)
    #     vergence_error_start = []
    #     vergence_error_end = []
    #     delta_reward = []
    #     for i in range(n_trajectories):
    #         for v0, r0 in zip(vergence_errors[i], rewards[i]):
    #             for v1, r1 in zip(vergence_errors[i], rewards[i]):
    #                 vergence_error_start.append(v0)
    #                 vergence_error_end.append(v1)
    #                 delta_reward.append(r1[ratio_index] - r0[ratio_index])
    #     print("ratio", r, "done")
    #     ax = fig.add_subplot(2, 4, ratio_index + 1)
    #     hexbin = ax.hexbin(vergence_error_start, vergence_error_end, delta_reward, cmap=seismic, norm=MidPointNorm(), gridsize=gridsize, mincnt=10)
    #     # hexbin = ax.hexbin(vergence_error_start, vergence_error_end, delta_reward, cmap=seismic, norm=None, reduce_C_function=np.std, gridsize=gridsize, mincnt=10)
    #     cb = fig.colorbar(hexbin, ax=ax)
    #     cb.set_label("Delta reward")
    # if save:
    #     fig.savefig(plotpath + "/delta_reward_wrt_delta_vergence_all_scales.png")
    # else:
    #     plt.show()
    #     plt.close(fig)
    # fig = plt.figure()
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
    hexbin = ax.hexbin(vergence_error_start, vergence_error_end, delta_reward, cmap=seismic, norm=MidPointNorm(), gridsize=gridsize, mincnt=10, vmin=-1, vmax=1)
    cb = fig.colorbar(hexbin, ax=ax)
    cb.set_label("Delta reward")
    if save:
        fig.savefig(plotpath + "/delta_reward_wrt_delta_vergence.png")
    else:
        plt.show()
        plt.close(fig)

def delta_reward_wrt_delta_vergence_2(data, save=False):
    gridsize = 100
    data = group_by_episode(data)
    n_trajectories = len(data["iteration"])
    fig = plt.figure()
    vergence_errors = vergence_error(data["eyes_position"], data["object_distance"])
    rewards = data["scale_rewards"]
    # for ratio_index, r in enumerate(ratios):
    #     print("ratio", r)
    #     vergence_error_start = []
    #     vergence_error_end = []
    #     delta_reward = []
    #     for i in range(n_trajectories):
    #         for v0, r0 in zip(vergence_errors[i], rewards[i]):
    #             for v1, r1 in zip(vergence_errors[i], rewards[i]):
    #                 vergence_error_start.append(v0)
    #                 vergence_error_end.append(v1)
    #                 delta_reward.append(r1[ratio_index] - r0[ratio_index])
    #     print("ratio", r, "done")
    #     ax = fig.add_subplot(2, 4, ratio_index + 1)
    #     hexbin = ax.hexbin(vergence_error_start, vergence_error_end, delta_reward, cmap=seismic, norm=MidPointNorm(), gridsize=gridsize, mincnt=10)
    #     # hexbin = ax.hexbin(vergence_error_start, vergence_error_end, delta_reward, cmap=seismic, norm=None, reduce_C_function=np.std, gridsize=gridsize, mincnt=10)
    #     cb = fig.colorbar(hexbin, ax=ax)
    #     cb.set_label("Delta reward")
    # if save:
    #     fig.savefig(plotpath + "/delta_reward_wrt_delta_vergence_all_scales.png")
    # else:
    #     plt.show()
    #     plt.close(fig)
    # fig = plt.figure()
    vergence_error_start = []
    vergence_error_end = []
    reward_start = []
    reward_end = []
    delta_reward = []
    delta_vergence = []
    count_reward_dict = {}
    rewards = data["total_reward"]
    for i in range(n_trajectories):
        for v0, r0 in zip(vergence_errors[i], rewards[i]):
            for v1, r1 in zip(vergence_errors[i], rewards[i]):
                vergence_error_start.append(v0)
                vergence_error_end.append(v1)
                delta_vergence.append(abs(v0) - abs(v1))
                d_r = (r1 - r0)
                delta_reward.append(d_r)
                reward_start.append(r0)
                reward_end.append(r1)
                if d_r in count_reward_dict:
                    count_reward_dict[d_r] += 1
                else:
                    count_reward_dict[d_r] = 1
    count_reward = []
    for d_r in delta_reward:
        count_reward.append(count_reward_dict[d_r])
    ax = fig.add_subplot(111)#, facecolor='grey')
    hexbin = ax.hexbin(delta_vergence, delta_reward, count_reward, cmap=seismic, norm=MidPointNorm(), gridsize=gridsize, mincnt=10, vmin=-30, vmax=30)
    cb = fig.colorbar(hexbin, ax=ax)
    cb.set_label("Delta reward")
    if save:
        fig.savefig(plotpath + "/delta_reward_wrt_delta_vergence.png")
    else:
        plt.show()
        plt.close(fig)



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
    im = ax.imshow(np.array(probs)[:, ::-1].T, extent=(bins[0], bins[-1], -(n_actions_per_joint // 2) - 0.5, n_actions_per_joint // 2 + 0.5), interpolation="none", vmin=0, vmax=1)
    fig.colorbar(im)
    ax.set_title("Action wrt vergence error")
    ax.set_xlabel("vergence error")
    ax.set_ylabel("action")
    ax.plot(action_set, np.arange(-4, 5), "k-", alpha=0.3)
    if save:
        filename = "/action_wrt_vergence_greedy.png" if greedy else "/action_wrt_vergence_sample.png"
        fig.savefig(plotpath + filename)
    else:
        plt.show()
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(np.log(np.array(probs)[:, ::-1].T + 0.03), extent=(bins[0], bins[-1], -(n_actions_per_joint // 2) - 0.5, n_actions_per_joint // 2 + 0.5), interpolation="none", vmin=-3, vmax=0)
    fig.colorbar(im)
    ax.set_title("Action wrt vergence error")
    ax.set_xlabel("vergence error")
    ax.set_ylabel("action")
    ax.plot(action_set, np.arange(-4, 5), "k-", alpha=0.3)
    if save:
        filename = "/action_wrt_vergence_greedy_log.png" if greedy else "/action_wrt_vergence_sample_log.png"
        fig.savefig(plotpath + filename)
    else:
        plt.show()
    plt.close(fig)


def action_wrt_vergence_based_on_critic(data, min_dist, max_dist, scale, save=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    where = np.where(np.logical_and(data["object_distance"] < max_dist, data["object_distance"] > min_dist))
    vergence_errors = vergence_error(data["eyes_position"][where], data["object_distance"][where])
    actions = np.argmax(data["scale_values_vergence"][where][:, scale], axis=-1)
    # actions = np.argmax(data["critic_values_vergence"][where], axis=-1)
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
    im = ax.imshow(np.array(probs)[:, ::-1].T, extent=(bins[0], bins[-1], -(n_actions_per_joint // 2) - 0.5, n_actions_per_joint // 2 + 0.5), interpolation="none", vmin=0, vmax=1)
    fig.colorbar(im)
    ax.set_title("Action wrt vergence error")
    ax.set_xlabel("vergence error")
    ax.set_ylabel("action")
    ax.plot(action_set, np.arange(-4, 5), "k-", alpha=0.3)
    if save:
        filename = "/action_wrt_vergence_{}.png".format(scale)
        fig.savefig(plotpath + filename)
    else:
        plt.show()
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(np.log(np.array(probs)[:, ::-1].T + 0.03), extent=(bins[0], bins[-1], -(n_actions_per_joint // 2) - 0.5, n_actions_per_joint // 2 + 0.5), interpolation="none", vmin=-3, vmax=0)
    fig.colorbar(im)
    ax.set_title("Action wrt vergence error")
    ax.set_xlabel("vergence error")
    ax.set_ylabel("action")
    ax.plot(action_set, np.arange(-4, 5), "k-", alpha=0.3)
    if save:
        filename = "/action_wrt_vergence_log_{}.png".format(scale)
        fig.savefig(plotpath + filename)
    else:
        plt.show()
    plt.close(fig)


def preference_for_correct_action(data, scale="all", save=False):
    fig = plt.figure()
    data = group_by_episode(data)
    vergence_errors = vergence_error(data["eyes_position"], data["object_distance"])
    if scale == "all":
        values = data["critic_values_vergence"]
    else:
        values = data["scale_values_vergence"][:, :, scale]
    for i in range(9):
        ax = fig.add_subplot(3, 3, i + 1)
        action_value = action_set[i]
        diff_with_prev = -action_set[0] if i == 0 else action_set[i] - action_set[i - 1]
        diff_with_next =  action_set[8] if i == 8 else action_set[i + 1] - action_set[i]
        where = np.where(np.logical_and(vergence_errors[:, :-1] < action_value + diff_with_next / 2, action_value - diff_with_prev / 2 < vergence_errors[:, :-1]))
        positions = np.argmax(np.argsort(values[:, 1:][where], axis=1)[::-1] == i, axis=1)
        ax.hist(positions, bins=range(10))
    if save:
        filename = "/preference_for_correct_action_scale_{}.png".format(scale)
        fig.savefig(plotpath + filename)
    else:
        plt.show()
    plt.close(fig)


def vergence_error_episode_end_wrt_episode(data, save=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # episode_length = max([d["iteration"] for d in data])
    # data = [d for d in data if d["iteration"] == episode_length]
    data = group_by_episode(data)
    data = {k: v[:, -1] for k, v in data.items()}
    episode_numbers = data["episode_number"]
    vergence_errors = vergence_error(data["eyes_position"], data["object_distance"])
    ax.plot(episode_numbers, vergence_errors, ".", alpha=0.05)
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
    # episode_length = max([d["iteration"] for d in data])
    # data = [d for d in data if d["iteration"] == episode_length]
    data = group_by_episode(data)
    data = {k: v[:, -1] for k, v in data.items()}
    episode_numbers = data["episode_number"]
    vergence_errors = vergence_error(data["eyes_position"], data["object_distance"])
    abs_vergence_errors = np.array(np.abs(vergence_errors))
    args = np.argsort(episode_numbers)
    episode_numbers = episode_numbers[args]
    abs_vergence_errors = abs_vergence_errors[args]
    for win in [500]:
        mean_abs_vergence_errors = [np.mean(abs_vergence_errors[np.where(np.logical_and(episode_numbers < x + win, episode_numbers > x - win))]) for x in episode_numbers]
        smooth = savgol_filter(mean_abs_vergence_errors, 21, 3)
        ax.plot(episode_numbers, smooth, "-", lw=2, label="+-{}".format(win))
    ax.axhline(90 / 320, color="k", linestyle="--")
    ax.legend()
    ax.set_xlabel("episode")
    ax.set_ylabel("abs vergence error in degrees")
    ax.set_ylim([0.0, 1.5])
    ax.set_title("vergence error wrt time")
    if save:
        fig.savefig(plotpath + "/vergence_error_wrt_time.png")
    else:
        plt.show()
    plt.close(fig)


def vergence_episode_end_wrt_episode(data, save=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # episode_length = max([d["iteration"] for d in data])
    # data = [d for d in data if d["iteration"] == episode_length]
    data = group_by_episode(data)
    data = {k: v[:, -1] for k, v in data.items()}
    episode_numbers = data["episode_number"]
    vergence = data["eyes_position"][:, -1]
    args = np.argsort(episode_numbers)
    episode_numbers = episode_numbers[args]
    vergence = vergence[args]
    ax.scatter(episode_numbers, vergence, alpha=0.1)
    ax.set_xlabel("episode")
    ax.set_ylabel("vergence position in degrees")
    ax.set_title("vergence wrt time")
    if save:
        fig.savefig(plotpath + "/vergence_wrt_time.png")
    else:
        plt.show()
    plt.close(fig)


def vergence_wrt_object_distance(data, save=False):
    data = group_by_episode(data)
    data = {k: v[:, -1] for k, v in data.items()}
    fig = plt.figure()
    ax = fig.add_subplot(111)
    object_distance = data["object_distance"]
    vergence = data["eyes_position"][:, -1]
    rewards = data["total_reward"]
    X = np.linspace(np.min(object_distance), np.max(object_distance), 100)
    correct = to_angle(X)
    ax.scatter(object_distance, vergence, alpha=0.1)
    ax.plot(X, correct, "k-")
    ax.set_xlabel("object position in meters")
    ax.set_ylabel("vergence position in degrees")
    ax.set_title("final vergence wrt object position")
    if save:
        fig.savefig(plotpath + "/vergence_wrt_object_distance.png")
    else:
        plt.show()
    plt.close(fig)


###
# SPEED RELATED
###

def speed_error(eyes_speed, object_speed, pantilt=''):
    if pantilt == 'pan':
        return np.abs(eyes_speed[..., 1] - object_speed[..., 1])
    elif pantilt == 'tilt':
        return np.abs(eyes_speed[..., 0] - object_speed[..., 0])
    else:
        return np.sqrt(np.sum((eyes_speed - object_speed) ** 2, axis=-1))




def speed_error_episode_end_wrt_episode(data, plotpath, pantilt='', save=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # episode_length = max([d["iteration"] for d in data])
    # data = [d for d in data if d["iteration"] == episode_length]
    data = group_by_episode(data)
    data = {k: v[:, -1] for k, v in data.items()}
    episode_numbers = data["episode_number"]
    speed_errors = speed_error(data["eyes_speed"], data["object_speed"], pantilt)
    ax.plot(episode_numbers, speed_errors, ".", alpha=0.05)
    ax.set_xlabel("episode")
    ax.set_ylabel("speed error in degrees")
    ax.set_title("speed error wrt time")
    if save:
        fig.savefig(plotpath + "/speed_error_wrt_time_" + pantilt + ".png")
    else:
        plt.show()
    plt.close(fig)


def delta_reward_wrt_delta_speed(data, plotpath, save=False):
    gridsize = 100
    data = group_by_episode(data)
    n_trajectories = len(data["iteration"])
    fig = plt.figure()
    speed_errors = speed_error_sub(data["eyes_speed"], data["object_speed"], '')
    print(speed_errors.shape)
    print(n_trajectories)
    speed_error_start = []
    speed_error_end = []
    delta_reward = []
    rewards = data["total_reward"]
    for i in range(n_trajectories):
        for v0, r0 in zip(speed_errors[i], rewards[i]):
            for v1, r1 in zip(speed_errors[i], rewards[i]):
                speed_error_start.append(v0)
                speed_error_end.append(v1)
                delta_reward.append(r1 - r0)
    ax = fig.add_subplot(111)#, facecolor='grey')
    hexbin = ax.hexbin(speed_error_start, speed_error_end, delta_reward, cmap=seismic, norm=MidPointNorm(), gridsize=gridsize, mincnt=10, vmin=-1, vmax=1)
    cb = fig.colorbar(hexbin, ax=ax)
    cb.set_label("Delta reward")
    if save:
        fig.savefig(plotpath + "/delta_reward_wrt_delta_speed.png")
    else:
        plt.show()
        plt.close(fig)


def cart2pol(coord):
    rho = np.sqrt(coord[0]**2 + coord[2]**2)
    phi = np.arctan2(coord[2], coord[0])
    return [rho, phi]


def movement(data):
    data = group_by_episode(data)
    n_trajectories = len(data["iteration"])
    np.apply_along_axis(cart2pol, 2, data['eyes_position'])
    print(data['eyes_position'].shape)
    print(data['eyes_position'][:,:,0])


def generate_sample_data():
    data = []
    data_color = []
    for i in range(5):
        trajectory = []
        color = []
        direction = np.random.uniform(np.deg2rad(0), np.deg2rad(360))
        speed = np.random.uniform(1, 3)
        for j in range(10):
            error = np.deg2rad(6 * (np.random.random_sample() - 0.5))
            color.append(error + 3)
            x, y = (speed*j), direction + error
            trajectory.append((x, y))
        data.append(trajectory)
        data_color.append(color)
    return (data, data_color)


def circular_xy_movement_plot(data, plotpath, save=False):
    data = group_by_episode(data)
    print(data['object_position'][0, :, 0])
    print(data['object_position'][0, :, 1])
    print(data['object_position'][0, :, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # line_list = []
    # for line in data[0]:
    #     r, theta = zip(*line)
    #     line_list.append(([*theta], [*r]))
    for i, line in enumerate(data['object_position'][:, 0, 0]):
    #     color = []
    #     for k in range(len(line[1])):
    #         color.append('green')
    #     Blues = plt.get_cmap('Blues')
        ax.plot(data['object_position'][i, :, 0], data['object_position'][i, :, 2])
    #max_value = max(map(lambda x: x[-1][0], data[0]))
    #ax.set_rmax(max_value)
    #ax.set_rticks(np.arange(max_value, step=max_value/5))  # less radial ticks
    #ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax.grid(True)
    ax.set_title("Pan/Tilt Error", va='bottom')
    if save:
        fig.savefig(plotpath + "/movement_pan_tilt_xy.png")
    else:
        plt.show()


def circular_polar_movement_plot(data, plotpath, save=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    line_list = []
    for line in data[0]:
        r, theta = zip(*line)
        line_list.append(([*theta], [*r]))
    for i, line in enumerate(line_list):
        color = []
        for k in range(len(line[1])):
            color.append('green')
        Blues = plt.get_cmap('Blues')
        ax.scatter(line[0], line[1], c=data[1][i])
    max_value = max(map(lambda x: x[-1][0], data[0]))
    ax.set_rmax(max_value)
    ax.set_rticks(np.arange(max_value, step=max_value/5))  # less radial ticks
    ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax.grid(True)
    ax.set_title("Pan/Tilt Error", va='bottom')
    if save:
        fig.savefig(plotpath + "/movement_pan_tilt.png")
    else:
        plt.show()


def circular_polar_movement_plot_2(data, plotpath, save=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')

    data = group_by_episode(data)
    print(data['eyes_position'][:, :, 0])
    data['eyes_position_polar'] = np.apply_along_axis(cart2pol, 2, data['eyes_position'])
    data['eyes_position_polar_sample'] = data['eyes_position_polar'][np.random.choice(data['eyes_position_polar'].shape[0], 100, replace=False)]
    print(data['eyes_position_polar_sample'].shape)
    print(data['eyes_position_polar_sample'][:,:,0])
    r_values = data['eyes_position_polar_sample'][:,:,0]
    phi_values = data['eyes_position_polar_sample'][:,:,1]
    for i, trajectory in enumerate(r_values):
        ax.plot(phi_values[i], r_values[i])

    max_value = 6# max(map(lambda x: x[-1][0], data[0]))
    ax.set_rmax(max_value)
    ax.set_rticks(np.arange(max_value, step=max_value/5))  # less radial ticks
    ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax.grid(True)
    ax.set_title("Pan/Tilt Error", va='bottom')
    if save:
        fig.savefig(plotpath + "/movement_pan_tilt.png")
    else:
        plt.show()
