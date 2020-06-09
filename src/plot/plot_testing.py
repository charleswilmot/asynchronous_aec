import pickle
import numpy as np
import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from read_data import filter_data, get_experiment_metadata
from helper.generate_test_conf import TestConf
from plot.write_data import FigureManager
from helper.utils import to_distance, to_angle

cdict = {'red': ((0.0, 1.0, 1.0),
                 (0.5, 0.0, 0.0),
                 (1.0, 0.0, 0.0)),
         'blue': ((0.00, 0.0, 0.0),
                  (0.25, 0.0, 0.0),
                  (0.50, 1.0, 1.0),
                  (0.75, 0.0, 0.0),
                  (1.00, 0.0, 0.0)),
         'green': ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (1.0, 1.0, 1.0))
         }
rbg_cmap = LinearSegmentedColormap("rbg", cdict)


def get_new_ax(fig, n_subplots, subplot_index, method="square"):
    if method == "square":
        n = int(np.ceil(np.sqrt(n_subplots)))
        if n != np.sqrt(n_subplots):
            m = n - 1
        else:
            m = n
        return fig.add_subplot(m, n, subplot_index + 1)
    if method == "horizontal":
        return fig.add_subplot(1, n_subplots, subplot_index + 1)
    if method == "vertical":
        return fig.add_subplot(n_subplots, 1, subplot_index + 1)


def plot_tilt_path_all(fig, anchors, data):
    # generate sub lists of param anchors
    data = filter_data(data, **anchors)
    list_of_subdata = [filter_data(data, object_distances=[d]) for d in anchors["object_distances"]]
    n_subplots = len(anchors["object_distances"])
    # for each sublist, plot in an ax
    for subplot_index, subdata in enumerate(list_of_subdata):
        ax = get_new_ax(fig, n_subplots, subplot_index, method="square")
        plot_tilt_path(ax, subdata)


def plot_tilt_path(ax, data):
    test_data = np.array([b for a, b in data])
    result = test_data["eye_position"][np.random.randint(test_data["eye_position"].shape[0], size=40), :]
    for elem in result:
        ax.plot(range(len(elem)), elem[:, 0])


def plot_vergence_path(fig, anchors, data):
    data = filter_data(data, **anchors)
    test_data = np.array([b for a, b in data])
    n_subplots = 1
    ax = get_new_ax(fig, n_subplots, 0)
    result = test_data["eye_position"][np.random.randint(test_data["eye_position"].shape[0], size=40), :]
    print(result)
    print(np.shape(result))
    print(result[0][2])
    for elem in result:
        ax.plot(range(len(elem)), elem[:, 2], )


# TODO: Vlean this method
def plot_stimulus_path(fig, anchors, data):
    path = "/home/aecgroup/aecdata/Textures/mcgillManMade_600x600_bmp_selection/"
    # path = "../local_copies_of_aecgroup/mcgillManMade_600x600_bmp_selection/"
    textures_names = os.listdir(path)
    textures_list = [np.array(Image.open(path + name), dtype=np.uint8) for name in textures_names]
    lim_stimulus = min(10, len(anchors["stimulus"]))
    data = filter_data(data, **anchors)
    list_of_subdata = [filter_data(data, stimulus=[s]) for s in anchors["stimulus"][:lim_stimulus]]
    n_subplots = lim_stimulus * 2
    for subplot_index, elem in enumerate(list_of_subdata):
        ax = get_new_ax(fig, n_subplots, subplot_index * 2)
        test_data = np.array([b for a, b in elem])
        ax.boxplot(test_data["vergence_error"], notch=True, showfliers=False)
        ax.axhline(0, color="k")
        ax.axhline(90 / 320, color="k", linestyle="--")
        ax.axhline(-90 / 320, color="k", linestyle="--")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Vergence error (deg)")
        # ax.set_ylim([np.min(test_cases["vergence_error"]) - 1, np.max(test_cases["vergence_error"]) + 1])
        ax.set_title("Stimulus  {:.1f}".format(elem[0][0]["stimulus"]))
        ax2 = get_new_ax(fig, n_subplots, (subplot_index * 2) + 1)
        ax2.imshow(textures_list[elem[0][0]["stimulus"]])


def plot_recerr_wrt_speed_error(fig, anchors, data, pan_or_tilt="pan"):
    ax = fig.add_subplot(111)
    plot_recerr_wrt_speed_error_ax(ax, anchors, data, pan_or_tilt=pan_or_tilt)


def plot_recerr_wrt_speed_error_ax(ax, anchors, data, pan_or_tilt="pan",
                                   ylim=[0, 0.04], set_title=True, set_ylabel=True, inset=True, legend=False):
    joint_index = 1 if pan_or_tilt == "pan" else 0
    data = filter_data(data, **anchors)
    test_cases = np.array([a for a, b in data])
    data = np.array([b for a, b in data])
    mean = None
    n = 0
    one_pixel = 90 / 320
    for stimulus in anchors["stimulus"]:
        where = np.where(test_cases["stimulus"] == stimulus)
        subdata = data[where][:, 0]
        speed_error = subdata["speed_error"][:, joint_index]
        recerr = subdata["total_recerrs_4_frames"]
        args = np.argsort(speed_error)
        ax.plot(speed_error[args] / one_pixel, recerr[args], 'b-', linewidth=1, alpha=0.6)
        if mean is None:
            mean = recerr[args]
        else:
            mean += recerr[args]
        n += 1
    ax.plot(speed_error[args] / one_pixel, mean / n, 'r-', linewidth=3, label="mean")
    if inset:
        axins = inset_axes(ax, width="20%", height="20%", borderpad=2)  # , loc=4
        axins.plot(speed_error[args] / one_pixel, mean / n, 'r-')
        axins.axvline(0, color="k", linestyle="--", alpha=0.5)
        axins.set_xticks([])
        axins.set_yticks([])
        axins.set_title("Mean only")
    ax.set_xlabel("{} error (px/it)".format(pan_or_tilt))
    if set_ylabel:
        ax.set_ylabel("Reconstruction error")
    else:
        ax.set_yticks([])
    if legend:
        ax.legend()
    if set_title:
        ax.set_title("Reconstruction error wrt speed error")
    ax.axvline(0, color="k", linestyle="--")
    ax.set_ylim(ylim)


def plot_recerr_wrt_vergence_error(fig, anchors, data, turn_2_frames_vergence_on=True):
    ax = fig.add_subplot(111)
    plot_recerr_wrt_vergence_error_ax(ax, anchors, data, turn_2_frames_vergence_on=turn_2_frames_vergence_on)


def plot_recerr_wrt_vergence_error_ax(ax, anchors, data, turn_2_frames_vergence_on=True,
                                      ylim=[0, 0.04], set_title=True, set_ylabel=True, inset=True, legend=False):
    total_recerrs = "total_recerrs_2_frames" if turn_2_frames_vergence_on else "total_recerrs_4_frames"
    joint_index = 2
    data = filter_data(data, **anchors)
    test_cases = np.array([a for a, b in data])
    data = np.array([b for a, b in data])
    mean = None
    n = 0
    one_pixel = 90 / 320
    for stimulus in anchors["stimulus"]:
        where = np.where(test_cases["stimulus"] == stimulus)
        subdata = data[where][:, 0]
        vergence_error = subdata["vergence_error"]
        recerr = subdata[total_recerrs]
        args = np.argsort(vergence_error)
        ax.plot(vergence_error[args] / one_pixel, recerr[args], 'b-', linewidth=1, alpha=0.6)
        if mean is None:
            mean = recerr[args]
        else:
            mean += recerr[args]
        n += 1
    ax.plot(vergence_error[args] / one_pixel, mean / n, 'r-', linewidth=3, label="mean")
    if inset:
        axins = inset_axes(ax, width="20%", height="20%", borderpad=2)  # , loc=4
        axins.plot(vergence_error[args] / one_pixel, mean / n, 'r-')
        axins.axvline(0, color="k", linestyle="--", alpha=0.5)
        axins.set_xticks([])
        axins.set_yticks([])
        axins.set_title("Mean only")
    ax.set_xlabel("vergence error (px)")
    if set_ylabel:
        ax.set_ylabel("Reconstruction error")
    else:
        ax.set_yticks([])
    if legend:
        ax.legend()
    if set_title:
        ax.set_title("Reconstruction error wrt vergence error")
    ax.axvline(0, color="k", linestyle="--")
    ax.set_ylim(ylim)


def plot_critic_accuracy_vergence(fig, anchors, data, reward_scaling_factor=200, stimulus=0, at_error=0,
                                  turn_2_frames_vergence_on=True):
    ax = fig.add_subplot(111)
    plot_critic_accuracy_vergence_ax(ax, anchors, data, reward_scaling_factor=reward_scaling_factor, stimulus=stimulus,
                                     at_error=at_error, turn_2_frames_vergence_on=turn_2_frames_vergence_on)


def plot_critic_accuracy_vergence_ax(ax, anchors, data, turn_2_frames_vergence_on=True,
                                     set_title=True, set_ylabel=True,
                                     reward_scaling_factor=200, stimulus=0, at_error=0):
    total_recerrs = "total_recerrs_2_frames" if turn_2_frames_vergence_on else "total_recerrs_4_frames"
    joint_index = 2
    data = filter_data(data, **anchors)
    test_cases = np.array([a for a, b in data])
    data = np.array([b for a, b in data])
    actions_in_pixels = np.array([-8, -4, -2, -1, 0, 1, 2, 4, 8]) / 2
    one_pixel = 90 / 320
    n_actions = len(actions_in_pixels)
    # x axis: action
    # y axis: recerr and critic val
    # text indicating which stimulus, which error
    # vertical line showing the error on the x axis
    where = np.where(test_cases["stimulus"] == stimulus)
    data = data[where][:, 0]
    rewards = []
    critic_values = []
    where = np.where(np.abs(data["vergence_error"] / one_pixel - at_error) < 1e-2)
    baseline_reconstructiuon_error = data[total_recerrs][where]
    critic_values = data["critic_value_vergence"][where][0]
    for action in actions_in_pixels:
        where = np.where(np.abs(data["vergence_error"] / one_pixel + action - at_error) < 1e-2)
        reconstruction_error = data[total_recerrs][where]
        rewards.append((baseline_reconstructiuon_error - reconstruction_error) * reward_scaling_factor)
    ax.plot(actions_in_pixels, rewards, label="reward")
    ax.plot(actions_in_pixels, critic_values, label="critic")
    ax.set_xlabel("Action (in pixel)")
    ax.set_ylabel("Reward / Return")
    ax.set_ylim([-2, 0.2])
    ax.legend()


def plot_critic_accuracy_speed(fig, anchors, data, reward_scaling_factor=200, pan_or_tilt="pan", stimulus=0,
                               at_error=0):
    ax = fig.add_subplot(111)
    plot_critic_accuracy_speed_ax(ax, anchors, data, pan_or_tilt, reward_scaling_factor=reward_scaling_factor,
                                  stimulus=stimulus, at_error=at_error)


def plot_critic_accuracy_speed_ax(ax, anchors, data, pan_or_tilt="pan",
                                  set_title=True, set_ylabel=True,
                                  reward_scaling_factor=200, stimulus=0, at_error=0):
    joint_index = 2
    data = filter_data(data, **anchors)
    test_cases = np.array([a for a, b in data])
    data = np.array([b for a, b in data])
    actions_in_pixels = np.array([-8, -4, -2, -1, 0, 1, 2, 4, 8]) / 2
    one_pixel = 90 / 320
    n_actions = len(actions_in_pixels)
    # x axis: action
    # y axis: recerr and critic val
    # text indicating which stimulus, which error
    # vertical line showing the error on the x axis
    where = np.where(test_cases["stimulus"] == stimulus)
    data = data[where][:, 0]
    rewards = []
    critic_values = []
    pan_tilt_index = 0 if pan_or_tilt == "tilt" else 1
    where = np.where(np.abs(data["speed_error"][:, pan_tilt_index] / one_pixel - at_error) < 1e-2)
    baseline_reconstructiuon_error = data["total_recerrs_4_frames"][where]
    critic_values = data["critic_value_{}".format(pan_or_tilt)][where][0]
    for action in actions_in_pixels:
        where = np.where(np.abs(data["speed_error"][:, pan_tilt_index] / one_pixel + action - at_error) < 1e-2)
        reconstruction_error = data["total_recerrs_4_frames"][where]
        rewards.append((baseline_reconstructiuon_error - reconstruction_error) * reward_scaling_factor)
    ax.plot(actions_in_pixels, rewards, label="reward")
    ax.plot(actions_in_pixels, critic_values, label="critic")
    ax.set_xlabel("Action (in pixel)")
    ax.set_ylabel("Reward / Return")
    ax.set_ylim([-2, 0.2])
    ax.legend()


def plot_recerr_wrt_speed_error_per_scale(fig, anchors, data, pan_or_tilt="pan"):
    ax = fig.add_subplot(111)
    plot_recerr_wrt_speed_error_per_scale_ax(ax, anchors, data, pan_or_tilt=pan_or_tilt)


def plot_recerr_wrt_speed_error_per_scale_ax(ax, anchors, data, pan_or_tilt="pan",
                                             ylim=[0, 0.03], set_title=True, set_ylabel=True, legend=True):
    joint_index = 1 if pan_or_tilt == "pan" else 0
    data = np.array([b for a, b in filter_data(data, **anchors)])
    one_pixel = 90 / 320
    speed_errors = np.array([s[joint_index] for s in anchors["speed_errors"]])
    args = np.argsort(data["speed_error"][:, 0, joint_index])
    data = data[args].reshape((len(speed_errors), -1))
    # labels = ["fine", "middle", "coarse", "very coarse", "very very coarse"]
    labels = ["fine", "coarse", "very coarse", "very very coarse"]
    for recerrs, scale_name in zip(data["scale_recerrs_4_frames"].T, labels):
        ax.plot(speed_errors / one_pixel, np.mean(recerrs, axis=0), linewidth=3, label=scale_name)
    ax.set_xlabel("{} error (px/it)".format(pan_or_tilt))
    if set_ylabel:
        ax.set_ylabel("Reconstruction error")
    else:
        ax.set_yticks([])
    if legend:
        ax.legend()
    if set_title:
        ax.set_title("Reconstruction error wrt speed error")
    ax.axvline(0, color="k", linestyle="--")
    ax.set_ylim(ylim)


def plot_recerr_wrt_vergence_error_per_scale(fig, anchors, data, turn_2_frames_vergence_on=True):
    ax = fig.add_subplot(111)
    plot_recerr_wrt_vergence_error_per_scale_ax(ax, anchors, data, turn_2_frames_vergence_on=turn_2_frames_vergence_on)


def plot_recerr_wrt_vergence_error_per_scale_ax(ax, anchors, data,
                                                ylim=[0, 0.03], set_title=True, set_ylabel=True, legend=True,
                                                turn_2_frames_vergence_on=True):
    scale_recerrs = "scale_recerrs_2_frames" if turn_2_frames_vergence_on else "scale_recerrs_4_frames"
    data = np.array([b for a, b in filter_data(data, **anchors)])
    one_pixel = 90 / 320
    vergence_errors = np.array(anchors["vergence_errors"])
    args = np.argsort(data["vergence_error"][:, 0])
    data = data[args].reshape((len(vergence_errors), -1))
    for recerrs, scale_name in zip(data[scale_recerrs].T, ["fine", "coarse", "very coarse", "very very coarse"]):
        ax.plot(vergence_errors / one_pixel, np.mean(recerrs, axis=0), linewidth=3, label=scale_name)
    ax.set_xlabel("vergence error (px)")
    if set_ylabel:
        ax.set_ylabel("Reconstruction error")
    else:
        ax.set_yticks([])
    if legend:
        ax.legend()
    if set_title:
        ax.set_title("Reconstruction error wrt speed error")
    ax.axvline(0, color="k", linestyle="--")
    ax.set_ylim(ylim)


def plot_reward_wrt_action_speed_error_pair(fig, anchors, data, reward_scaling_factor=200, pan_or_tilt="pan",
                                            title_supplement=""):
    joint_index = 1 if pan_or_tilt == "pan" else 0
    data = filter_data(data, **anchors)
    test_cases = np.array([a for a, b in data])
    data = np.array([b for a, b in data])
    ax = fig.add_subplot(111)
    speed_errors = np.sort(np.unique(test_cases["speed_error"][:, joint_index]))
    actions_in_semi_pixels = np.array([-8, -4, -2, -1, 0, 1, 2, 4, 8])
    count_matrix = np.zeros((actions_in_semi_pixels.shape[0], speed_errors.shape[0]), dtype=np.int32)
    reward_matrix = np.zeros((actions_in_semi_pixels.shape[0], speed_errors.shape[0]), dtype=np.float32)
    n_speed_errors = len(speed_errors)
    n_actions = len(actions_in_semi_pixels)
    for action_index, action in enumerate(actions_in_semi_pixels):
        for index_start_error in range(n_speed_errors):
            index_end_error = index_start_error + action
            if 0 < index_end_error < n_speed_errors:
                where_start = np.where(data["speed_error"][:, 0, joint_index] == speed_errors[index_start_error])
                where_end = np.where(data["speed_error"][:, 0, joint_index] == speed_errors[index_end_error])
                rewards = data["total_recerrs_4_frames"][where_start] - data["total_recerrs_4_frames"][where_end]
                reward_matrix[action_index][index_start_error] += np.sum(rewards)
                count_matrix[action_index][index_start_error] += rewards.shape[0]
    where = np.where(count_matrix == 0)
    count_matrix[where] = 1
    reward_matrix /= count_matrix
    reward_matrix[where] = np.nan
    one_pixel = 90 / 320
    xmin = speed_errors[0] / one_pixel
    xmax = speed_errors[-1] / one_pixel
    ymin = -n_actions / 2
    ymax = n_actions / 2
    im = ax.imshow(reward_matrix * reward_scaling_factor, aspect="auto", extent=[xmin, xmax, ymin, ymax],
                   cmap="seismic")
    # im = ax.imshow(reward_matrix * reward_scaling_factor, aspect="auto", extent=[xmin, xmax, ymin, ymax], cmap=rbg_cmap)
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel('reward', rotation=270)
    ax.axvline(0, color="k", linestyle="--", alpha=0.8)
    ax.set_xlabel("{} speed error in pixel/it".format(pan_or_tilt))
    ax.set_ylabel("action in px.it-1 (vergence) or px.it-2 (pan, tilt)")
    ax.set_yticks(np.linspace(-(n_actions - 1) / 2, (n_actions - 1) / 2, n_actions))
    ax.set_yticklabels(actions_in_semi_pixels / 2)
    ax.set_title("Reward wrt action / speed error pair" + title_supplement)


def plot_policy_pan_tilt(fig, anchors, data, pan_or_tilt="pan", title_supplement=""):
    ax = fig.add_subplot(111)
    im = plot_policy_pan_tilt_ax(ax, anchors, data, pan_or_tilt=pan_or_tilt, title_supplement=title_supplement)
    fig.colorbar(im)


def plot_policy_pan_tilt_ax(ax, anchors, data, pan_or_tilt="pan", title_supplement="", set_ylabel=True, set_title=True,
                            cmap_name="Greys"):
    joint_index = 1 if pan_or_tilt == "pan" else 0
    data = filter_data(data, **anchors)
    test_cases = np.array([a for a, b in data])
    data = np.array([b for a, b in data])
    speed_errors = np.sort([s[joint_index] for s in anchors["speed_errors"]])
    actions_in_semi_pixels = np.array([-8, -4, -2, -1, 0, 1, 2, 4, 8])
    n_actions = len(actions_in_semi_pixels)
    result_matrix = np.zeros((actions_in_semi_pixels.shape[0], speed_errors.shape[0]), dtype=np.float32)
    for speed_error_index, speed_error in enumerate(speed_errors):
        where = np.where(np.abs(data["speed_error"][:, 0, joint_index] - speed_error) < 1e-5)
        greedy_actions = data["action_index"][where]  # shape = (n_stimulus, 1, 3)
        for action_index in greedy_actions[:, 0, joint_index]:
            result_matrix[action_index, speed_error_index] += 1
        # print(result_matrix[:, speed_error_index])
        result_matrix[:, speed_error_index] /= len(greedy_actions)
    one_pixel = 90 / 320
    im = ax.imshow(result_matrix, aspect="auto",
                   extent=[speed_errors[0] / one_pixel, speed_errors[-1] / one_pixel, -n_actions / 2, n_actions / 2],
                   cmap=plt.get_cmap(cmap_name))
    ax.axvline(0, color="k", linestyle="--", alpha=0.8)
    ax.set_xlabel("{} error (px/it)".format(pan_or_tilt))
    if set_ylabel:
        ax.set_ylabel("action in px.it-1 (vergence) or px.it-2 (pan, tilt)")
        ax.set_yticks(np.linspace(-(n_actions - 1) / 2, (n_actions - 1) / 2, n_actions))
        ax.set_yticklabels(actions_in_semi_pixels / 2)
    else:
        ax.set_yticks([])
    if set_title:
        ax.set_title("Action probabilities" + title_supplement)
    return im


def plot_policy_vergence(fig, anchors, data, title_supplement=""):
    ax = fig.add_subplot(111)
    im = plot_policy_vergence_ax(ax, anchors, data, title_supplement=title_supplement)
    fig.colorbar(im)


def plot_policy_vergence_ax(ax, anchors, data, title_supplement="", set_ylabel=True, set_title=True, cmap_name="Greys"):
    joint_index = 2
    data = filter_data(data, **anchors)
    test_cases = np.array([a for a, b in data])
    data = np.array([b for a, b in data])
    vergence_errors = np.sort(anchors["vergence_errors"])
    actions_in_semi_pixels = np.array([-8, -4, -2, -1, 0, 1, 2, 4, 8])
    n_actions = len(actions_in_semi_pixels)
    result_matrix = np.zeros((actions_in_semi_pixels.shape[0], vergence_errors.shape[0]), dtype=np.float32)
    for vergence_error_index, vergence_error in enumerate(vergence_errors):
        where = np.where(np.abs(data["vergence_error"][:, 0] - vergence_error) < 1e-5)
        greedy_actions = data["action_index"][where]
        for action_index in greedy_actions[:, 0, joint_index]:
            result_matrix[action_index, vergence_error_index] += 1
        result_matrix[:, vergence_error_index] /= len(greedy_actions)
    one_pixel = 90 / 320
    im = ax.imshow(result_matrix, aspect="auto",
                   extent=[vergence_errors[0] / one_pixel, vergence_errors[-1] / one_pixel, -n_actions / 2,
                           n_actions / 2], cmap=plt.get_cmap(cmap_name))
    ax.axvline(0, color="k", linestyle="--", alpha=0.8)
    ax.set_xlabel("vergence error (px)")
    if set_ylabel:
        ax.set_ylabel("action in px.it-1 (vergence) or px.it-2 (pan, tilt)")
        ax.set_yticks(np.linspace(-(n_actions - 1) / 2, (n_actions - 1) / 2, n_actions))
        ax.set_yticklabels(actions_in_semi_pixels / 2)
    else:
        ax.set_yticks([])
    if set_title:
        ax.set_title("Action probabilities" + title_supplement)
    return im


def plot_vergence_trajectory_all(fig, anchors, data):
    # generate sub lists of param anchors
    data = filter_data(data, **anchors)
    list_of_subdata = [filter_data(data, object_distances=[d]) for d in anchors["object_distances"]]
    n_subplots = len(anchors["object_distances"])
    # for each sublist, plot in an ax
    for subplot_index, subdata in enumerate(list_of_subdata):
        ax = get_new_ax(fig, n_subplots, subplot_index, method="square")
        plot_vergence_trajectory_sub(ax, subdata)


def plot_vergence_trajectory_sub(ax, data):
    test_cases = np.array([a for a, b in data])
    for vergence_error in np.unique(test_cases["vergence_error"]):
        ax.axhline(y=vergence_error, xmin=0, xmax=0.1, color="r")
        # filtered = filter_data(data, vergence_errors=[vergence_error])
        # # plot for each stimulus in light grey
        # test_data = np.array([b for a, b in filtered])
        # ax.plot(range(1, test_data.shape[0] + 1), test_data["vergence_error"].T, color="grey", alpha=0.8)
    # plot the abscissa, add title, axis label etc...
    test_data = np.array([b for a, b in data])
    ax.boxplot(test_data["vergence_error"], notch=True, showfliers=False)  # , whis=[10, 90])
    ax.axhline(0, color="k")
    ax.axhline(90 / 320, color="k", linestyle="--")
    ax.axhline(-90 / 320, color="k", linestyle="--")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Vergence error (deg)")
    ax.set_ylim([np.min(test_cases["vergence_error"]) - 1, np.max(test_cases["vergence_error"]) + 1])
    ax.set_title("Object distance  {:.4f}".format(data[0][0]["object_distance"]))


def plot_speed_trajectory(fig, anchors, data, pan_or_tilt="pan"):
    joint_index = 1 if pan_or_tilt == "pan" else 0
    data = filter_data(data, **anchors)
    test_cases = np.array([a for a, b in data])
    data = np.array([b for a, b in data], dtype=data[0][1].dtype)
    n_speeds = len(anchors["speed_errors"])
    at = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 19])
    for i, speed in enumerate(anchors["speed_errors"]):
        ax = get_new_ax(fig, n_speeds, i, method="square")
        where = np.where(test_cases["speed_error"][:, joint_index] == speed[joint_index])
        subdata = data[where]
        # print(subdata["speed_error"][:6, :10, joint_index])
        # print("\n\n\n\n\n\n\n")
        # ax.plot(subdata["speed_error"][:, :10, joint_index].T / 90 * 320, "b-", alpha=0.2)
        bp = ax.boxplot(subdata["speed_error"][:, at, joint_index] / 90 * 320, showfliers=False)
        for i, line in enumerate(bp["medians"]):
            x_left, x_right = line.get_xdata()
            y_left, y_right = line.get_ydata()
            if i != 0:
                ax.plot([xprev_right, x_left], [yprev_right, y_left], color=line.get_color(),
                        linewidth=line.get_linewidth())
            xprev_left, xprev_right = x_left, x_right
            yprev_left, yprev_right = y_left, y_right
        ax.set_title("{:.1f} pix/it".format(speed[joint_index] / 90 * 320))
        geo = ax.get_geometry()
        if (geo[-1] - 1) // geo[1] == geo[0] - 1:
            ax.set_xlabel("Iteration within episode")
        if geo[-1] % geo[1] == 1:
            ax.set_ylabel("speed error in pixel/it")
        xticks = np.arange(1, 1 + len(at))
        xticklabels = at + 1
        ax.set_xticks(xticks[1::2])
        ax.set_xticklabels(xticklabels[1::2])
    fig.tight_layout()


def plot_abs_speed_trajectory(fig, anchors, data, pan_or_tilt="pan"):
    ax = fig.add_subplot(111)
    plot_abs_speed_trajectory_ax(ax, anchors, data, pan_or_tilt=pan_or_tilt)


def plot_abs_speed_trajectory_ax(ax, anchors, data, pan_or_tilt="pan", set_title=True, set_ylabel=True,
                                 ylim=[-0.2, 4.5]):
    joint_index = 1 if pan_or_tilt == "pan" else 0
    data = filter_data(data, **anchors)
    test_cases = np.array([a for a, b in data])
    data = np.array([b for a, b in data], dtype=data[0][1].dtype)
    at = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 19])
    bp = ax.boxplot(np.abs(data["speed_error"][:, at, joint_index]) / 90 * 320, showfliers=False)
    ax.axhline(1, color="grey", linestyle="--")
    for i, line in enumerate(bp["medians"]):
        x_left, x_right = line.get_xdata()
        y_left, y_right = line.get_ydata()
        if i != 0:
            ax.plot([xprev_right, x_left], [yprev_right, y_left], color=line.get_color(),
                    linewidth=line.get_linewidth())
        xprev_left, xprev_right = x_left, x_right
        yprev_left, yprev_right = y_left, y_right
    if set_title:
        ax.set_title("Absolute speed error within one episode")
    ax.set_xlabel("Iteration ({})".format(pan_or_tilt))
    ax.set_ylim(ylim)
    if set_ylabel:
        ax.set_ylabel("absolute error in px (vergence) or px.it (pan, tilt)")
    else:
        ax.set_yticks([])
    xticks = np.arange(1, 1 + len(at))
    xticklabels = at + 1
    ax.set_xticks(xticks[1::2])
    ax.set_xticklabels(xticklabels[1::2])


def plot_abs_speed_trajectory_mean_std(fig, anchors, data, pan_or_tilt="pan"):
    ax = fig.add_subplot(111)
    plot_abs_speed_trajectory_mean_std_ax(ax, anchors, data, pan_or_tilt=pan_or_tilt)


def plot_abs_speed_trajectory_mean_std_ax(ax, anchors, data, pan_or_tilt="pan", set_title=True, set_ylabel=True,
                                          ylim=[-0.2, 4.5]):
    joint_index = 1 if pan_or_tilt == "pan" else 0
    data = filter_data(data, **anchors)
    test_cases = np.array([a for a, b in data])
    data = np.array([b for a, b in data], dtype=data[0][1].dtype)
    values = np.abs(data["speed_error"][:, :, joint_index]) / 90 * 320
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    X = np.arange(len(mean))
    ax.fill_between(X, mean - std, mean + std, alpha=0.5)
    ax.plot(X, mean)
    ax.axhline(1, color="grey", linestyle="--")
    if set_title:
        ax.set_title("Mean absolute speed error within one episode")
    ax.set_xlabel("Iteration ({})".format(pan_or_tilt))
    ax.set_ylim(ylim)
    if set_ylabel:
        ax.set_ylabel("Mean absolute error in px (vergence) or px.it-1 (pan, tilt)")
    else:
        ax.set_yticks([])
    xticks = [0, len(X) - 1]
    xticklabels = [1, len(X)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)


def plot_abs_vergence_trajectory(fig, anchors, data):
    ax = fig.add_subplot(111)
    plot_abs_vergence_trajectory_ax(ax, anchors, data)


def plot_abs_vergence_trajectory_ax(ax, anchors, data, set_title=True, set_ylabel=True, ylim=[-0.2, 4.5]):
    data = filter_data(data, **anchors)
    test_cases = np.array([a for a, b in data])
    data = np.array([b for a, b in data], dtype=data[0][1].dtype)
    at = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 19])
    bp = ax.boxplot(np.abs(data["vergence_error"][:, at]) / 90 * 320, showfliers=False)
    ax.axhline(1, color="grey", linestyle="--")
    for i, line in enumerate(bp["medians"]):
        x_left, x_right = line.get_xdata()
        y_left, y_right = line.get_ydata()
        if i != 0:
            ax.plot([xprev_right, x_left], [yprev_right, y_left], color=line.get_color(),
                    linewidth=line.get_linewidth())
        xprev_left, xprev_right = x_left, x_right
        yprev_left, yprev_right = y_left, y_right
    if set_title:
        ax.set_title("Absolute speed error within one episode")
    ax.set_xlabel("Iteration (vergence)")
    ax.set_ylim(ylim)
    if set_ylabel:
        ax.set_ylabel("absolute error in px (vergence) or px.it (pan, tilt)")
    else:
        ax.set_yticks([])
    xticks = np.arange(1, 1 + len(at))
    xticklabels = at + 1
    ax.set_xticks(xticks[1::2])
    ax.set_xticklabels(xticklabels[1::2])


def plot_abs_vergence_trajectory_mean_std(fig, anchors, data):
    ax = fig.add_subplot(111)
    plot_abs_vergence_trajectory_mean_std_ax(ax, anchors, data)


def plot_abs_vergence_trajectory_mean_std_ax(ax, anchors, data, set_title=True, set_ylabel=True, ylim=[-0.2, 4.5]):
    data = filter_data(data, **anchors)
    test_cases = np.array([a for a, b in data])
    data = np.array([b for a, b in data], dtype=data[0][1].dtype)
    values = np.abs(data["vergence_error"]) / 90 * 320
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    X = np.arange(len(mean))
    ax.fill_between(X, mean - std, mean + std, alpha=0.5)
    ax.plot(X, mean)
    ax.axhline(1, color="grey", linestyle="--")
    if set_title:
        ax.set_title("Mean absolute vergence error within one episode")
    ax.set_xlabel("Iteration (vergence)")
    ax.set_ylim(ylim)
    if set_ylabel:
        ax.set_ylabel("Mean absolute error in px (vergence) or px.it-1 (pan, tilt)")
    else:
        ax.set_yticks([])
    xticks = [0, len(X) - 1]
    xticklabels = [1, len(X)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)


def all_wrt_speed_error(lists_of_param_anchors, data, reward_scaling_factor=200, pan_or_tilt="pan", per_stimulus=False):
    anchors = lists_of_param_anchors["wrt_{}_speed_error".format(pan_or_tilt)]

    if per_stimulus:
        for stimulus in anchors["stimulus"]:
            with FigureManager("{}_critic_accuracy_stimulus_{}.png".format(pan_or_tilt, stimulus)) as fig:
                plot_critic_accuracy_speed(fig, anchors, data, reward_scaling_factor=reward_scaling_factor,
                                           pan_or_tilt=pan_or_tilt, stimulus=stimulus)

    ### plot v shape pan
    with FigureManager("{}_recerr_wrt_speed_error.png".format(pan_or_tilt)) as fig:
        plot_recerr_wrt_speed_error(fig, anchors, data, pan_or_tilt=pan_or_tilt)

    with FigureManager("{}_recerr_wrt_speed_error_per_scale.png".format(pan_or_tilt)) as fig:
        plot_recerr_wrt_speed_error_per_scale(fig, anchors, data, pan_or_tilt=pan_or_tilt)

    ### plot reward pan
    with FigureManager("{}_reward_wrt_action_speed_error_pair.png".format(pan_or_tilt)) as fig:
        plot_reward_wrt_action_speed_error_pair(fig, anchors, data, reward_scaling_factor=reward_scaling_factor,
                                                pan_or_tilt=pan_or_tilt)

    with FigureManager("{}_policy.png".format(pan_or_tilt)) as fig:
        plot_policy_pan_tilt(fig, anchors, data, pan_or_tilt=pan_or_tilt)

    ## plot reward per stimulus pan
    if per_stimulus:
        for stimulus in anchors["stimulus"]:
            with FigureManager(
                    "{}_reward_wrt_action_speed_error_pair_stimulus_{}.png".format(pan_or_tilt, stimulus)) as fig:
                anchors["stimulus"] = [stimulus]
                plot_reward_wrt_action_speed_error_pair(fig, anchors, data, reward_scaling_factor=reward_scaling_factor,
                                                        pan_or_tilt=pan_or_tilt,
                                                        title_supplement="   (Stimulus #{})".format(stimulus))
    return


def all_wrt_vergence_error(lists_of_param_anchors, data, reward_scaling_factor=200, turn_2_frames_vergence_on=True,
                           per_stimulus=False):
    anchors = lists_of_param_anchors["wrt_vergence_error"]

    if per_stimulus:
        for stimulus in anchors["stimulus"]:
            with FigureManager("vergence_critic_accuracy_stimulus_{}.png".format(stimulus)) as fig:
                plot_critic_accuracy_vergence(fig, anchors, data, reward_scaling_factor=reward_scaling_factor,
                                              stimulus=stimulus, turn_2_frames_vergence_on=turn_2_frames_vergence_on)

    with FigureManager("vergence_policy.png") as fig:
        plot_policy_vergence(fig, anchors, data)

    with FigureManager("recerr_wrt_vergence_error.png") as fig:
        plot_recerr_wrt_vergence_error(fig, anchors, data, turn_2_frames_vergence_on=turn_2_frames_vergence_on)

    with FigureManager("recerr_wrt_vergence_error_per_scale.png") as fig:
        plot_recerr_wrt_vergence_error_per_scale(fig, anchors, data,
                                                 turn_2_frames_vergence_on=turn_2_frames_vergence_on)

    return


def all_speed_trajectory(lists_of_param_anchors, data, pan_or_tilt="pan"):
    anchors = lists_of_param_anchors["{}_speed_trajectory".format(pan_or_tilt)]
    with FigureManager("{}_speed_trajectory.png".format(pan_or_tilt)) as fig:
        plot_speed_trajectory(fig, anchors, data, pan_or_tilt=pan_or_tilt)

    with FigureManager("{}_abs_speed_trajectory.png".format(pan_or_tilt)) as fig:
        plot_abs_speed_trajectory(fig, anchors, data, pan_or_tilt=pan_or_tilt)

    with FigureManager("{}_abs_speed_trajectory_mean_std.png".format(pan_or_tilt)) as fig:
        plot_abs_speed_trajectory_mean_std(fig, anchors, data, pan_or_tilt=pan_or_tilt)

    return


def all_vergence_trajectory(lists_of_param_anchors, data):
    anchors = lists_of_param_anchors["vergence_trajectory"]
    with FigureManager("vergence_trajectory.png") as fig:
        plot_vergence_trajectory_all(fig, anchors, data)

    with FigureManager("abs_vergence_trajectory.png") as fig:
        plot_abs_vergence_trajectory(fig, anchors, data)

    with FigureManager("abs_vergence_trajectory_mean_std.png") as fig:
        plot_abs_vergence_trajectory_mean_std(fig, anchors, data)

    return


def triple_trajectory(lists_of_param_anchors, data):
    with FigureManager("triple_trajectory.png") as fig:
        anchors = lists_of_param_anchors["pan_speed_trajectory"]
        ax = fig.add_subplot(131)
        plot_abs_speed_trajectory_ax(ax, anchors, data, pan_or_tilt="pan", set_title=False)
        anchors = lists_of_param_anchors["tilt_speed_trajectory"]
        ax = fig.add_subplot(132)
        plot_abs_speed_trajectory_ax(ax, anchors, data, pan_or_tilt="tilt", set_title=False, set_ylabel=False)
        anchors = lists_of_param_anchors["vergence_trajectory"]
        ax = fig.add_subplot(133)
        plot_abs_vergence_trajectory_ax(ax, anchors, data, set_title=False, set_ylabel=False)

    with FigureManager("triple_trajectory_mean_std.png") as fig:
        anchors = lists_of_param_anchors["pan_speed_trajectory"]
        ax = fig.add_subplot(131)
        plot_abs_speed_trajectory_mean_std_ax(ax, anchors, data, pan_or_tilt="pan", set_title=False)
        anchors = lists_of_param_anchors["tilt_speed_trajectory"]
        ax = fig.add_subplot(132)
        plot_abs_speed_trajectory_mean_std_ax(ax, anchors, data, pan_or_tilt="tilt", set_title=False, set_ylabel=False)
        anchors = lists_of_param_anchors["vergence_trajectory"]
        ax = fig.add_subplot(133)
        plot_abs_vergence_trajectory_mean_std_ax(ax, anchors, data, set_title=False, set_ylabel=False)

    return


def triple_wrt_error(lists_of_param_anchors, data, turn_2_frames_vergence_on=True):
    for cmap_name in ['viridis', 'Greys', 'Blues', 'GnBu', 'PuBu', 'PuBuGn', 'binary', 'gist_yarg']:
        with FigureManager("triple_policy_cmap_{}.png".format(cmap_name)) as fig:
            axs = []
            anchors = lists_of_param_anchors["wrt_pan_speed_error"]
            ax = fig.add_subplot(131)
            axs.append(ax)
            im = plot_policy_pan_tilt_ax(ax, anchors, data, pan_or_tilt="pan", set_title=False, cmap_name=cmap_name)
            anchors = lists_of_param_anchors["wrt_tilt_speed_error"]
            ax = fig.add_subplot(132)
            axs.append(ax)
            im = plot_policy_pan_tilt_ax(ax, anchors, data, pan_or_tilt="tilt", set_title=False, set_ylabel=False,
                                         cmap_name=cmap_name)
            anchors = lists_of_param_anchors["wrt_vergence_error"]
            ax = fig.add_subplot(133)
            axs.append(ax)
            im = plot_policy_vergence_ax(ax, anchors, data, set_title=False, set_ylabel=False, cmap_name=cmap_name)
            fig.colorbar(im, ax=axs)

    with FigureManager("triple_recerr_wrt_error.png") as fig:
        anchors = lists_of_param_anchors["wrt_pan_speed_error"]
        ax = fig.add_subplot(131)
        plot_recerr_wrt_speed_error_ax(ax, anchors, data, pan_or_tilt="pan", set_title=False, inset=False, legend=False,
                                       ylim=None)
        anchors = lists_of_param_anchors["wrt_tilt_speed_error"]
        ax = fig.add_subplot(132)
        plot_recerr_wrt_speed_error_ax(ax, anchors, data, pan_or_tilt="tilt", set_ylabel=False, set_title=False,
                                       inset=False, legend=False, ylim=None)
        anchors = lists_of_param_anchors["wrt_vergence_error"]
        ax = fig.add_subplot(133)
        plot_recerr_wrt_vergence_error_ax(ax, anchors, data, set_ylabel=False, set_title=False, inset=False,
                                          turn_2_frames_vergence_on=turn_2_frames_vergence_on, legend=True, ylim=None)

    with FigureManager("triple_recerr_wrt_error_per_scale.png") as fig:
        anchors = lists_of_param_anchors["wrt_pan_speed_error"]
        ax = fig.add_subplot(131)
        plot_recerr_wrt_speed_error_per_scale_ax(ax, anchors, data, pan_or_tilt="pan", set_title=False, legend=False)
        anchors = lists_of_param_anchors["wrt_tilt_speed_error"]
        ax = fig.add_subplot(132)
        plot_recerr_wrt_speed_error_per_scale_ax(ax, anchors, data, pan_or_tilt="tilt", set_ylabel=False,
                                                 set_title=False, legend=False)
        anchors = lists_of_param_anchors["wrt_vergence_error"]
        ax = fig.add_subplot(133)
        plot_recerr_wrt_vergence_error_per_scale_ax(ax, anchors, data, set_ylabel=False, set_title=False, legend=True,
                                                    turn_2_frames_vergence_on=turn_2_frames_vergence_on)

    return


def plot_test(experiment_metadata, test_data_filename, save, overwrite):
    test_plots_dir = experiment_metadata["plot_testing_path"]
    test_data_path = experiment_metadata["test_data_path"] + "/" + test_data_filename
    plots_dir = test_plots_dir + "/" + os.path.splitext(os.path.basename(test_data_path))[0] + "/"
    if save:
        os.makedirs(test_plots_dir, exist_ok=True)
        try:
            os.makedirs(plots_dir, exist_ok=overwrite)
        except Exception as e:
            print(plots_dir, " : file exists")
    FigureManager._path = plots_dir
    FigureManager._save = save
    test_conf_filename = "_".join(test_data_filename.split("_")[1:])
    test_conf_path = "../test_conf/" + test_conf_filename
    lists_of_param_anchors = TestConf.load_test_description(test_conf_path)
    with open(test_data_path, "rb") as f:
        data = pickle.load(f)

    reward_scaling_factor = experiment_metadata["conf"].reward_scaling_factor
    turn_2_frames_vergence_on = experiment_metadata["conf"].turn_2_frames_vergence_on
    if "wrt_pan_speed_error" in lists_of_param_anchors:
        all_wrt_speed_error(lists_of_param_anchors, data, reward_scaling_factor=reward_scaling_factor,
                            pan_or_tilt="pan")
    if "wrt_tilt_speed_error" in lists_of_param_anchors:
        all_wrt_speed_error(lists_of_param_anchors, data, pan_or_tilt="tilt")
    if "wrt_vergence_error" in lists_of_param_anchors:
        all_wrt_vergence_error(lists_of_param_anchors, data, reward_scaling_factor=reward_scaling_factor,
                               turn_2_frames_vergence_on=turn_2_frames_vergence_on)
    if "pan_speed_trajectory" in lists_of_param_anchors:
        all_speed_trajectory(lists_of_param_anchors, data, pan_or_tilt="pan")
    if "tilt_speed_trajectory" in lists_of_param_anchors:
        all_speed_trajectory(lists_of_param_anchors, data, pan_or_tilt="tilt")
    if "vergence_trajectory" in lists_of_param_anchors:
        all_vergence_trajectory(lists_of_param_anchors, data)
    ### Triple plots:
    if "pan_speed_trajectory" in lists_of_param_anchors and \
            "tilt_speed_trajectory" in lists_of_param_anchors and \
            "vergence_trajectory" in lists_of_param_anchors:
        triple_trajectory(lists_of_param_anchors, data)
    if "wrt_pan_speed_error" in lists_of_param_anchors and \
            "wrt_tilt_speed_error" in lists_of_param_anchors and \
            "wrt_vergence_error" in lists_of_param_anchors:
        triple_wrt_error(lists_of_param_anchors, data, turn_2_frames_vergence_on=turn_2_frames_vergence_on)
    ### Depth plot:
    if "depth_trajectory" in lists_of_param_anchors:
        plot_depth_vergence_error(lists_of_param_anchors, data)
        plot_sample_depth_trajectory(lists_of_param_anchors, data)


def plot_sample_depth_trajectory(lists_of_param_anchors, data, n=10):
    anchors = lists_of_param_anchors['depth_trajectory']
    data = filter_data(data, **anchors)
    test_cases = np.array([a for a, b in data])
    data = np.array([b for a, b in data], dtype=data[0][1].dtype)  # np.array of np.ndarray
    random_indices = np.random.randint(len(data), size=n)
    with FigureManager("depth.png") as fig:
        ax = fig.add_subplot(111)
        for i in random_indices:
            object_distance = data[i]["object_distance"]
            ax.plot(range(len(object_distance)), to_angle(object_distance), color="blue", linestyle="--")
            vergence = data[i]["eye_position"][:, -1]
            ax.plot(range(len(object_distance)), vergence, color="red", linestyle="--")
        ax.set(xlabel='Iteration', ylabel='Vergence Angle (Deg)')
        ax.set_title("Example Distance Change (Object & Vergence)")
        ax.axhline(to_angle(0.5), ls='--', color='g')
        ax.axhline(to_angle(5), ls='--', color='g')
    plt.show()


def plot_depth_vergence_error(lists_of_param_anchors, data,):
    anchors = lists_of_param_anchors['depth_trajectory']
    data = filter_data(data, **anchors)
    test_cases = np.array([a for a, b in data])
    data = np.array([b for a, b in data], dtype=data[0][1].dtype)  # np.array of np.ndarray
    #df = pd.DataFrame(data, columns = ['Iteration', 'Vergence_Error'])
    iter = []
    error = []
    for i in data:
        if np.amax(i['object_distance']) > 5.0:
            continue
        if np.amin(i['object_distance']) < 0.5:
            continue
        for it, j in enumerate(to_angle(i['object_distance']) - i['eye_position'][...,-1]):
            iter.append(it)
            error.append(np.abs(j) * 320/90)
    dict = {"Iteration": iter, "Vergence Error": error}
    df = pd.DataFrame(dict)
    error = []
    for i in data:
        error.append(np.clip(to_angle(i['object_distance']) - i['eye_position'][...,-1],-5,5))
    error = np.array(error)
    median = np.median(error, axis=0)
    mean = np.mean(error, axis=0)
    #df = df.rename(columns={"0": "Iteration", "1": "Vergence_Error"})
    ax = sns.lineplot(x="Iteration", y="Vergence Error", data=df)
    #ax = sns.lineplot(x=range(len(median)), y=median)
    ax.axhline(0, ls='--')
    ax.axhline(1, ls='--', color='g')
    ax.axhline(2, ls='--', color='r')
    #ax.axhline(-0.2, ls='--', color='r')
    plt.show()
    # print(data[:]["object_distance"])
    # object_distance = data[:]["object_distance"]
    # eye_position = data[:]["eye_position"]
    # print(eye_position)
    # num_iterations = range(len(eye_position[0]))
    # print(data["eye_position"][300])
    # #mean = np.mean(data["object_distance"], axis=0)
    # mean = np.mean(data["vergence_error"], axis=0)
    # print(mean)
    # print(len(num_iterations))
    # print(len(mean))
    #
    # with FigureManager("depth.png") as fig:
    #     ax = fig.add_subplot(111)
    #     # for i in range(10):
    #     #     random_pos = np.random.randint(len(data["eye_position"]))
    #     #     ax.plot(num_iterations, data["eye_position"][random_pos][..., -1], color="blue", linestyle="--")
    #     #     ax.plot(num_iterations, to_angle(data["object_distance"][random_pos]), color="green", linestyle="--")
    #     #for i in range(len(data)):
    #     #    ax.plot(num_iterations, data["vergence_error"][i], color="blue", linestyle="--")
    #     ax.plot(num_iterations, mean)
    return df



# def plot_test_2(experiment_metadata, test_data_filename, plot_list, save, overwrite):
#     '''
#     Load a single test data file and plots graphs according to plot_list
#     :param plot_list:
#     :param experiment_metadata:
#     :param test_data_filename:
#     :param save:
#     :param overwrite:
#     :return:
#     '''
#     df = None
#     plots_output_dir = experiment_metadata["plot_testing_path"]
#     data_input_path = experiment_metadata["test_data_path"] + "/" + test_data_filename
#     plots_output_subdir = plots_output_dir + "/" + os.path.splitext(os.path.basename(data_input_path))[0] + "/"
#     if save:
#         os.makedirs(plots_output_subdir, exist_ok=True)
#         try:
#             os.makedirs(plots_output_subdir, exist_ok=overwrite)
#         except Exception as e:
#             print(plots_output_subdir, " : file exists")
#     FigureManager._path = plots_output_subdir
#     FigureManager._save = save
#     test_conf_filename = "_".join(test_data_filename.split("_")[1:])
#     test_conf_path = "../test_conf/" + test_conf_filename
#     lists_of_param_anchors = TestConf.load_test_description(test_conf_path)
#     with open(data_input_path, "rb") as f:
#         data = pickle.load(f)
#     for plot_name in plot_list:
#         if plot_name == 'depth_movement':
#             df = plot_depth_movement(lists_of_param_anchors, data)
#         if plot_name == 'sample_depth_path':
#             plot_depth_trajectory(lists_of_param_anchors, data)
#     return df


def plot_all_tests(experiment_metadata, save, overwrite):
    '''
    Generates sets of plots for all testing data folders (& contained files) inside the test_data folder.
    :param experiment_metadata: Metadata collection for the experiment
    :param save: Save or show plot
    :param overwrite: Overwrite files if they exist
    '''
    for test_data_filename in os.listdir(experiment_metadata["test_data_path"]):
        if test_data_filename.endswith(".pkl"):
            plot_test(experiment_metadata, test_data_filename, save, overwrite)


# plot_list = [
#     'depth_movement',
#     'sample_depth_path',
# ]  # List of all the plots that should be generated

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
    # paths = [
    #     "/home/jkling/Desktop/aec/asynchronous_aec/experiments/2020_05_16-22.24.36_mlr5.00e-04_clr5.00e-04__Default_Run_Main_Branch",
    #     "/home/jkling/Desktop/aec/asynchronous_aec/experiments/2020_05_24-16.34.44_mlr5.00e-04_clr5.00e-04__Radial_Depth_Change_0_01_Choice_Episode_Length_20_Rotation_Off/",
    #     "/home/jkling/Desktop/aec/asynchronous_aec/experiments/2020_05_24-16.34.05_mlr5.00e-04_clr5.00e-04__Radial_Depth_Change_0_02_Choice_Episode_Length_20_Rotation_Off/",
    #     "/home/jkling/Desktop/aec/asynchronous_aec/experiments/2020_05_25-08.36.16_mlr5.00e-04_clr5.00e-04__Radial_Depth_Change_0_03_Choice_Episode_Length_20_Rotation_Off/",
    #     "/home/jkling/Desktop/aec/asynchronous_aec/experiments/2020_05_25-08.39.17_mlr5.00e-04_clr5.00e-04__Radial_Depth_Change_0_025_Continuous_Episode_Length_10_Rotation_Off"
    # ]
    # file = [
    #     '0100235_radial_depth_movement.pkl',
    #     '0100162_radial_depth_movement.pkl',
    #     '0100154_radial_depth_movement.pkl',
    #     '0100152_radial_depth_movement.pkl',
    #     '0100152_radial_depth_movement.pkl',
    #         ]
    #
    # label = [
    #     '0 cm',
    #     '1 cm',
    #     '2 cm',
    #     '3 cm',
    #     'Rand. unif. 0 cm - 2.5 cm'
    # ]
    #
    # df_list = []
    # for i, path in enumerate(paths):
    #     experiment_metadata = get_experiment_metadata(path)
    #     df_list.append(plot_test(experiment_metadata, file[i], plot_list, save=args.save, overwrite=args.overwrite))
    #     df_list[-1]["Screen Speed (cm/it)"] = label[i]
    # result = df_list[0]
    # for df in df_list[1:]:
    #     result = result.append(df, ignore_index=True)
    # ax = sns.lineplot(x="Iteration", y="Vergence Error", data=result, hue="Screen Speed (cm/it)",)
    # ax.axhline(0, ls='--')
    # ax.axhline(1, ls='--', color='r')
    # ax.set(xlabel='Iteration', ylabel='Vergence Error (px)')
    # ax.set_title("Mean Absolute Vergence Error (within one episode)")
    # #ax.axhline(-0.2, ls='--', color='r')
    # plt.show()

    experiment_metadata = get_experiment_metadata(args.path)
    plot_all_tests(experiment_metadata, args.save, args.overwrite)
    #plot_test(experiment_metadata, '0100154_sample_test_file.pkl', plot_list, args.save, args.overwrite)
    #plot_test_2(experiment_metadata, '0100235_radial_depth_movement.pkl', args.save, args.overwrite)
