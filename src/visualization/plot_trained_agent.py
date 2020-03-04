import pickle
import numpy as np
import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


cdict = {'red':   ((0.0,  1.0, 1.0),
                   (0.5,  0.0, 0.0),
                   (1.0,  0.0, 0.0)),
         'blue': ((0.00, 0.0, 0.0),
                  (0.25, 0.0, 0.0),
                  (0.50, 1.0, 1.0),
                  (0.75, 0.0, 0.0),
                  (1.00, 0.0, 0.0)),
         'green':  ((0.0,  0.0, 0.0),
                   (0.5,  0.0, 0.0),
                   (1.0,  1.0, 1.0))
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


def filter_data(data, stimulus=None, object_distances=None, vergence_errors=None, speed_errors=None, n_iterations=None):
    test_cases = np.array([a for a, b in data])
    condition = np.ones_like(test_cases, dtype=np.bool)
    if stimulus is not None:
        condition = np.logical_and(condition, np.isin(test_cases["stimulus"], stimulus))
    if object_distances is not None:
        condition = np.logical_and(condition, np.isin(test_cases["object_distance"], object_distances))
    if vergence_errors is not None:
        condition = np.logical_and(condition, np.isin(test_cases["vergence_error"], vergence_errors))
    if speed_errors is not None:
        speed_errors = np.array(speed_errors, dtype=np.float32).view(dtype='f,f')
        speed_errors2 = np.ascontiguousarray(test_cases["speed_error"].reshape((-1,))).view(dtype='f,f')
        condition = np.logical_and(condition, np.isin(speed_errors2, speed_errors))
    if n_iterations is not None:
        condition = np.logical_and(condition, np.isin(test_cases["n_iterations"], n_iterations))
    return [data_point for data_point, pred in zip(data, condition) if pred]


def plot_vergence_trajectory_all(fig, lists_of_param_anchors, data):
    # generate sub lists of param anchors
    data = filter_data(data, **lists_of_param_anchors)
    list_of_subdata = [filter_data(data, object_distances=[d]) for d in lists_of_param_anchors["object_distances"]]
    n_subplots = len(lists_of_param_anchors["object_distances"])
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
    ax.boxplot(test_data["vergence_error"], notch=True, showfliers=False) #, whis=[10, 90])
    ax.axhline(0, color="k")
    ax.axhline(90 / 320, color="k", linestyle="--")
    ax.axhline(-90 / 320, color="k", linestyle="--")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Vergence error (deg)")
    ax.set_ylim([np.min(test_cases["vergence_error"]) - 1, np.max(test_cases["vergence_error"]) + 1])
    ax.set_title("Object distance  {:.4f}".format(data[0][0]["object_distance"]))


def plot_tilt_path_all(fig, lists_of_param_anchors, data):
    # generate sub lists of param anchors
    data = filter_data(data, **lists_of_param_anchors)
    list_of_subdata = [filter_data(data, object_distances=[d]) for d in lists_of_param_anchors["object_distances"]]
    n_subplots = len(lists_of_param_anchors["object_distances"])
    # for each sublist, plot in an ax
    for subplot_index, subdata in enumerate(list_of_subdata):
        ax = get_new_ax(fig, n_subplots, subplot_index, method="square")
        plot_tilt_path(ax, subdata)


def plot_tilt_path(ax, data):
    test_data = np.array([b for a, b in data])
    result = test_data["eye_position"][np.random.randint(test_data["eye_position"].shape[0], size=40), :]
    for elem in result:
        ax.plot(range(len(elem)), elem[:, 0])


def plot_vergence_path(fig, lists_of_param_anchors, data):
    data = filter_data(data, **lists_of_param_anchors)
    test_data = np.array([b for a, b in data])
    n_subplots = 1
    ax = get_new_ax(fig, n_subplots, 0)
    result = test_data["eye_position"][np.random.randint(test_data["eye_position"].shape[0], size=40), :]
    print(result)
    print(np.shape(result))
    print(result[0][2])
    for elem in result:
        ax.plot(range(len(elem)), elem[:, 2], )


#TODO: Vlean this method
def plot_stimulus_path(fig, lists_of_param_anchors, data):
    path = "/home/aecgroup/aecdata/Textures/mcgillManMade_600x600_bmp_selection/"
    #path = "../local_copies_of_aecgroup/mcgillManMade_600x600_bmp_selection/"
    textures_names = os.listdir(path)
    textures_list = [np.array(Image.open(path + name), dtype=np.uint8) for name in textures_names]
    lim_stimulus = min(10, len(lists_of_param_anchors["stimulus"]))
    data = filter_data(data, **lists_of_param_anchors)
    list_of_subdata = [filter_data(data, stimulus=[s]) for s in lists_of_param_anchors["stimulus"][:lim_stimulus]]
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
        #ax.set_ylim([np.min(test_cases["vergence_error"]) - 1, np.max(test_cases["vergence_error"]) + 1])
        ax.set_title("Stimulus  {:.1f}".format(elem[0][0]["stimulus"]))
        ax2 = get_new_ax(fig, n_subplots, (subplot_index * 2) + 1)
        ax2.imshow(textures_list[elem[0][0]["stimulus"]])


def plot_recerr_wrt_speed_error(fig, lists_of_param_anchors, data, pan_or_tilt="pan"):
    joint_index = 1 if pan_or_tilt == "pan" else 0
    data = filter_data(data, **lists_of_param_anchors)
    test_cases = np.array([a for a, b in data])
    data = np.array([b for a, b in data])
    ax = fig.add_subplot(111)
    mean = None
    n = 0
    one_pixel = 90 / 320
    for stimulus in lists_of_param_anchors["stimulus"]:
        where = np.where(test_cases["stimulus"] == stimulus)
        subdata = data[where]
        speed_error = subdata["speed_error"][:, 0, joint_index]
        recerr = subdata["total_reconstruction_error"][:, 0]
        args = np.argsort(speed_error)
        ax.plot(speed_error[args] / one_pixel, recerr[args], 'b-', linewidth=1)
        if mean is None:
            mean = recerr[args]
        else:
            mean += recerr[args]
        n += 1
    ax.plot(speed_error[args] / one_pixel, mean / n, 'r-', linewidth=3, label="mean")
    axins = inset_axes(ax, width="20%", height="20%", borderpad=2)  # , loc=4
    axins.plot(speed_error[args] / one_pixel, mean / n, 'r-')
    axins.axvline(0, color="k", linestyle="--", alpha=0.5)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_title("Mean only")
    ax.set_xlabel("{} speed error in pixels".format(pan_or_tilt))
    ax.set_ylabel("Reconstruction error")
    ax.legend()
    ax.set_title("Reconstruction error wrt speed error")
    ax.axvline(0, color="k", linestyle="--")


def plot_reward_wrt_action_speed_error_pair(fig, lists_of_param_anchors, data, pan_or_tilt="pan", title_supplement=""):
    joint_index = 1 if pan_or_tilt == "pan" else 0
    data = filter_data(data, **lists_of_param_anchors)
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
                rewards = data["total_reconstruction_error"][where_start] - data["total_reconstruction_error"][where_end]
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
    reward_scaling_factor = 100
    im = ax.imshow(reward_matrix * reward_scaling_factor, aspect="auto", extent=[xmin, xmax, ymin, ymax], cmap="seismic")
    # im = ax.imshow(reward_matrix * reward_scaling_factor, aspect="auto", extent=[xmin, xmax, ymin, ymax], cmap=rbg_cmap)
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel('reward', rotation=270)
    ax.axvline(0, color="k", linestyle="--", alpha=0.8)
    ax.set_xlabel("{} speed error in pixel/it".format(pan_or_tilt))
    ax.set_ylabel("action in pixels")
    ax.set_yticks(np.linspace(-(n_actions - 1) / 2, (n_actions - 1) / 2, n_actions))
    ax.set_yticklabels(actions_in_semi_pixels / 2)
    ax.set_title("Reward wrt action / speed error pair" + title_supplement)


def plot_policy(fig, lists_of_param_anchors, data, pan_or_tilt="pan", title_supplement=""):
    joint_index = 1 if pan_or_tilt == "pan" else 0
    data = filter_data(data, **lists_of_param_anchors)
    test_cases = np.array([a for a, b in data])
    data = np.array([b for a, b in data])
    ax = fig.add_subplot(111)
    speed_errors = np.sort(np.unique(test_cases["speed_error"][:, joint_index]))
    actions_in_semi_pixels = np.array([-8, -4, -2, -1, 0, 1, 2, 4, 8])
    n_actions = len(actions_in_semi_pixels)
    result_matrix = np.zeros((actions_in_semi_pixels.shape[0], speed_errors.shape[0]), dtype=np.float32)
    for speed_error_index, speed_error in enumerate(speed_errors):
        where = np.where(data["speed_error"][:, 0, joint_index] == speed_errors[speed_error_index])
        greedy_actions = data["action_index"][where]  # shape = (n_stimulus, 1, 3)
        for action_index in greedy_actions[:, 0, joint_index]:
            result_matrix[action_index, speed_error_index] += 1
        # print(result_matrix[:, speed_error_index])
        result_matrix[:, speed_error_index] /= len(greedy_actions)
    one_pixel = 90 / 320
    im = ax.imshow(result_matrix, aspect="auto", extent=[speed_errors[0] / one_pixel, speed_errors[-1] / one_pixel, -n_actions / 2, n_actions / 2])
    fig.colorbar(im)
    ax.axvline(0, color="k", linestyle="--", alpha=0.8)
    ax.set_xlabel("{} speed error in pixel/it".format(pan_or_tilt))
    ax.set_ylabel("action in pixels")
    ax.set_yticks(np.linspace(-(n_actions - 1) / 2, (n_actions - 1) / 2, n_actions))
    ax.set_yticklabels(actions_in_semi_pixels / 2)
    ax.set_title("Action probabilities" + title_supplement)


def plot_speed_trajectory(fig, lists_of_param_anchors, data, pan_or_tilt="pan"):
    joint_index = 1 if pan_or_tilt == "pan" else 0
    data = filter_data(data, **lists_of_param_anchors)
    test_cases = np.array([a for a, b in data])
    data = np.array([b for a, b in data], dtype=data[0][1].dtype)
    n_speeds = len(lists_of_param_anchors["speed_errors"])
    at = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 19])
    for i, speed in enumerate(lists_of_param_anchors["speed_errors"]):
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
                ax.plot([xprev_right, x_left], [yprev_right, y_left], color=line.get_color(), linewidth=line.get_linewidth())
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

def all_wrt_speed_error(lists_of_param_anchors, data, pan_or_tilt="pan"):
    anchors = lists_of_param_anchors["wrt_{}_speed_error".format(pan_or_tilt)]

    ### plot v shape pan
    with FigureManager("{}_recerr_wrt_speed_error.png".format(pan_or_tilt)) as fig:
        plot_recerr_wrt_speed_error(fig, anchors, data, pan_or_tilt=pan_or_tilt)

    ### plot reward pan
    with FigureManager("{}_reward_wrt_action_speed_error_pair.png".format(pan_or_tilt)) as fig:
        plot_reward_wrt_action_speed_error_pair(fig, anchors, data, pan_or_tilt=pan_or_tilt)

    with FigureManager("{}_policy.png".format(pan_or_tilt)) as fig:
        plot_policy(fig, anchors, data, pan_or_tilt=pan_or_tilt)

    ## plot reward per stimulus pan
    for stimulus in anchors["stimulus"]:
        with FigureManager("{}_reward_wrt_action_speed_error_pair_stimulus_{}.png".format(pan_or_tilt, stimulus)) as fig:
            anchors["stimulus"] = [stimulus]
            plot_reward_wrt_action_speed_error_pair(fig, anchors, data, pan_or_tilt=pan_or_tilt, title_supplement="   (Stimulus #{})".format(stimulus))
    return


def all_speed_trajectory(lists_of_param_anchors, data, pan_or_tilt="pan"):
    anchors = lists_of_param_anchors["{}_speed_trajectory".format(pan_or_tilt)]
    with FigureManager("{}_speed_trajectory.png".format(pan_or_tilt)) as fig:
        plot_speed_trajectory(fig, anchors, data, pan_or_tilt=pan_or_tilt)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'path', metavar="PATH",
        type=str,
        action='store',
        help="Path to the test_data."
    )
    parser.add_argument(
        'test_conf_path', metavar="TEST_CONF_PATH",
        type=str,
        action='store',
        help="Path to the test config file."
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
    test_plots_dir = os.path.dirname(os.path.abspath(args.path)) + "/../test_plots/"
    plots_dir = test_plots_dir + os.path.splitext(os.path.basename(args.path))[0] + "/"
    if args.save:
        os.makedirs(test_plots_dir, exist_ok=True)
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


    with open(args.test_conf_path, "rb") as f:
        lists_of_param_anchors = pickle.load(f)["test_descriptions"]
        # lists_of_param_anchors = pickle.load(f)["test_descriptions"]["vergence_trajectory"]

    with open(args.path, "rb") as f:
        data = pickle.load(f)


    #plot_stimulus_path(fig, lists_of_param_anchors, data)
    #plot_tilt_path_all(fig, lists_of_param_anchors, data)
    # plot_vergence_trajectory_all(fig, lists_of_param_anchors, data)
    # plt.show()

    # if "wrt_pan_speed_error" in lists_of_param_anchors:
    #     all_wrt_speed_error(lists_of_param_anchors, data, pan_or_tilt="pan")
    # if "wrt_tilt_speed_error" in lists_of_param_anchors:
    #     all_wrt_speed_error(lists_of_param_anchors, data, pan_or_tilt="tilt")
    if "pan_speed_trajectory" in lists_of_param_anchors:
        all_speed_trajectory(lists_of_param_anchors, data, pan_or_tilt="pan")
    if "tilt_speed_trajectory" in lists_of_param_anchors:
        all_speed_trajectory(lists_of_param_anchors, data, pan_or_tilt="tilt")
