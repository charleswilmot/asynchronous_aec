import matplotlib.pyplot as plt
import pickle
import numpy as np
import argparse
import os
from PIL import Image

OBJECT_DISTANCE = 1
VERGENCE_ERROR = 2


def get_new_ax(fig, n_subplots, subplot_index, methode="square"):
    if methode == "square":
        n = int(np.ceil(np.sqrt(n_subplots)))
        if n != np.sqrt(n_subplots):
            m = n + 1
        else:
            m = n
        return fig.add_subplot(m, n, subplot_index + 1)
    if methode == "horizontal":
        return fig.add_subplot(1, n_subplots, subplot_index + 1)
    if methode == "vertical":
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
    return [data_point for data_point, pred in zip(data, condition) if pred]


def plot_vergence_trajectory_all(fig, lists_of_param_anchors, data):
    # generate sub lists of param anchors
    data = filter_data(data, **lists_of_param_anchors)
    list_of_subdata = [filter_data(data, object_distances=[d]) for d in lists_of_param_anchors["object_distances"]]
    n_subplots = len(lists_of_param_anchors["object_distances"])
    # for each sublist, plot in an ax
    for subplot_index, subdata in enumerate(list_of_subdata):
        ax = get_new_ax(fig, n_subplots, subplot_index, methode="square")
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
        ax = get_new_ax(fig, n_subplots, subplot_index, methode="square")
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
    path = "/home/aecgroup/aecdata/Textures/mcgillManMade_600x600_png_selection/"
    #path = "../local_copies_of_aecgroup/mcgillManMade_600x600_png_selection/"
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


def plot_recerr_wrt_speed_error(fig, lists_of_param_anchors, data):
    data = filter_data(data, **lists_of_param_anchors)
    test_cases = np.array([a for a, b in data])
    # for x in test_cases:
    #     print(x)
    data = np.array([b for a, b in data])
    ax = fig.add_subplot(111)
    mean = None
    n = 0
    for stimulus in lists_of_param_anchors["stimulus"]:
        where = np.where(test_cases["stimulus"] == stimulus)
        subdata = data[where]
        speed_error = subdata["speed_error"][:, 0, 1]
        recerr = subdata["total_reconstruction_error"][:, 0]
        args = np.argsort(speed_error)
        ax.plot(speed_error[args], recerr[args], 'b-', linewidth=1)
        if mean is None:
            mean = recerr[args]
        else:
            mean += recerr[args]
        n += 1
        # print(speed_error[args], recerr[args], "\n\n\n\nhwdhcwbadch\n\n\n")
    ax.plot(speed_error[args], mean / n, 'r-', linewidth=3)
    ax.axvline(90 / 320, color="k", linestyle="--")
    ax.axvline(-90 / 320, color="k", linestyle="--")



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
    args = parser.parse_args()

    fig = plt.figure()

    with open(args.test_conf_path, "rb") as f:
        lists_of_param_anchors = pickle.load(f)["test_descriptions"]
        # lists_of_param_anchors = pickle.load(f)["test_descriptions"]["wrt_tilt_speed_error"]
        # lists_of_param_anchors = pickle.load(f)["test_descriptions"]["vergence_trajectory"]
        #print(lists_of_param_anchors)

    with open(args.path, "rb") as f:
        data = pickle.load(f)


    #plot_stimulus_path(fig, lists_of_param_anchors, data)
    #plot_tilt_path_all(fig, lists_of_param_anchors, data)
    # plot_vergence_trajectory_all(fig, lists_of_param_anchors, data)
    # plt.show()


    pan_anchors = lists_of_param_anchors["wrt_pan_speed_error"]
    tilt_anchors = lists_of_param_anchors["wrt_tilt_speed_error"]
    plot_recerr_wrt_speed_error(fig, pan_anchors, data)
    plt.show()
