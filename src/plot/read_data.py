import pickle
import numpy as np


def read_training_data(path):
    with open(path, "rb") as f:
        pickle_obj_length = np.frombuffer(f.read(4), dtype=np.int32)[0]  # get length of the pickled dtype
        dtype = pickle.loads(f.read(pickle_obj_length))                  # read pickled dtype
        episode_length = np.frombuffer(f.read(4), dtype=np.int32)[0]     # get episode_length
        data = np.fromfile(f, dtype=dtype).reshape((-1, episode_length)) # read all data and reshape according to episode_length
    return data


if __name__ == "__main__":
    fields = [
        'all_rewards',
        'total_recerrs',
        'eyes_speed',
        'object_speed',
        'greedy_actions_indices',
        'sampled_actions_indices',
        'episode_number',
        'patch_target_return',
        'total_target_return',
        'eyes_position',
        'object_distance',
        'scale_rewards',
        'patch_recerrs',
        'scale_recerrs',
        'scale_target_return',
        'patch_rewards'
    ]

    import matplotlib.pyplot as plt

    # data = read_training_data("../../experiments/2020_02_25-09.40.18_mlr1.00e-04_clr1.00e-04__debug/data/training.data")
    data = read_training_data("../../experiments/2020_02_27-15.42.15_mlr1.00e-04_clr1.00e-04__pan_tilt_geometric_action_set_uniform_screen_speed_0_to_1.125_after_fix_buffer_size_2000_crop_size_32_filter_8/data/training.data")

    speed_errors_tilt = data["eyes_speed"][:, -1, 0] + data["object_speed"][:, -1, 0]
    speed_errors_pan = data["eyes_speed"][:, -1, 1] - data["object_speed"][:, -1, 1]

    win_size = 500
    stddev = 200
    kernel = np.exp(-(np.arange(-win_size / 2, win_size / 2) ** 2) / stddev ** 2)
    kernel /= np.sum(kernel)
    a = np.convolve(np.abs(speed_errors_tilt), kernel, mode="valid")
    b = np.convolve(np.abs(speed_errors_pan), kernel, mode="valid")
    x = np.arange(win_size / 2, win_size / 2 + a.shape[0]) * data["eyes_speed"].shape[1]
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111)
    ax.plot(x, a, label="tilt")
    ax.plot(x, b, label="pan")
    ax.axhline(90 / 320, color="k", linestyle="--")
    ax.set_ylim([-0.1, 1.6])
    ax.legend()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Speed error in deg/it")
    ax.set_title("Speed error wrt train time")
    # fig.savefig("/tmp/pan_tilt.png")
    plt.show()
