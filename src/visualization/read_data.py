import pickle
import numpy as np


def read_training_data(path):
    with open(path, "rb") as f:
        pickle_obj_length = np.frombuffer(f.read(4), dtype=np.int32)[0]
        dtype = pickle.loads(f.read(pickle_obj_length))
        episode_length = np.frombuffer(f.read(4), dtype=np.int32)[0]
        data = np.fromfile(f, dtype=dtype).reshape((-1, episode_length))
    return data


if __name__ == "__main__":
    fields = [
        'all_rewards',
        'all_recerrs',
        'eyes_speed',
        'object_speed',
        'greedy_actions_indices',
        'sampled_actions_indices',
        'episode_number',
        'patch_target_return',
        'all_target_return',
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
    data = read_training_data("../../experiments/2020_02_25-12.38.33_mlr1.00e-04_clr1.00e-03__debug/data/training.data")

    speed_errors = data["eyes_speed"][:, -1] - data["object_speed"][:, -1]

    win_size = 500
    stddev = 200
    kernel = np.exp(-(np.arange(-win_size / 2, win_size / 2) ** 2) / stddev ** 2)
    kernel /= np.sum(kernel)
    a = np.convolve(np.abs(speed_errors)[:, 0], kernel, mode="valid")
    b = np.convolve(np.abs(speed_errors)[:, 1], kernel, mode="valid")
    x = np.arange(win_size / 2, win_size / 2 + a.shape[0]) * data["eyes_speed"].shape[1]
    plt.plot(x, a)
    plt.plot(x, b)
    plt.show()
