import pickle
import numpy as np
import os
from helper.generate_test_conf import TestConf


def read_training_data(path):
    with open(path, "rb") as f:
        pickle_obj_length = np.frombuffer(f.read(4), dtype=np.int32)[0]  # get length of the pickled dtype
        dtype = pickle.loads(f.read(pickle_obj_length))                  # read pickled dtype
        episode_length = np.frombuffer(f.read(4), dtype=np.int32)[0]     # get episode_length
        data = np.fromfile(f, dtype=dtype).reshape((-1, episode_length)) # read all data and reshape according to episode_length
    return data


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


def read_all_abs_testing_performance(path):
    at = 10
    results = {"pan": {}, "tilt": {}, "vergence": {}}
    for filename in os.listdir(path):
        print("reading {}".format(filename))
        splits = filename.split("_")
        iteration = int(splits[0])
        test_conf = "_".join(splits[1:])
        lists_of_param_anchors = TestConf.load_test_description("../test_conf/{}".format(test_conf))
        vergence, pan, tilt = False, False, False
        if "vergence_trajectory" in lists_of_param_anchors:
            vergence = True
        if "pan_speed_trajectory" in lists_of_param_anchors:
            pan = True
        if "tilt_speed_trajectory" in lists_of_param_anchors:
            tilt = True
        if pan or tilt or vergence:
            with open(os.path.join(path, filename), 'rb') as f:
                test_data = pickle.load(f)
            if pan:
                data = filter_data(test_data, **lists_of_param_anchors["pan_speed_trajectory"])
                data = [b for a, b in data]
                errors = np.array(data)["speed_error"][:, at, 1]
                results["pan"][iteration] = np.mean(np.abs(errors))
            if tilt:
                data = filter_data(test_data, **lists_of_param_anchors["tilt_speed_trajectory"])
                data = [b for a, b in data]
                errors = np.array(data)["speed_error"][:, at, 0]
                results["tilt"][iteration] = np.mean(np.abs(errors))
            if vergence:
                data = filter_data(test_data, **lists_of_param_anchors["vergence_trajectory"])
                data = [b for a, b in data]
                errors = np.array(data)["vergence_error"][:, at]
                results["vergence"][iteration] = np.mean(np.abs(errors))
    for key in ["pan", "tilt", "vergence"]:
        # transforms dict into pair of sorted lists
        results[key] = np.array(list(zip(*list(sorted(results[key].items(), key=lambda x: x[0])))))
    return results


if __name__ == "__main__":
    results = read_all_abs_testing_performance("../../experiments/2020_03_12-15.45.43_mlr1.00e-04_clr1.00e-04__replicate_without_patch_and_scale_DQN/test_data/")
