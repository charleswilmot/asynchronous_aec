import pickle
import numpy as np
import os
from helper.generate_test_conf import TestConf
from algorithm.conf import Conf


def read_training_data(path):
    """
    Read the data collected during data - stored in a pickled file.
    First lines of the file define the dtype of the continuously written data to file during training.
    :param path: Path to folder with log file. Folder data is used.
    :return: np array with training data log
    """
    with open(path + "/training.data", "rb") as f:
        pickle_obj_length = np.frombuffer(f.read(4), dtype=np.int32)[0]  # get length of the pickled dtype
        dtype = pickle.loads(f.read(pickle_obj_length))                  # read pickled dtype
        episode_length = np.frombuffer(f.read(4), dtype=np.int32)[0]     # get episode_length
        data = np.fromfile(f, dtype=dtype).reshape((-1, episode_length)) # read all data and reshape according to episode_length
    return data


def filter_data(data, stimulus=None, object_distances=None, vergence_errors=None, speed_errors=None, depth_speed=None, n_iterations=None):
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
    if depth_speed is not None:
        condition = np.logical_and(condition, np.isin(test_cases["depth_speed"], depth_speed))
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


def get_experiment_metadata(experiment_path):
    ret = {}
    ret["experiment_path"] = os.path.normpath(experiment_path)
    ret["train_data_path"] = os.path.normpath(ret["experiment_path"] + "/data")
    ret["test_data_path"] = os.path.normpath(ret["experiment_path"] + "/test_data")
    ret["plot_testing_path"] = os.path.normpath(ret["experiment_path"] + "/test_plots")
    ret["plot_training_path"] = os.path.normpath(ret["experiment_path"] + "/train_plots")
    ret["merge_path"] = os.path.normpath(ret["experiment_path"] + "/merge")
    with open(os.path.normpath(ret["experiment_path"] + "/conf/worker_conf.pkl"), "rb") as f:
        ret["conf"] = pickle.load(f)
    with open(os.path.normpath(ret["experiment_path"] + "/conf/test_conf_path.txt"), "r") as f:
        ret["test_conf_path"] = os.path.normpath(f.readline().replace("\n", ""))
    return ret


if __name__ == "__main__":
    # results = read_all_abs_testing_performance("../../experiments/2020_03_12-15.45.43_mlr1.00e-04_clr1.00e-04__replicate_without_patch_and_scale_DQN/test_data/")
    #print(get_experiment_metadata("../experiments/2020_04_02-09.11.34_mlr5.00e-04_clr5.00e-04__12_workers_compression_4_16"))
    data = read_training_data("/home/jkling/Desktop/aec/asynchronous_aec/experiments/2020_05_24-16.34.05_mlr5.00e-04_clr5.00e-04__Radial_Depth_Change_0_02_Choice_Episode_Length_20_Rotation_Off/data")
    print(data["object_distance"][-4:])
    """
    dtype of the numpy ndarray:
    [('sampled_actions_indices', '<i4', (3,)), ('greedy_actions_indices', '<i4', (3,)), ('patch_recerrs', '<f4', (2, 2, 7, 7)), ('scale_recerrs', '<f4', (2, 2)), ('total_recerrs', '<f4', (2,)), ('all_rewards', '<f4', (2,)), ('total_target_return', '<f4', (2,)), ('object_distance', '<f4'), ('object_speed', '<f4', (2,)), ('eyes_position', '<f4', (3,)), ('eyes_speed', '<f4', (2,)), ('episode_number', '<i4')]
    
    or from worker.py:
        self.training_data_type = np.dtype([
            ("sampled_actions_indices", np.int32, (self.n_joints)),
            ("greedy_actions_indices", np.int32, (self.n_joints)),
            ("patch_recerrs", np.float32, (self.n_encoders, self.n_scales, n_patches, n_patches)),
            ("scale_recerrs", np.float32, (self.n_encoders, self.n_scales)),
            ("total_recerrs", np.float32, (self.n_encoders,)),
            ("all_rewards", np.float32, (self.n_encoders,)),
            ("total_target_return", np.float32, (self.n_encoders,)),
            ("object_distance", np.float32, ()),
            ("object_speed", np.float32, (2, )),
            ("eyes_position", np.float32, (3, )),
            ("eyes_speed", np.float32, (2, )),
            ("episode_number", np.int32, ())
        ])
    """