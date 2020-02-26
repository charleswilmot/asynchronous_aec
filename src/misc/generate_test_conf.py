import numpy as np
import pickle
import numpy
from itertools import product, starmap


dttest_case = np.dtype([
    ("stimulus", np.int32),
    ("object_distance", np.float32),
    ("vergence_error", np.float32),
    ("speed_error", (np.float32, 2)),
    ("n_iterations", np.int32)
])


def test_case(stimulus, object_distance, vergence_error, speed_error, n_iterations):
    return np.array((
        int(stimulus),
        float(object_distance),
        float(vergence_error),
        (float(speed_error[0]), float(speed_error[1])),
        int(n_iterations)), dtype=dttest_case)


def test_cases_between(stimulus, object_distances, vergence_errors, speed_errors, n_iterations):
    total_length = np.product(
        [len(x) for x in [stimulus, object_distances, vergence_errors, speed_errors, n_iterations]])
    ret = np.zeros(total_length, dtype=dttest_case)
    for i, case in enumerate(starmap(test_case, product(stimulus, object_distances, vergence_errors, speed_errors, n_iterations))):
        ret[i] = case
    return ret


def generate_vergence_trajectory(stimulus, object_distances, vergence_errors):
    lists = {
        "stimulus": stimulus,
        "object_distances": object_distances,
        "vergence_errors": vergence_errors,
        "speed_errors": [(0, 0)],
        "n_iterations": [20]
    }
    test_description = {"vergence_trajectory": lists}
    test_cases = test_cases_between(**lists)
    return test_description, test_cases


def generate_wrt_speed_error(stimulus, n_speed_errors, pan_or_tilt="tilt"):
    min_action = 90 / 320 / 2
    speed_errors = np.arange(- min_action * n_speed_errors, min_action * (n_speed_errors + 1), min_action)
    speed_errors = [(0, err) for err in speed_errors] if pan_or_tilt == "pan" else [(err, 0) for err in speed_errors]
    lists = {
        "stimulus": stimulus,
        "object_distances": [1],
        "vergence_errors": [0],
        "speed_errors": speed_errors,
        "n_iterations": [1]
    }
    test_description = {"wrt_{}_speed_error".format(pan_or_tilt): lists}
    test_cases = test_cases_between(**lists)
    return test_description, test_cases


def update_test_conf(test_conf, test_description, test_cases):
    test_conf["test_descriptions"].update(test_description)
    if test_conf["test_cases"] is None:
        test_conf["test_cases"] = test_cases
    else:
        test_conf["test_cases"] = np.unique(np.concatenate([test_conf["test_cases"], test_cases], axis=0))
    return test_conf


if __name__ == "__main__":
    # test_conf = {
    #    "test_descriptions":{"test_name_1": test_anchors_lists, "test_name_2": test_anchors_lists},
    #    "test_cases": big_array_of_type_dttest_case
    # }
    test_conf = {"test_descriptions": {}, "test_cases": np.zeros(0, dtype=dttest_case)}

    # test_description, test_cases = generate_vergence_trajectory(range(20), [0.5, 1, 2, 3, 3.5], [-2, -1, 1, 2])
    # update_test_conf(test_conf, test_description, test_cases)

    n_speed_errors = 30
    stimulus = range(20)
    # test_description, test_cases = generate_wrt_speed_error(stimulus, n_speed_errors, pan_or_tilt="tilt")
    # update_test_conf(test_conf, test_description, test_cases)
    test_description, test_cases = generate_wrt_speed_error(stimulus, n_speed_errors, pan_or_tilt="pan")
    update_test_conf(test_conf, test_description, test_cases)

    with open("../../test_conf/delete_me_plz.pkl", "wb") as f:
        pickle.dump(test_conf, f)
