import os
import numpy as np
import pickle
from algorithm.environment import Environment
from helper.utils import to_angle
from itertools import product, starmap

dttest_case = np.dtype([
    ("stimulus", np.int32),
    ("object_distance", np.float32),
    ("vergence_error", np.float32),
    ("speed_error", (np.float32, 2)),
    ("depth_speed", np.float32),
    ("n_iterations", np.int32),
    # ('object_radial_direction')

])

min_action = 90 / 320 / 2


class TestConf:
    def __init__(self, stimulus=None, object_distances=None, vergence_errors=None, speed_errors=None, depth_speed=None,
                 n_iterations=None):
        self.default_stimulus = stimulus
        self.default_object_distances = object_distances
        self.default_vergence_errors = vergence_errors
        self.default_speed_errors = speed_errors
        self.default_depth_speed = depth_speed
        self.default_n_iterations = n_iterations
        self.data = {
            "test_description": {},
            "test_cases_policy_dependent": np.zeros(shape=0, dtype=dttest_case),
            "test_cases_policy_independent": np.zeros(shape=0, dtype=dttest_case)
        }
        self.policy_independent_inputs = None

    # ToDo: Proper data representation of data in TestConf
    def __repr__(self):
        print(self.data)

    def dump(self, path):
        self.generate_policy_independent_inputs()
        policy_independent_inputs = self.policy_independent_inputs
        self.policy_independent_inputs = None
        pickle_bytes = pickle.dumps(self)
        pickle_size = len(pickle_bytes)
        with open(path, "wb") as f:
            f.write(np.int32(pickle_size))
            f.write(pickle_bytes)
            f.write(policy_independent_inputs.tobytes())

    @classmethod
    def load(cls, path):
        input_size = os.path.getsize(path)
        with open(path, 'rb') as f:
            pickle_size = np.frombuffer(f.read(4), dtype=np.int32)[0]
            me = pickle.loads(f.read(pickle_size))
            policy_independent_inputs = np.fromfile(f, dtype=np.float32).reshape((-1, 4, 240, 320, 3))
        me.policy_independent_inputs = policy_independent_inputs
        return me

    @classmethod
    def load_test_description(cls, path):
        input_size = os.path.getsize(path)
        with open(path, 'rb') as f:
            pickle_size = np.frombuffer(f.read(4), dtype=np.int32)[0]
            me = pickle.loads(f.read(pickle_size))
        return me.data["test_description"]

    def add(self, plot_name, **anchors):
        self.data["test_description"][plot_name] = anchors
        test_cases = test_cases_between(**anchors)
        if anchors["n_iterations"] == [1]:
            self.add_to_policy_independent(test_cases)
        else:
            self.add_to_policy_dependent(test_cases)

    def generate_policy_independent_inputs(self):
        if self.policy_independent_inputs is None:
            n = len(self.data["test_cases_policy_independent"])
            height = 240
            width = 320
            self.policy_independent_inputs = np.zeros((n, 4, height, width, 3), dtype=np.float32)
            if n > 0:
                env = Environment()
                for i, test_case in enumerate(self.data["test_cases_policy_independent"]):
                    print(" " * 80, end="\r")
                    print("generating input for test case: {: 4d}/ {: 4d}\t{}".format(i + 1, n, test_case), end="\r")
                    # place_robot preinit
                    vergence_init = to_angle(test_case["object_distance"]) + test_case["vergence_error"]
                    screen_speed = -test_case["speed_error"]
                    env.robot.reset_speed()
                    env.robot.set_position([0, 0, vergence_init], joint_limit_type="none")
                    env.screen.set_texture(test_case["stimulus"])
                    env.screen.set_trajectory(
                        test_case["object_distance"],
                        screen_speed[0],
                        screen_speed[1],
                        test_case["depth_speed"],
                        preinit=True)
                    env.step()
                    left_cam_before, right_cam_before = env.robot.get_vision()
                    # place_robot iteration 0
                    env.step()
                    left_cam, right_cam = env.robot.get_vision()
                    self.policy_independent_inputs[i] = np.stack(
                        [left_cam_before, right_cam_before, left_cam, right_cam], axis=0)
                print("")

    def add_to_policy_independent(self, test_cases):
        self.policy_independent_inputs = None
        self.data["test_cases_policy_independent"] = np.union1d(self.data["test_cases_policy_independent"], test_cases)

    def add_to_policy_dependent(self, test_cases):
        self.data["test_cases_policy_dependent"] = np.union1d(self.data["test_cases_policy_dependent"], test_cases)

    def add_vergence_trajectory(self, stimulus=None, object_distances=None, vergence_errors=None, n_iterations=None):
        IMPOSED_SPEED_ERROR = [(0, 0)]
        IMPOSED_DEPTH_SPEED = [0]
        self.add("vergence_trajectory",
                 stimulus=stimulus if stimulus is not None else self.default_stimulus,
                 object_distances=object_distances if object_distances is not None else self.default_object_distances,
                 vergence_errors=vergence_errors if vergence_errors is not None else self.default_vergence_errors,
                 speed_errors=IMPOSED_SPEED_ERROR,
                 depth_speed=IMPOSED_DEPTH_SPEED,
                 n_iterations=n_iterations if n_iterations is not None else self.default_n_iterations
                 )

    def add_depth_trajectory(self, stimulus=None, object_distances=None, depth_speed=None, n_iterations=None):
        IMPOSED_SPEED_ERROR = [(0, 0)]
        IMPOSED_VERGENCE_ERROR = [0]
        self.add("depth_trajectory",
                 stimulus=stimulus if stimulus is not None else self.default_stimulus,
                 object_distances=object_distances if object_distances is not None else self.default_object_distances,
                 vergence_errors=IMPOSED_VERGENCE_ERROR,
                 speed_errors=IMPOSED_SPEED_ERROR,
                 depth_speed=depth_speed if depth_speed is not None else self.default_depth_speed,
                 n_iterations=n_iterations if n_iterations is not None else self.default_n_iterations
                 )

    def add_speed_trajectory(self, pan_or_tilt, stimulus=None, object_distances=None, speed_errors=None, n_iterations=None):
        IMPOSED_VERGENCE_ERROR = [0]
        IMPOSED_DEPTH_SPEED = [0]
        speed_errors = speed_errors if speed_errors is not None else self.default_speed_errors
        if pan_or_tilt == "tilt":
            to_stack = [speed_errors, np.zeros(len(speed_errors))]
        else:
            to_stack = [np.zeros(len(speed_errors)), speed_errors]
        speed_errors = np.stack(to_stack, axis=-1)
        self.add("{}_speed_trajectory".format(pan_or_tilt),
                 stimulus=stimulus if stimulus is not None else self.default_stimulus,
                 object_distances=object_distances if object_distances is not None else self.default_object_distances,
                 vergence_errors=IMPOSED_VERGENCE_ERROR,
                 speed_errors=speed_errors if speed_errors is not None else self.default_speed_errors,
                 depth_speed=IMPOSED_DEPTH_SPEED,
                 n_iterations=n_iterations if n_iterations is not None else self.default_n_iterations
                 )

    def add_wrt_vergence_error(self, vergence_error_bound_in_px, stimulus=None, object_distances=None,
                               depth_speed=None, ):
        IMPOSED_N_ITERATIONS = [1]
        IMPOSED_SPEED_ERROR = [(0, 0)]
        IMPOSED_DEPTH_SPEED = [0]
        vergence_error = vergence_error_bound_in_px * 90 / 320
        vergence_errors = np.arange(-vergence_error, vergence_error + min_action, min_action)
        self.add("wrt_vergence_error",
                 stimulus=stimulus if stimulus is not None else self.default_stimulus,
                 object_distances=object_distances if object_distances is not None else self.default_object_distances,
                 vergence_errors=vergence_errors,
                 speed_errors=IMPOSED_SPEED_ERROR,
                 depth_speed=IMPOSED_DEPTH_SPEED,
                 n_iterations=IMPOSED_N_ITERATIONS
                 )

    def add_wrt_speed_error(self, pan_or_tilt, speed_error_bound_in_px, stimulus=None, object_distances=None,
                            depth_speed=None, ):
        IMPOSED_N_ITERATIONS = [1]
        IMPOSED_VERGENCE_ERROR = [0]
        IMPOSED_DEPTH_SPEED = [0]
        speed_error = speed_error_bound_in_px * 90 / 320
        speed_errors = np.arange(-speed_error, speed_error + min_action, min_action)
        if pan_or_tilt == "tilt":
            to_stack = [speed_errors, np.zeros(len(speed_errors))]
        else:
            to_stack = [np.zeros(len(speed_errors)), speed_errors]
        speed_errors = np.stack(to_stack, axis=-1)
        self.add("wrt_{}_speed_error".format(pan_or_tilt),
                 stimulus=stimulus if stimulus is not None else self.default_stimulus,
                 object_distances=object_distances if object_distances is not None else self.default_object_distances,
                 vergence_errors=IMPOSED_VERGENCE_ERROR,
                 speed_errors=speed_errors,
                 depth_speed=IMPOSED_DEPTH_SPEED,
                 n_iterations=IMPOSED_N_ITERATIONS
                 )

    def add_test_case_video(self, stimulus=None, object_distances=None, vergence_errors=None, speed_errors=None,
                            depth_speed=None, n_iterations=None):
        speed_errors = speed_errors if speed_errors is not None else self.default_speed_errors
        to_stack_tilt = [speed_errors, np.zeros(len(speed_errors))]
        speed_errors_tilt = np.stack(to_stack_tilt, axis=-1)
        to_stack_pan = [np.zeros(len(speed_errors)), speed_errors]
        speed_errors_pan = np.stack(to_stack_pan, axis=-1)
        speed_errors = np.concatenate([speed_errors_tilt, speed_errors_pan], axis=0)
        self.add("test_case_video",
                stimulus=stimulus if stimulus is not None else self.default_stimulus,
                object_distances=object_distances if object_distances is not None else self.default_object_distances,
                vergence_errors=vergence_errors if vergence_errors is not None else self.default_vergence_errors,
                speed_errors=speed_errors if speed_errors is not None else self.default_speed_errors,
                depth_speed=depth_speed if depth_speed is not None else self.default_depth_speed,
                n_iterations=n_iterations if n_iterations is not None else self.default_n_iterations
                )


def test_case(stimulus, object_distance, vergence_error, speed_error, depth_speed, n_iterations):
    return np.array((
        int(stimulus),
        float(object_distance),
        float(vergence_error),
        (float(speed_error[0]), float(speed_error[1])),
        float(depth_speed),
        int(n_iterations)), dtype=dttest_case)


def test_cases_between(stimulus, object_distances, vergence_errors, speed_errors, depth_speed, n_iterations):
    total_length = np.product(
        [len(x) for x in [stimulus, object_distances, vergence_errors, speed_errors, depth_speed, n_iterations]])
    ret = np.zeros(total_length, dtype=dttest_case)
    for i, case in enumerate(starmap(test_case, product(stimulus, object_distances, vergence_errors, speed_errors,
                                                        depth_speed, n_iterations))):
        ret[i] = case
    return ret


# def generate_vergence_trajectory(stimulus, object_distances, vergence_errors):
#     lists = {
#         "stimulus": stimulus,
#         "object_distances": object_distances,
#         "vergence_errors": vergence_errors,
#         "speed_errors": [(0, 0)],
#         "n_iterations": [20]
#     }
#     test_description = {"vergence_trajectory": lists}
#     test_cases = test_cases_between(**lists)
#     return test_description, test_cases


# def generate_wrt_speed_error(stimulus, n_speed_errors, pan_or_tilt="tilt"):
#     min_action = 90 / 320 / 2
#     speed_errors = np.arange(- min_action * n_speed_errors, min_action * (n_speed_errors + 1), min_action)
#     speed_errors = [(0, err) for err in speed_errors] if pan_or_tilt == "pan" else [(err, 0) for err in speed_errors]
#     lists = {
#         "stimulus": stimulus,
#         "object_distances": [1],
#         "vergence_errors": [0],
#         "speed_errors": speed_errors,
#         "n_iterations": [1]
#     }
#     test_description = {"wrt_{}_speed_error".format(pan_or_tilt): lists}
#     test_cases = test_cases_between(**lists)
#     return test_description, test_cases


# def generate_wrt_vergence_error(stimulus, n_vergence_errors):
#     min_action = 90 / 320 / 2
#     vergence_errors = np.arange(- min_action * n_vergence_errors, min_action * (n_vergence_errors + 1), min_action)
#     lists = {
#         "stimulus": stimulus,
#         "object_distances": [1],
#         "speed_errors": [(0, 0)],
#         "vergence_errors": vergence_errors,
#         "n_iterations": [1]
#     }
#     test_description = {"wrt_vergence_error": lists}
#     test_cases = test_cases_between(**lists)
#     return test_description, test_cases


# def generate_speed_trajectory(stimulus, speeds, pan_or_tilt="tilt"):
#     speeds = np.array([(0, s) for s in speeds]) if pan_or_tilt == "pan" else np.array([(s, 0) for s in speeds])
#     lists = {
#         "stimulus": stimulus,
#         "object_distances": [1],
#         "vergence_errors": [0],
#         "speed_errors": speeds,
#         "n_iterations": [20]
#     }
#     test_description = {"{}_speed_trajectory".format(pan_or_tilt): lists}
#     test_cases = test_cases_between(**lists)
#     return test_description, test_cases


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

    # test_conf = {"test_descriptions": {}, "test_cases": np.zeros(0, dtype=dttest_case)}
    # n_speed_errors = 30
    # n_vergence_errors = n_speed_errors
    # stimulus = range(20)
    # errors = [90 / 320 * i for i in [-2, -4, -8, 2, 4, 8]]
    #
    # test_description, test_cases = generate_wrt_speed_error(stimulus, n_speed_errors, pan_or_tilt="tilt")
    # update_test_conf(test_conf, test_description, test_cases)
    # test_description, test_cases = generate_wrt_speed_error(stimulus, n_speed_errors, pan_or_tilt="pan")
    # update_test_conf(test_conf, test_description, test_cases)
    # test_description, test_cases = generate_wrt_vergence_error(stimulus, n_vergence_errors)
    # update_test_conf(test_conf, test_description, test_cases)
    # test_description, test_cases = generate_speed_trajectory(stimulus, errors, pan_or_tilt="tilt")
    # update_test_conf(test_conf, test_description, test_cases)
    # test_description, test_cases = generate_speed_trajectory(stimulus, errors, pan_or_tilt="pan")
    # update_test_conf(test_conf, test_description, test_cases)
    # test_description, test_cases = generate_vergence_trajectory(stimulus, [1], errors)
    # update_test_conf(test_conf, test_description, test_cases)
    #
    #
    # with open("../../test_conf/test_pan_tilt_vergence.pkl", "wb") as f:
    #     pickle.dump(test_conf, f)

    '''
    New content here - delete above if not needed!
    '''

    # stimulus = range(20)
    # object_distance = [0.5, 2, 3.5, 5]
    # # Errors in deg
    # speed_and_vergence_errors = [90 / 320 * i for i in
    #                              [-0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4]]
    # depth_speed = [-0.03, -0.01, 0.0, 0.01, 0.03]
    # iterations = [20]
    #
    # bound_in_px = 8
    #
    # test_conf = TestConf(
    #     stimulus=stimulus,
    #     object_distances=object_distance,
    #     speed_errors=speed_and_vergence_errors,
    #     vergence_errors=speed_and_vergence_errors,
    #     depth_speed=depth_speed,
    #     n_iterations=iterations
    # )
    #
    # test_conf.add_depth_trajectory()
    # test_conf.add_vergence_trajectory()
    # test_conf.add_speed_trajectory("tilt")
    # test_conf.add_speed_trajectory("pan")
    #
    # test_conf.add_wrt_vergence_error(bound_in_px)
    # test_conf.add_wrt_speed_error("tilt", bound_in_px)
    # test_conf.add_wrt_speed_error("pan", bound_in_px)

    #For Video Test Cases
    stimulus = [2, 4, 8, 16]
    object_distance = [1, 3, 5]
    speed_and_vergence_errors = [90 / 320 * i for i in [-1.5, -4,  1.5, 4]]
    depth_speed = [-0.02, 0.0, 0.02]
    iterations = [30]

    test_conf = TestConf(
        stimulus=stimulus,
        object_distances=object_distance,
        speed_errors=speed_and_vergence_errors,
        vergence_errors=speed_and_vergence_errors,
        depth_speed=depth_speed,
        n_iterations=iterations
    )

    test_conf.add_test_case_video(vergence_errors=[0])
    test_conf.dump("../test_conf/video_test_cases.pkl")
