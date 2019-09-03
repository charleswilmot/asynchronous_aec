import sys
import vrep
from tempfile import TemporaryDirectory
import png
from replay_buffer import Buffer
from returns import tf_returns
import socket
import multiprocessing
import subprocess
import numpy as np
from numpy import pi, log
import tensorflow as tf
import tensorflow.contrib.layers as tl
import time
import os
import filelock
from PIL import Image
import vbridge as vb
import RESSOURCES
import pickle
from itertools import cycle, islice, product


dttest_data = np.dtype([
    ("action_index", (np.int32, 3)),
    ("action_probability", (np.float32, 3)),
    ("action_value", (np.float32, 3)),
    ("critic_value", np.float32),
    ("total_reconstruction_error", np.float32),
    ("eye_position", (np.float32, 3)),
    ("eye_speed", (np.float32, 2)),
    ("speed_error", (np.float32, 2)),
    ("vergence_error", np.float32)
])


anaglyph_matrix = np.array([
    [0.299, 0    , 0    ],
    [0.587, 0    , 0    ],
    [0.114, 0    , 0    ],
    [0    , 0.299, 0.299],
    [0    , 0.587, 0.587],
    [0    , 0.114, 0.114],
    ])


def make_frame(left_image, right_image, object_distance, vergence_error, episode_number, total_episode_number):
    image = np.matmul(np.concatenate([left_image, right_image], axis=-1), anaglyph_matrix).astype(np.uint8)
    image = Image.fromarray(image)
    drawer = ImageDraw.Draw(image)
    string = "Object distance (m): {: .2f}\nVergence error (deg): {: .2f}\nEpisode {: 3d}/{: 3d}".format(
        object_distance, vergence_error, episode_number, total_episode_number
    )
    drawer.text((20,15), string, fill=(255,255,0))
    return np.array(image, dtype=np.uint8)


class ProperDisplay:
    def __init__(self, data, i, n):
        self.data = data
        self.i = i
        self.n = n

    def __repr__(self):
        return "chunksize: {} - chunk {}/{}".format(len(self.data), self.i, self.n)


def repeatlist(it, count):
    return islice(cycle(it), count)


def _chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def chunks(l, n):
    return list(_chunks(l, n))


def get_textures(path):
    filepaths = [path + "/{}".format(x) for x in os.listdir(path) if x.endswith(".bmp") or x.endswith(".png")]
    return np.array([np.array(Image.open(x)) for x in filepaths])


def actions_dict_from_array(actions):
    return {
        "Arm1_to_Arm2_Left": actions[0],
        "Ground_to_Arm1_Left": actions[1],
        "Arm1_to_Arm2_Right": actions[2],
        "Ground_to_Arm1_Right": actions[3]
    }


def lrelu(x):
    alpha = 0.2
    return tf.nn.relu(x) * (1 - alpha) + x * alpha


# def lrelu(x):
#     return tf.tanh(x)


def exponential_moving_stats(ten, alpha):
    mean = tf.reduce_mean(ten, axis=0)
    moving_mean = tf.Variable(tf.zeros_like(mean))
    moving_mean_assign = moving_mean.assign(alpha * moving_mean + (1 - alpha) * mean)
    delta2 = tf.reduce_mean((ten - moving_mean) ** 2, axis=0)
    moving_var = tf.Variable(tf.ones_like(mean))
    moving_var_assign = moving_var.assign(alpha * (moving_var + (1 - alpha) * delta2))
    cond = tf.less(tf.shape(ten)[0], 2)
    moving_mean_cond = tf.cond(cond, lambda: moving_mean, lambda: moving_mean_assign)
    moving_var_cond = tf.cond(cond, lambda: moving_var, lambda: moving_var_assign)
    return moving_mean_cond, tf.sqrt(moving_var_cond)


def normalize(ten, alpha):
    mean, std = exponential_moving_stats(ten, alpha)
    return (ten - mean) / (std + 1e-5)


def get_cluster(n_parameter_servers, n_workers):
    spec = {}
    port = get_available_port(2222)
    for i in range(n_parameter_servers):
        if "ps" not in spec:
            spec["ps"] = []
        port = get_available_port(port + 1)
        spec["ps"].append("localhost:{}".format(port))
    for i in range(n_workers):
        if "worker" not in spec:
            spec["worker"] = []
        port = get_available_port(port + 1)
        spec["worker"].append("localhost:{}".format(port))
    return tf.train.ClusterSpec(spec)


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def get_available_port(start_port=6006):
    port = start_port
    while is_port_in_use(port):
        port += 1
    return port


def custom_loss(x):
    return (1 - 1 / tf.cosh(2 * x)) / (1 - 1 / np.cosh(2))


class Conf:
    def __init__(self, args):
        self.mlr, self.clr = args.model_learning_rate, args.critic_learning_rate
        self.epsilon = args.epsilon
        self.epsilon_decay = args.epsilon_decay
        self.discount_factor = args.discount_factor
        self.sequence_length = args.sequence_length
        self.update_per_episode = args.update_per_episode
        self.buffer_size = 20


class Worker:
    def __init__(self, cluster, task_index, pipe, logdir, simulator_port,
                 model_lr, critic_lr, discount_factor,
                 epsilon_init, epsilon_decay,
                 sequence_length, model_buffer_size, update_per_episode,
                 worker0_display=False):
        self.task_index = task_index
        self.cluster = cluster
        self._n_workers = self.cluster.num_tasks("worker") - 1
        self.server = tf.train.Server(cluster, "worker", task_index)
        self.name = "/job:worker/task:{}".format(task_index)
        self.device = tf.train.replica_device_setter(worker_device=self.name, cluster=cluster)
        self.trajectory_count = tf.Variable(0).assign_add(1)
        self.discount_factor = discount_factor
        self.epsilon_init = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.sequence_length = sequence_length
        self.update_per_episode = update_per_episode
        self.model_lr = model_lr
        self.critic_lr = critic_lr
        self.buffer = Buffer(size=model_buffer_size)
        self.pipe = pipe
        self.logdir = logdir
        self.n_actions_per_joint = 9
        self.ratios = list(range(1, 9))
        self.define_networks()
        self.define_actions_sets()
        self.end_episode_data = []
        self._flush_id = 0
        self._update_number = 0
        graph = tf.get_default_graph() if task_index == 0 else None
        self.summary_writer = tf.summary.FileWriter(self.logdir + "/worker{}".format(task_index), graph=graph)
        self.saver = tf.train.Saver()
        self.sess = tf.Session(target=self.server.target)
        if task_index == 0 and len(self.sess.run(tf.report_uninitialized_variables())) > 0:  # todo: can be done in Experiment
            self.sess.run(tf.global_variables_initializer())
            print("{}  variables initialized".format(self.name))
        # lock = filelock.FileLock("/home/wilmot/Documents/code/asynchronous_aec/experiments/lock_universe")
        # lock.acquire()
        self.universe = vb.Universe(task_index == 0 and worker0_display, port=simulator_port)
        self.robot = self.universe.robot
        resA, resolution, imgCamLeft = vrep.simxGetVisionSensorImageFast(self.robot.sim.clientID,
                                                                         self.robot.sim.handles["camLeft"],
                                                                         not self.robot.rgbEyes,
                                                                         vrep.simx_opmode_blocking)
        while imgCamLeft is None:
            print("{} cameras return None ... waiting 1 sec".format(self.name))
            time.sleep(1)
            resA, resolution, imgCamLeft = vrep.simxGetVisionSensorImageFast(self.robot.sim.clientID,
                                                                            self.robot.sim.handles["camLeft"],
                                                                            not self.robot.rgbEyes,
                                                                            vrep.simx_opmode_blocking)
        # lock.release()
        self.textures = get_textures("/home/aecgroup/aecdata/Textures/mcgillManMade_600x600_bmp_selection/")
        self.screen = vb.ConstantSpeedScreen(self.universe.sim, (0.5, 5), 0.0, 0, self.textures)
        self.screen.set_positions(-2, 0, 0)
        self.tilt_delta = 0.0
        self.pan_delta = 0.0

    def define_scale_inp(self, ratio):
        crop_side_length = 16
        height_slice = slice(
            (240 - crop_side_length * ratio) // 2,
            (240 + crop_side_length * ratio) // 2)
        width_slice = slice(
            (320 - crop_side_length * ratio) // 2,
            (320 + crop_side_length * ratio) // 2)
        scale_uint8 = self.cams[:, height_slice, width_slice, :]
        scale_uint8 = tf.image.resize_bilinear(scale_uint8, [crop_side_length, crop_side_length])
        scale = tf.placeholder_with_default(
            tf.cast(scale_uint8, tf.float32) / 127.5 - 1,
            shape=scale_uint8.get_shape())
        self.scales_inp[ratio] = scale

    def define_autoencoder_scale(self, ratio, filter_size=4, stride=2):
        inp = self.scales_inp[ratio]
        batch_size = tf.shape(inp)[0]
        conv1 = tl.conv2d(inp + 0, filter_size ** 2 * 3 * 2 // 2, filter_size, stride, "valid", activation_fn=lrelu)
        # conv2 = tl.conv2d(conv1, filter_size ** 2 * 3 * 2 // 4, 1, 1, "valid", activation_fn=lrelu)
        # conv3 = tl.conv2d(conv2, filter_size ** 2 * 3 * 2 // 8, 1, 1, "valid", activation_fn=lrelu)
        # bottleneck = tl.conv2d(conv3, filter_size ** 2 * 3 * 2 // 8, 1, 1, "valid", activation_fn=lrelu)
        bottleneck = tl.conv2d(conv1, filter_size ** 2 * 3 * 2 // 8, 1, 1, "valid", activation_fn=lrelu)
        # conv5 = tl.conv2d(bottleneck, filter_size ** 2 * 3 * 2 // 4, 1, 1, "valid", activation_fn=lrelu)
        # conv6 = tl.conv2d(conv5, filter_size ** 2 * 3 * 2 // 2, 1, 1, "valid", activation_fn=lrelu)
        # reconstruction = tl.conv2d(conv6, filter_size ** 2 * 3 * 2, 1, 1, "valid", activation_fn=None)
        reconstruction = tl.conv2d(bottleneck, filter_size ** 2 * 3 * 2, 1, 1, "valid", activation_fn=None)
        target = tf.extract_image_patches(
            inp, [1, filter_size, filter_size, 1], [1, stride, stride, 1], [1, 1, 1, 1], 'VALID'
        )
        size = np.prod(bottleneck.get_shape()[1:])
        self.scale_latent_conv[ratio] = bottleneck
        self.scale_latent[ratio] = tf.reshape(bottleneck, (-1, size))
        self.scale_rec[ratio] = reconstruction
        sub = self.scale_rec[ratio] - target
        self.patch_recerrs[ratio] = tf.reduce_mean(sub ** 2, axis=3)
        self.scale_recerrs[ratio] = tf.reduce_mean(self.patch_recerrs[ratio], axis=[1, 2])
        self.patch_rewards[ratio] = (self.patch_recerrs[ratio][:-1] - self.patch_recerrs[ratio][1:]) / 0.01
        self.scale_rewards[ratio] = (self.scale_recerrs[ratio][:-1] - self.scale_recerrs[ratio][1:]) / 0.01
        self.scale_rewards__partial[ratio] = self.scale_recerrs[ratio] / 0.01
        self.scale_recerr[ratio] = tf.reduce_mean(self.scale_recerrs[ratio])
        ### Images for tensorboard:
        n_patches = (inp.get_shape()[1] - filter_size + stride) // stride
        left_right = tf.reshape(reconstruction[-1], (n_patches, n_patches, filter_size, filter_size, 6))
        left_right = tf.transpose(left_right, perm=[0, 2, 1, 3, 4])
        left_right = tf.reshape(left_right, (n_patches * filter_size, n_patches * filter_size, 6))
        left = left_right[..., :3]
        right = left_right[..., 3:]
        image_left = tf.concat([left, right], axis=0)
        left_right = tf.reshape(target[-1], (n_patches, n_patches, filter_size, filter_size, 6))
        left_right = tf.transpose(left_right, perm=[0, 2, 1, 3, 4])
        left_right = tf.reshape(left_right, (n_patches * filter_size, n_patches * filter_size, 6))
        left = left_right[..., :3]
        right = left_right[..., 3:]
        image_right = tf.concat([left, right], axis=0)
        self.scale_tensorboard_images[ratio] = tf.expand_dims(tf.concat([image_left, image_right], axis=1), axis=0)

    def define_critic_patch(self, ratio, joint_name):
        inp = tf.stop_gradient(self.scale_latent_conv[ratio])
        conv1 = tl.conv2d(inp, 20, 1, 1, "valid", activation_fn=lrelu)
        patch_values = tl.conv2d(conv1, self.n_actions_per_joint, 1, 1, "valid", activation_fn=None)
        self.patch_values[joint_name][ratio] = patch_values
        patch_rewards = self.patch_rewards[ratio]
        with tf.device(self.device):
            start = tf.reduce_max(patch_values[-1], axis=-1)[tf.newaxis]
            patch_returns = tf_returns(patch_rewards, self.discount_factor, start=start, axis=0)
        self.patch_returns[joint_name][ratio] = patch_returns
        actions = self.picked_actions[joint_name]
        # patch_values  40, 7, 7, 9
        params = tf.transpose(patch_values, perm=[0, 3, 1, 2])
        # params        40, 9, 7, 7
        indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)  # 40, 2
        patch_values_picked_actions = tf.gather_nd(params, indices)  # 40, 9, 7, 7
        losses = (patch_values_picked_actions[:-1] - patch_returns) ** 2
        mask = tf.reshape(self.action_mask[joint_name], (-1, 1, 1, self.n_actions_per_joint))
        stay_the_same_loss = mask * (patch_values[:-1] - tf.stop_gradient(patch_values[:-1])) ** 2
        size = self.n_actions_per_joint * 7 * 7
        self.patch_loss[joint_name][ratio] = (tf.reduce_sum(losses) + tf.reduce_sum(stay_the_same_loss)) / size

    def define_critic_scale(self, ratio, joint_name):
        size = np.prod(self.patch_values[joint_name][ratio].get_shape()[1:])
        inp = tf.concat([
            tf.stop_gradient(self.scale_latent[ratio]),
            tf.stop_gradient(tf.reshape(self.patch_values[joint_name][ratio], (-1, size)))
        ], axis=1)
        # inp = self.scale_latent[ratio]
        fc1 = tl.fully_connected(inp, 200, activation_fn=lrelu)
        fc2 = tl.fully_connected(fc1, 200, activation_fn=lrelu)
        # fc3 = tl.fully_connected(fc2, 200, activation_fn=lrelu)
        # fc4 = tl.fully_connected(fc3, 1, activation_fn=None)
        scale_values = tl.fully_connected(fc2, self.n_actions_per_joint, activation_fn=None)
        self.scale_values[joint_name][ratio] = scale_values
        scale_rewards = self.scale_rewards[ratio]
        with tf.device(self.device):
            start = tf.reduce_max(scale_values[-1], axis=-1)[tf.newaxis]
            scale_returns = tf_returns(scale_rewards, self.discount_factor, start=start, axis=0)
        self.scale_returns[joint_name][ratio] = scale_returns
        actions = self.picked_actions[joint_name]
        indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
        scale_values_picked_actions = tf.gather_nd(scale_values, indices)
        losses = (scale_values_picked_actions[:-1] - scale_returns) ** 2
        mask = self.action_mask[joint_name]
        stay_the_same_loss = mask * (scale_values[:-1] - tf.stop_gradient(scale_values[:-1])) ** 2
        size = self.n_actions_per_joint
        self.scale_loss[joint_name][ratio] = (tf.reduce_sum(losses) + tf.reduce_sum(stay_the_same_loss)) / size

    def define_critic_joint(self, joint_name):
        self.picked_actions[joint_name] = tf.placeholder(shape=(None,), dtype=tf.int32)
        self.action_mask[joint_name] = tf.one_hot(
            self.picked_actions[joint_name][:-1],
            self.n_actions_per_joint,
            on_value=0.0,
            off_value=1.0
        )
        for ratio in self.ratios:
            self.define_critic_patch(ratio, joint_name)
            self.define_critic_scale(ratio, joint_name)
        inp = tf.concat(
            [tf.stop_gradient(self.latent)] +
            [tf.stop_gradient(self.scale_values[joint_name][r]) for r in self.ratios], axis=-1)
        # inp = tf.stop_gradient(self.latent)
        fc1 = tl.fully_connected(inp, 200, activation_fn=lrelu)
        fc2 = tl.fully_connected(fc1, 200, activation_fn=lrelu)
        critic_values = tl.fully_connected(fc2, self.n_actions_per_joint, activation_fn=None)
        self.critic_values[joint_name] = critic_values
        with tf.device(self.device):
            start = tf.reduce_max(self.critic_values[joint_name][-1], axis=-1)[tf.newaxis]
            self.returns[joint_name] = tf_returns(self.rewards, self.discount_factor, start=start, axis=0)
        actions = self.picked_actions[joint_name]
        indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
        critic_values_picked_actions = tf.gather_nd(critic_values, indices)
        losses = (critic_values_picked_actions[:-1] - self.returns[joint_name]) ** 2
        mask = self.action_mask[joint_name]
        stay_the_same_loss = mask * (critic_values[:-1] - tf.stop_gradient(critic_values[:-1])) ** 2
        self.critic_loss[joint_name] = tf.reduce_sum(losses) + tf.reduce_sum(stay_the_same_loss)
        self.critic_loss_all_levels[joint_name] = self.critic_loss[joint_name]
        for ratio in self.ratios:
            self.critic_loss_all_levels[joint_name] += self.patch_loss[joint_name][ratio]
            self.critic_loss_all_levels[joint_name] += self.scale_loss[joint_name][ratio]
        ### ACTIONS:
        self.greedy_actions_indices[joint_name] = tf.cast(tf.argmax(self.critic_values[joint_name], axis=1), dtype=tf.int32)
        shape = tf.shape(self.greedy_actions_indices[joint_name])
        condition = tf.greater(tf.random_uniform(shape=shape), self.epsilon)
        random = tf.random_uniform(shape=shape, maxval=self.n_actions_per_joint, dtype=tf.int32)
        self.sampled_actions_indices[joint_name] = tf.where(condition, x=self.greedy_actions_indices[joint_name], y=random)

    def define_critic(self):
        # done once for every joint
        self.picked_actions = {}
        self.action_mask = {}
        self.patch_values = {}
        self.patch_returns = {}
        self.patch_loss = {}
        self.scale_values = {}
        self.scale_returns = {}
        self.scale_loss = {}
        self.critic_values = {}
        self.returns = {}
        self.critic_loss = {}
        self.critic_loss_all_levels = {}
        self.greedy_actions_indices = {}
        self.sampled_actions_indices = {}
        self.critic_loss_all_levels_all_joints = 0
        for joint_name in ["tilt", "pan", "vergence"]:
            # done once per joint
            self.patch_values[joint_name] = {}
            self.patch_returns[joint_name] = {}
            self.patch_loss[joint_name] = {}
            self.scale_values[joint_name] = {}
            self.scale_returns[joint_name] = {}
            self.scale_loss[joint_name] = {}
            self.define_critic_joint(joint_name)
            self.critic_loss_all_levels_all_joints += self.critic_loss_all_levels[joint_name]

    def define_inps(self):
        self.scales_inp = {}
        for ratio in self.ratios:
            self.define_scale_inp(ratio)

    def define_autoencoder(self):
        self.scale_latent = {}
        self.scale_latent_conv = {}
        self.scale_rec = {}
        self.patch_recerrs = {}
        self.patch_rewards = {}
        self.scale_recerrs = {}
        self.scale_rewards = {}
        self.scale_rewards__partial = {}
        self.scale_recerr = {}
        self.scale_tensorboard_images = {}
        for ratio in self.ratios:
            self.define_autoencoder_scale(ratio, filter_size=4, stride=2)
        self.latent = tf.concat([self.scale_latent[r] for r in self.ratios], axis=1)
        self.autoencoder_loss = sum([self.scale_recerr[r] for r in self.ratios])
        self.rewards = sum([self.scale_rewards[r] for r in self.ratios]) / len(self.ratios)
        self.rewards__partial = sum([self.scale_rewards__partial[r] for r in self.ratios]) / len(self.ratios)

    def define_epsilon(self):
        self.epsilon = tf.Variable(self.epsilon_init)
        self.epsilon_update = self.epsilon.assign(self.epsilon * self.epsilon_decay)

    def define_networks(self):
        self.left_cam = tf.placeholder(shape=(None, 240, 320, 3), dtype=tf.uint8)
        self.right_cam = tf.placeholder(shape=(None, 240, 320, 3), dtype=tf.uint8)
        self.cams = tf.concat([self.left_cam, self.right_cam], axis=-1)  # (None, 240, 320, 6)
        ### graph definitions:
        self.define_inps()
        self.define_autoencoder()
        self.define_epsilon()
        self.define_critic()
        ### summaries
        summary_loss = tf.summary.scalar("/autoencoders/loss", self.autoencoder_loss)
        summary_per_joint = [tf.summary.scalar("/critic/{}".format(jn), self.critic_loss[jn]) for jn in ["tilt", "pan", "vergence"]]
        summary_critic_loss = tf.summary.scalar("/critic/loss", self.critic_loss_all_levels_all_joints)
        summary_scale_critic_loss = [tf.summary.scalar("/critic/{}_ratio_{}".format(jn, r), self.scale_loss[jn][r]) for jn, r in product(["tilt", "pan", "vergence"], self.ratios)]
        summary_images = [tf.summary.image("ratio_{}".format(r), self.scale_tensorboard_images[r], max_outputs=1) for r in self.ratios]
        self.summary = tf.summary.merge(
            [summary_loss, summary_critic_loss] +
            summary_per_joint +
            summary_scale_critic_loss +
            summary_images
        )
        ### Ops
        self.train_step = tf.Variable(0, dtype=tf.int32)
        self.train_step_inc = self.train_step.assign_add(1, use_locking=True)
        self.optimizer_autoencoder = tf.train.AdamOptimizer(self.model_lr, use_locking=True)
        self.optimizer_critic = tf.train.AdamOptimizer(self.critic_lr, use_locking=True)
        self.train_op_autoencoder = self.optimizer_autoencoder.minimize(self.autoencoder_loss)
        self.train_op_critic = self.optimizer_critic.minimize(self.critic_loss_all_levels_all_joints)
        self.train_op = tf.group([self.train_op_autoencoder, self.train_op_critic])

    def wait_for_variables_initialization(self):
        while len(self.sess.run(tf.report_uninitialized_variables())) > 0:
            print("{}  waiting for variable initialization...".format(self.name))
            time.sleep(1)

    def __call__(self):
        try:
            self.pipe.send("{} going idle".format(self.name))
            cmd = self.pipe.recv()
            while not cmd == "done":
                print("{} got command {}".format(self.name, cmd))
                self.__getattribute__(cmd[0])(*cmd[1:])
                cmd = self.pipe.recv()
        except Exception as e:
            print("{} caught exception in worker".format(self.name))
            self.universe.sim.stopSimulator()
            raise e

    def save(self, path):
        save_path = self.saver.save(self.sess, path + "/network.ckpt")
        self.pipe.send("{} saved model to {}".format(self.name, save_path))

    def get_current_step(self):
        self.pipe.send(self.sess.run(self.train_step))

    def restore(self, path):
        self.saver.restore(self.sess, os.path.normpath(path + "/network.ckpt"))
        self.pipe.send("{} variables restored from {}".format(self.name, path))

    def test_chunks(self, proper_display_of_chunk_of_test_cases):
        chunk_of_test_cases = proper_display_of_chunk_of_test_cases.data
        ret = []
        fetches = {
            "total_reconstruction_error": self.autoencoder_loss,
            "critic_value": self.critic_values,
            "action_index": self.greedy_actions_indices
        }
        for test_case in chunk_of_test_cases:
            vergence_init = - np.degrees(np.arctan2(RESSOURCES.Y_EYES_DISTANCE, 2 * test_case["object_distance"])) * 2 + test_case["vergence_error"]
            self.robot.setEyePositions((0, 0, vergence_init))
            self.screen.set_texture_by_index(test_case["stimulus"])
            self.screen.set_movement(test_case["object_distance"], 0, 0, 0, test_case["speed_error"][0], test_case["speed_error"][1])
            self.screen.put_to_position()
            test_data = np.zeros(test_case["n_iterations"], dtype=dttest_data)
            for i in range(test_case["n_iterations"]):
                eye_position = self.robot.getEyePositions()
                vergence = eye_position[-1]
                vergence_error = - np.degrees(np.arctan2(RESSOURCES.Y_EYES_DISTANCE, 2 * test_case["object_distance"])) * 2 - vergence
                # print("vergence_error:  ", vergence_error, "object_distance:    ", test_case["object_distance"], self.screen.distance)
                left_image, right_image = self.robot.receiveImages()
                feed_dict = {self.left_cam: [left_image], self.right_cam: [right_image]}
                data = self.sess.run(fetches, feed_dict)
                action = [data["action_index"][jn] for jn in ["tilt", "pan", "vergence"]]
                test_data[i]["action_index"] = action
                test_data[i]["action_value"] = (self.action_set_tilt[action[0]], self.action_set_pan[action[1]], self.action_set_vergence[action[2]])
                test_data[i]["critic_value"] = data["critic_value"]
                test_data[i]["total_reconstruction_error"] = data["total_reconstruction_error"]
                test_data[i]["eye_position"] = eye_position
                test_data[i]["eye_speed"] = (self.tilt_delta, self.pan_delta)
                test_data[i]["speed_error"] = test_data[i]["eye_speed"] - test_case["speed_error"]
                test_data[i]["vergence_error"] = vergence_error
                self.apply_action(action)
            ret.append((test_case, test_data))
        self.pipe.send(ret)
        return ret

    def play_trajectory(self, greedy=False):
        fetches = (self.greedy_actions_indices if greedy else self.sampled_actions_indices), self.rewards__partial
        self.reset_env()
        mean = 0
        total_reward__partial = 0
        for iteration in range(self.sequence_length):
            left_image, right_image = self.robot.receiveImages()
            feed_dict = {self.left_cam: [left_image], self.right_cam: [right_image]}
            ret, total_reward__partial_new = self.sess.run(fetches, feed_dict)
            reward = total_reward__partial - total_reward__partial_new[0]
            total_reward__partial = total_reward__partial_new[0]
            object_distance = self.screen.distance
            vergence = self.robot.getEyePositions()[-1]
            vergence_error = - np.degrees(np.arctan2(RESSOURCES.Y_EYES_DISTANCE, 2 * object_distance)) * 2 - vergence
            mean += np.abs(vergence_error)
            print("vergence error: {:.4f}\treward: {:.4f}".format(vergence_error, reward), end="\n")
            self.apply_action([ret[jn] for jn in ["tilt", "pan", "vergence"]])
        print("mean abs vergence error: {:.4f}".format(mean / self.sequence_length))

    def get_trajectory(self):
        states = []
        actions = []
        rewards = []
        fetches = {
            "scales_inp": self.scales_inp,
            "scale_reward__partial": self.scale_rewards__partial,
            "total_reward__partial": self.rewards__partial,
            "greedy_actions_indices": self.greedy_actions_indices
        }
        fetches_store = {
            "scales_inp": self.scales_inp,
            "scale_reward__partial": self.scale_rewards__partial,
            "total_reward__partial": self.rewards__partial,
            "greedy_actions_indices": self.greedy_actions_indices,
            "sampled_actions_indices": self.sampled_actions_indices,
            "scale_values": self.scale_values,
            "critic_values": self.critic_values
        }
        self.reset_env()
        trajectory = self.sess.run(self.trajectory_count)
        scale_reward__partial = np.zeros(shape=(len(self.ratios),), dtype=np.float32)
        total_reward__partial = np.zeros(shape=(), dtype=np.float32)
        for iteration in range(self.sequence_length):
            left_image, right_image = self.robot.receiveImages()
            feed_dict = {self.left_cam: [left_image], self.right_cam: [right_image]}
            if iteration < self.sequence_length:
                ret = self.sess.run(fetches_store, feed_dict)
                ### Emulate reward computation (reward is (rec_err_i - rec_err_i+1) / 0.01)
                scale_reward__partial_new = np.array([ret["scale_reward__partial"][r][0] for r in self.ratios])
                scale_reward = scale_reward__partial - scale_reward__partial_new
                scale_reward__partial = scale_reward__partial_new
                total_reward__partial_new = ret["total_reward__partial"][0]
                total_reward = total_reward__partial - total_reward__partial_new
                total_reward__partial = total_reward__partial_new
                ###
                object_distance = self.screen.distance
                eyes_position = self.robot.getEyePositions()
                eyes_speed = (self.tilt_delta, self.pan_delta)
                self.store_data(ret, object_distance, eyes_position, eyes_speed, iteration, trajectory, scale_reward, total_reward)
            else:
                ret = self.sess.run(fetches, feed_dict)
            states.append({r: ret["scales_inp"][r][0] for r in self.ratios})
            actions.append(ret["sampled_actions_indices"])
            # actions.append(ret["greedy_actions_indices"])
            self.apply_action([ret["sampled_actions_indices"][jn] for jn in ["tilt", "pan", "vergence"]])
        self.buffer.incorporate((states, actions))
        return states, actions

    def store_data(self, ret, object_distance, eyes_position, eyes_speed, iteration, trajectory, scale_reward, total_reward):
        data = {
            "worker": np.squeeze(np.array(self.task_index)),
            "trajectory": np.squeeze(np.array(trajectory)),
            "iteration": np.squeeze(np.array(iteration)),
            "global_iteration": np.squeeze(np.array(trajectory * self.sequence_length + iteration)),
            "total_reward": np.squeeze(total_reward),
            "scale_rewards": np.squeeze(scale_reward),
            "greedy_actions_indices": np.squeeze(np.array([ret["greedy_actions_indices"][k] for k in ["tilt", "pan", "vergence"]])),
            "sampled_actions_indices": np.squeeze(np.array([ret["sampled_actions_indices"][k] for k in ["tilt", "pan", "vergence"]])),
            "scale_values_tilt": np.squeeze(np.array([ret["scale_values"]["tilt"][ratio] for ratio in self.ratios])),
            "scale_values_pan": np.squeeze(np.array([ret["scale_values"]["pan"][ratio] for ratio in self.ratios])),
            "scale_values_vergence": np.squeeze(np.array([ret["scale_values"]["vergence"][ratio] for ratio in self.ratios])),
            "critic_values_tilt": np.squeeze(np.array(ret["critic_values"]["tilt"])),
            "critic_values_pan": np.squeeze(np.array(ret["critic_values"]["pan"])),
            "critic_values_vergence": np.squeeze(np.array(ret["critic_values"]["vergence"])),
            "object_distance": np.squeeze(np.array(object_distance)),
            "eyes_position": np.squeeze(np.array(eyes_position)),
            "eyes_speed": np.squeeze(np.array(eyes_speed))
        }
        self.end_episode_data.append(data)

    def flush_data(self, path):
        length = len(self.end_episode_data)
        data = {k: np.zeros([length] + list(self.end_episode_data[0][k].shape), dtype=self.end_episode_data[0][k].dtype) for k in self.end_episode_data[0]}
        for i, d in enumerate(self.end_episode_data):
            for k, v in d.items():
                data[k][i] = v
        with open(path + "/worker_{:04d}_flush_{:04d}.pkl".format(self.task_index, self._flush_id), "wb") as f:
            pickle.dump(data, f)
        self.end_episode_data.clear()
        self._flush_id += 1
        self.pipe.send("{} flushed data on the hard drive".format(self.name))

    def apply_action(self, action):
        tilt, pan, verg = action
        tilt_pos, pan_pos, verg_pos = self.robot.getEyePositions()
        self.tilt_delta += self.action_set_tilt[tilt]
        self.pan_delta += self.action_set_pan[pan]
        tilt_new_pos = tilt_pos + self.tilt_delta
        pan_new_pos = pan_pos + self.pan_delta
        verg_new_pos = np.clip(verg_pos + self.action_set_vergence[verg], -8, 0)
        self.robot.setEyePositions((tilt_new_pos, pan_new_pos, verg_new_pos))
        self.screen.iteration_init()
        self.universe.sim.stepSimulation()

    def apply_action_with_reset(self, action):
        tilt, pan, verg = action
        if tilt > 8 or pan > 8 or verg > 8:
            print("!!!!!")
            print(action)
            print("!!!!!")
        tilt_pos, pan_pos, verg_pos = self.robot.getEyePositions()
        self.tilt_delta += self.action_set_tilt[tilt]
        self.pan_delta += self.action_set_pan[pan]
        tilt_new_pos = tilt_pos + self.tilt_delta
        pan_new_pos = pan_pos + self.pan_delta
        # verg_new_pos = np.clip(verg_pos + self.action_set_vergence[verg], -8, 0)
        verg_new_pos = verg_pos + self.action_set_vergence[verg]
        if verg_new_pos >= 0 or verg_new_pos <= -8:
            random_distance = np.random.uniform(low=0.5, high=5)
            verg_new_pos = -np.arctan(RESSOURCES.Y_EYES_DISTANCE / (2 * random_distance)) * 360 / np.pi
            perfect_vergence = -np.arctan(RESSOURCES.Y_EYES_DISTANCE / (2 * self.screen.distance)) * 360 / np.pi
            random_vergence = -np.arctan(RESSOURCES.Y_EYES_DISTANCE / (2 * random_distance)) * 360 / np.pi
            # one_pixel_in_angle = 0.28
            one_pixel_in_angle = 90 / 320
            verg_new_pos = perfect_vergence + np.round((random_vergence - perfect_vergence) / one_pixel_in_angle) * one_pixel_in_angle
            # low = -np.arctan(RESSOURCES.Y_EYES_DISTANCE / (2 * 0.5)) * 360 / np.pi
            # high = -np.arctan(RESSOURCES.Y_EYES_DISTANCE / (2 * 5)) * 360 / np.pi
            # random_angle = np.random.uniform(low=low, high=high)
            # verg_new_pos = random_angle
        self.robot.setEyePositions((tilt_new_pos, pan_new_pos, verg_new_pos))
        self.screen.iteration_init()
        self.universe.sim.stepSimulation()

    def reset_env(self):
        self.screen.episode_init()
        random_distance = np.random.uniform(low=0.5, high=5)
        # random_distance = self.screen.distance
        perfect_vergence = -np.arctan(RESSOURCES.Y_EYES_DISTANCE / (2 * self.screen.distance)) * 360 / np.pi
        random_vergence = -np.arctan(RESSOURCES.Y_EYES_DISTANCE / (2 * random_distance)) * 360 / np.pi
        # one_pixel_in_angle = 0.28
        one_pixel_in_angle = 90 / 320
        random_vergence = perfect_vergence + np.round((random_vergence - perfect_vergence) / one_pixel_in_angle) * one_pixel_in_angle
        self.robot.setEyePositions((0.0, 0.0, random_vergence))
        self.tilt_delta = 0.0
        self.pan_delta = 0.0
        self.universe.sim.stepSimulation()

    def define_actions_sets(self):
        n = self.n_actions_per_joint // 2
        # tilt
        self.action_set_tilt = np.zeros(self.n_actions_per_joint)
        # pan
        self.action_set_pan = np.zeros(self.n_actions_per_joint)
        # vergence
        one_pixel_in_angle = 90 / 320
        mini = one_pixel_in_angle
        maxi = one_pixel_in_angle * 2 ** (n - 1)
        positive = np.logspace(np.log2(mini), np.log2(maxi), n, base=2)
        negative = -positive[::-1]
        self.action_set_vergence = np.concatenate([negative, [0], positive])

    def train(self):
        self.get_trajectory()
        transitions = self.buffer.batch(self.update_per_episode)
        for states, actions in transitions:
            fetches = {
                "ops": [self.train_op, self.epsilon_update],
                "summary": self.summary,
                "step": self.train_step_inc,
                "autoencoder_loss": self.autoencoder_loss,
                "epsilon": self.epsilon
            }
            feed_dict = {self.scales_inp[r]: [s[r] for s in states] for r in self.ratios}
            for jn in ["tilt", "pan", "vergence"]:
                feed_dict[self.picked_actions[jn]] = [a[jn][0] for a in actions]
            ret = self.sess.run(fetches, feed_dict=feed_dict)
            if self._update_number % 20 == 0:
                self.summary_writer.add_summary(ret["summary"], global_step=ret["step"])
            print("{} update {}\tautoencoder loss:  {:6.3f}\t\tepsilon {:.2f}".format(
                self.name,
                ret["step"],
                ret["autoencoder_loss"],
                ret["epsilon"]))
            self._update_number += 1
        return ret["step"]

    def start_training(self, n_updates):
        step = self.sess.run(self.train_step)
        n_updates += step
        while step < n_updates - self._n_workers:
            step = self.train()
        self.pipe.send("{} going IDLE".format(self.name))

    def start_playback(self, n_trajectories, greedy=False):
        for i in range(n_trajectories):
            self.play_trajectory(greedy=greedy)
            print("{} trajectory {}/{}".format(self.name, i, n_trajectories))
        self.pipe.send("{} going IDLE".format(self.name))

    def run_video(self, path, n_sequences, training=False):
        fetches = self.greedy_actions_indices if not training else self.sampled_actions_indices
        print("{} will store the video under {}".format(self.name, path))
        with get_writer(path, fps=25, format="mp4") as writer:
            for sequence_number in range(n_sequences):
                print("{} trajectory {}/{}".format(self.name, sequence_number + 1, n_sequences))
                self.reset_env()
                for iteration in range(self.sequence_length):
                    left_image, right_image = self.robot.receiveImages()
                    object_distance = self.screen.distance
                    vergence = self.robot.getEyePositions()[-1]
                    vergence_error = - np.degrees(np.arctan2(RESSOURCES.Y_EYES_DISTANCE, 2 * object_distance)) * 2 - vergence
                    frame = make_frame(left_image, right_image, object_distance, vergence_error, sequence_number + 1, n_sequences)
                    writer.append_data(frame)
                    if iteration == 0:
                        for i in range(24):
                            writer.append_data(frame)
                    feed_dict = {self.left_cam: [left_image], self.right_cam: [right_image]}
                    ret = self.sess.run(fetches, feed_dict)
                    self.apply_action([ret[jn] for jn in ["tilt", "pan", "vergence"]])
                    print("vergence error: {:.4f}".format(vergence_error))
        self.pipe.send("{} going IDLE".format(self.name))


def get_n_ports(n, start_port=19000):
    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    port = start_port
    ports = []
    for i in range(n):
        while is_port_in_use(port):
            port += 1
        ports.append(port)
        port += 1
    return ports


class Experiment:
    def __init__(self, n_parameter_servers, n_workers, experiment_dir, worker_conf, worker0_display=False):
        lock = filelock.FileLock("/home/wilmot/Documents/code/asynchronous_aec/experiments/lock")
        lock.acquire()
        self.n_parameter_servers = n_parameter_servers
        self.n_workers = n_workers
        self.experiment_dir = experiment_dir
        self.worker_conf = worker_conf
        self.worker0_display = worker0_display
        self.mktree()
        with open(self.confdir + "/worker_conf.pkl", "wb") as f:
            pickle.dump(self.worker_conf, f)
        with open(self.confdir + "/command_line.txt", "w") as f:
            f.write("python3 " + " ".join(sys.argv) + "\n")
        self.ports = get_n_ports(n_workers, start_port=19000)
        self.cluster = get_cluster(n_parameter_servers, n_workers)
        pipes = [multiprocessing.Pipe(True) for i in range(n_workers)]
        self.here_pipes = [a for a, b in pipes]
        self.there_pipes = [b for a, b in pipes]
        ### DEFINE PROCESSES ###
        self.tensorboard_process = None
        self.chromium_process = None
        self.parameter_servers_processes = [multiprocessing.Process(
            target=self.parameter_server_func,
            args=(i,),
            daemon=True)
            for i in range(self.n_parameter_servers)]
        self.workers_processes = [multiprocessing.Process(
            target=self.worker_func,
            args=(i,),
            daemon=True)
            for i in range(self.n_workers)]
        ### start all processes ###
        all_processes = self.parameter_servers_processes + self.workers_processes
        for p in all_processes:
            p.start()
        for p in self.here_pipes:
            print(p.recv())
        time.sleep(5)
        lock.release()

    def mktree(self):
        self.logdir = self.experiment_dir + "/log"
        self.checkpointsdir = self.experiment_dir + "/checkpoints"
        self.videodir = self.experiment_dir + "/video"
        self.datadir = self.experiment_dir + "/data"
        self.testdatadir = self.experiment_dir + "/test_data"
        self.confdir = self.experiment_dir + "/conf"
        if not os.path.exists(self.experiment_dir) or os.listdir(self.experiment_dir) == ['log']:
            os.makedirs(self.experiment_dir, exist_ok=True)
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.videodir)
            os.makedirs(self.datadir)
            os.makedirs(self.testdatadir)
            os.makedirs(self.confdir)
            os.makedirs(self.checkpointsdir)

    def parameter_server_func(self, task_index):
        server = tf.train.Server(self.cluster, "ps", task_index)
        server.join()

    def worker_func(self, task_index):
        np.random.seed(task_index)
        ### CONF
        # self.mlr, self.clr, self.alr = args.model_learning_rate, args.critic_learning_rate, args.actor_learning_rate
        # self.entropy_reg = args.entropy_reg
        # self.entropy_reg_decay = args.entropy_reg_decay
        # self.discount_factor = 0
        # self.sequence_length = args.sequence_length
        # self.update_per_episode = args.update_per_episode
        # self.buffer_size = 20
        ### WORKER INIT
        # model_lr, critic_lr, actor_lr,
        # discount_factor, entropy_coef, entropy_coef_decay,
        # sequence_length,
        # model_buffer_size,
        # update_per_episode,
        # worker0_display=False
        worker = Worker(self.cluster, task_index, self.there_pipes[task_index], self.logdir, self.ports[task_index],
            self.worker_conf.mlr,
            self.worker_conf.clr,
            self.worker_conf.discount_factor,
            self.worker_conf.epsilon,
            self.worker_conf.epsilon_decay,
            self.worker_conf.sequence_length,
            self.worker_conf.buffer_size,
            self.worker_conf.update_per_episode,
            self.worker0_display)
        worker.wait_for_variables_initialization()
        worker()

    def start_tensorboard(self):
        if self.tensorboard_process is not None and self.chromium_process is not None:
            if self.tensorboard_process.is_alive() or self.chromium_process.is_alive():
                print("restarting tensorboard")
                self.close_tensorboard()
        port = get_available_port()
        self.tensorboard_process = subprocess.Popen(["tensorboard", "--logdir", self.logdir, "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        time.sleep(2)
        self.chromium_process = subprocess.Popen(["chromium-browser", "http://localhost:{}".format(port)], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    def close_tensorboard(self):
        if self.tensorboard_process is not None and self.chromium_process is not None:
            self.tensorboard_process.terminate()
            self.chromium_process.terminate()

    def close_parameter_servers(self):
        for p in self.parameter_servers_processes:
            if p.is_alive():
                p.terminate()
        for p in self.parameter_servers_processes:
            while p.is_alive():
                time.sleep(0.1)

    def asynchronously_train(self, n_updates):
        for p in self.here_pipes:
            p.send(("start_training", n_updates))
        for p in self.here_pipes:
            p.recv()

    def get_current_step(self):
        for p in self.here_pipes:
            p.send(("get_current_step", ))
        for p in self.here_pipes:
            current_step = p.recv()
        return current_step

    def asynchronously_test(self, test_conf_path, chunks_size=10):
        # generate list_of_test_cases
        with open(test_conf_path, "rb") as f:
            list_of_test_cases = pickle.load(f)["test_cases"]
        # generate list_of_chunks_of_test_cases
        list_of_chunks_of_test_cases = chunks(list_of_test_cases, chunks_size)
        for i, (pipe, chunk_of_test_cases) in enumerate(zip(cycle(self.here_pipes), list_of_chunks_of_test_cases)):
            pipe.send(("test_chunks", ProperDisplay(chunk_of_test_cases, i + 1, len(list_of_chunks_of_test_cases))))
        test_data_summary = []
        for p in repeatlist(self.here_pipes, len(list_of_chunks_of_test_cases)):
            test_data = p.recv()
            test_data_summary += test_data
        # get the current iteration...
        current_step = self.get_current_step()
        # store the data
        path = self.testdatadir + "/{}.pkl".format(current_step)
        with open(path, "wb")as f:
            pickle.dump(test_data_summary, f)

    def playback(self, n_trajectories, greedy=False):
        for p in self.here_pipes:
            p.send(("start_playback", n_trajectories, greedy))
        for p in self.here_pipes:
            p.recv()

    def flush_data(self):
        for p in self.here_pipes:
            p.send(("flush_data", self.datadir))
        for p in self.here_pipes:
            p.recv()

    def save_model(self, name):
        path = self.checkpointsdir + "/{}/".format(name)
        os.mkdir(path)
        self.here_pipes[0].send(("save", path))
        print(self.here_pipes[0].recv())

    def save_video(self, name, n_sequences, training=False, outpath=None):
        path = self.videodir if outpath is None else outpath
        path += "/{}.mp4".format(name)
        self.here_pipes[0].send(("run_video", path, n_sequences, training))
        print(self.here_pipes[0].recv())

    def restore_model(self, path):
        self.here_pipes[0].send(("restore", path))
        print(self.here_pipes[0].recv())
        # for p in self.here_pipes:
        #     p.send(("restore", path))
        #     print(p.recv())

    def close_workers(self):
        for p in self.here_pipes:
            p.send("done")

    def close(self):
        self.close_tensorboard()
        self.close_workers()
        self.close_parameter_servers()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


default_mlr = 1e-4
default_clr = 1e-4


def make_experiment_path(date=None, mlr=None, clr=None, description=None):
    date = date if date else time.strftime("%Y_%m_%d-%H.%M.%S", time.localtime())
    mlr = mlr if mlr else default_mlr
    clr = clr if clr else default_clr
    description = ("__" + description) if description else ""
    experiment_dir = "../experiments/{}_mlr{:.2e}_clr{:.2e}{}".format(
        date, mlr, clr, description)
    return experiment_dir


if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-ep', '--experiment-path',
        type=str,
        action='store',
        default="",
        help="Path to the experiment directory."
    )

    parser.add_argument(
        '-np', '--n-parameter-servers',
        type=int,
        help="Number of parameter servers.",
        default=1
    )

    parser.add_argument(
        '-nw', '--n-workers',
        type=int,
        help="Number of workers.",
        default=8
    )

    parser.add_argument(
        '-d', '--description',
        type=str,
        help="Short description of the experiment.",
        default=""
    )

    parser.add_argument(
        '-t', '--tensorboard',
        action="store_true",
        help="Start TensorBoard."
    )

    parser.add_argument(
        '-fe', '--flush-every',
        type=int,
        help="Flush every N simulated trajectory.",
        default=50000
    )

    parser.add_argument(
        '-nt', '--n-trajectories',
        type=int,
        help="Number of trajectories to be simulated.",
        default=1000000
    )

    parser.add_argument(
        '-sl', '--sequence-length',
        type=int,
        help="Length of an episode.",
        default=10
    )

    parser.add_argument(
        '-u', '--update-per-episode',
        type=int,
        help="Number of updates per gathered trajectory.",
        default=10
    )

    parser.add_argument(
        '-r', '--restore-from',
        type=str,
        default="none",
        help="Checkpoint to restore from."
    )

    parser.add_argument(
        '-mlr', '--model-learning-rate',
        type=float,
        default=default_mlr,
        help="model learning rate."
    )

    parser.add_argument(
        '-clr', '--critic-learning-rate',
        type=float,
        default=default_clr,
        help="critic learning rate."
    )

    parser.add_argument(
        '-df', '--discount-factor',
        type=float,
        default=0.1,
        help="Discount factor."
    )

    parser.add_argument(
        '-eps', '--epsilon',
        type=float,
        default=0.2,
        help="Initial value for epsilon."
    )

    parser.add_argument(
        '-epsd', '--epsilon-decay',
        type=float,
        default=1.0,
        help="Decay for epsilon."
    )

    args = parser.parse_args()

    if not args.experiment_path:
        experiment_dir = make_experiment_path(
            mlr=args.model_learning_rate,
            clr=args.critic_learning_rate,
            description=args.description)
    else:
        experiment_dir = args.experiment_path

    worker_conf = Conf(args)


    with Experiment(args.n_parameter_servers, args.n_workers, experiment_dir, worker_conf) as exp:
        if args.restore_from != "none":
            exp.restore_model(args.restore_from)
        if args.tensorboard:
            exp.start_tensorboard()
        for i in range(args.n_trajectories // args.flush_every):
            if i in [2, 5, 10, 20, 30, 40, 50, 75]:
                exp.save_model("{:08d}".format(i * args.flush_every))
            exp.asynchronously_train(args.flush_every)
            exp.flush_data()
        exp.save_model("{:08d}".format((i + 1) * args.flush_every))
