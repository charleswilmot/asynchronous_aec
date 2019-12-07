import sys
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
from utils import to_angle
from environment import Environment
import pickle
from itertools import cycle, islice, product
from imageio import get_writer
import imageio
from PIL import ImageDraw, Image #, ImageFont
import logging


dttest_data = np.dtype([
    ("action_index", (np.int32, 3)),
    ("action_value", (np.float32, 3)),
    ("critic_value_tilt", (np.float32, 9)),  # 9 <--> n_actions_per_joint (dttest_data shoud be member of the worker)
    ("critic_value_pan", (np.float32, 9)),
    ("critic_value_vergence", (np.float32, 9)),
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


def make_frame(left_image, right_image, object_distance, vergence_error, episode_number, total_episode_number, rectangles, iteration=None):
    """Makes an anaglyph from a left and right images, plus writes some infos on the frame
    """
    # left right to anaglyph
    image = np.matmul(np.concatenate([left_image * 255, right_image * 255], axis=-1), anaglyph_matrix).astype(np.uint8)
    # convert to PIL image
    image = Image.fromarray(image)
    # create a drawer form writing text on the frame
    drawer = ImageDraw.Draw(image)
    for rec in rectangles:
        drawer.rectangle(rec, outline=(50, 0, 50, 50))
    string = "Object distance (m): {: .2f}\nVergence error (deg): {: .2f}\nEpisode {: 3d}/{: 3d}".format(
        object_distance, vergence_error, episode_number, total_episode_number,
    )
    if iteration:
        string_iteration = "Iteration {}".format(
            iteration
        )
    drawer.text((20,15), string, fill=(255,255,0))
    drawer.text((20, 55), string_iteration, fill=(255, 0, 0))
    return np.array(image, dtype=np.uint8)


class ProperDisplay:
    """Just a helper tool for conveniently showing infos in the terminal during testing phases.
    This class just implements the __repr__ function, which is call when the object is printed.
    """
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
    """Reads and return all the textures (ie images) found under path.
    """
    filepaths = [path + "/{}".format(x) for x in os.listdir(path) if x.endswith(".bmp") or x.endswith(".png")]
    return np.array([np.array(Image.open(x)) for x in filepaths])


def lrelu(x):
    """Tensorflow activation function (slope 0.2 for x < 0, slope 1 for x > 0)"""
    alpha = 0.2
    return tf.nn.relu(x) * (1 - alpha) + x * alpha


def get_cluster(n_parameter_servers, n_workers):
    """Returns a cluster object that enables tensorflow to perform asynchronous updates of the variables"""
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
    """Returns true is the port is available, else false"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def get_available_port(start_port=6006):
    """Returns the first available port, starting from start_port"""
    port = start_port
    while is_port_in_use(port):
        port += 1
    return port


class Conf:
    """Meant to contain all parameters related to the model.
    todo: add the ratios parameter, that controls which ratios are used
    todo: pass this conf object directly to the constructor of the Worker object
    todo: buffer size should not be fixed (=20) but should be defined from the command line
    """
    def __init__(self, args):
        self.mlr, self.clr = args.model_learning_rate, args.critic_learning_rate
        self.epsilon = args.epsilon
        self.epsilon_decay = args.epsilon_decay
        self.discount_factor = args.discount_factor
        self.episode_length = args.episode_length
        self.update_factor = args.update_factor
        self.buffer_size = 20


class Worker:
    """A worker is a client in a subprocess waiting for instruction.
    It first defines a model according to the model's parameters, then goes to the idle mode, and waits for instructions
    todo: see the Conf object: pass a Conf instance to the Worker constructor.
    """
    def __init__(self, cluster, task_index, pipe, logdir, simulator_port,
                 model_lr, critic_lr, discount_factor,
                 epsilon_init, epsilon_decay,
                 episode_length, model_buffer_size, update_factor,
                 worker0_display=False):
        self.task_index = task_index
        self.cluster = cluster
        self._n_workers = self.cluster.num_tasks("worker") - 1
        self.server = tf.train.Server(cluster, "worker", task_index)
        self.name = "/job:worker/task:{}".format(task_index)
        self.device = tf.train.replica_device_setter(worker_device=self.name, cluster=cluster)
        self.episode_count = tf.Variable(0)
        self.episode_count_inc = self.episode_count.assign_add(1)
        self.discount_factor = discount_factor
        self.epsilon_init = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.episode_length = episode_length
        self.update_factor = update_factor
        self.model_lr = model_lr
        self.critic_lr = critic_lr
        self.buffer = Buffer(size=model_buffer_size)
        self.pipe = pipe
        self.logdir = logdir
        self.n_actions_per_joint = 9
        # todo: see Conf object: put ratios in the Conf object
        # self.ratios = list(range(1, 9))
        self.ratios = [1, 2, 3]
        self.define_networks()
        self.define_actions_sets()
        self.end_episode_data = []
        self._flush_id = 0
        self._update_number = 0
        graph = tf.get_default_graph() if task_index == 0 else None
        self.summary_writer = tf.summary.FileWriter(self.logdir + "/worker{}".format(task_index), graph=graph)
        self.saver = tf.train.Saver()
        self.sess = tf.Session(target=self.server.target)
        # todo: variable initialization can be done in the experiment constructor, would be more elegent
        if task_index == 0 and len(self.sess.run(tf.report_uninitialized_variables())) > 0:  # todo: can be done in Experiment
            self.sess.run(tf.global_variables_initializer())
            print("{} variables initialized".format(self.name))
        # starting VREP
        print("{} starting V-Rep ...".format(self.name))
        self.environment = Environment(headless=task_index != 0 or not worker0_display)
        print("{} starting V-Rep ... done.".format(self.name))

    def define_scale_inp(self, ratio):
        """Crops, downscales and converts a central region of the camera images
        Input: Image patch with size 16*ratio x 16*ratio --> Output: Image patch with 16 x 16"""
        crop_side_length = 16
        height_slice = slice(
            (240 - (crop_side_length * ratio)) // 2,
            (240 + (crop_side_length * ratio)) // 2)
        width_slice = slice(
            (320 - (crop_side_length * ratio)) // 2,
            (320 + (crop_side_length * ratio)) // 2)
        # CROP
        scale = self.cams[:, height_slice, width_slice, :]
        # DOWNSCALE
        scale = tf.image.resize_bilinear(scale, [crop_side_length, crop_side_length])
        self.scales_inp[ratio] = tf.placeholder_with_default(scale * 2 - 1, shape=scale.get_shape())

    def define_autoencoder_scale(self, ratio, filter_size=4, stride=2):
        """Defines an autoencoder that operates at one scale (one downscaling ratio)"""
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
        """Defines the critic at the level of the patches, for one scale, for one joint
        """
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
        """Defines the critic at the level of one scale, for one joint
        """
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
        """Defines the critic merging every scales, for one joint"""
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
        """Defines the critics for each joints"""
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
        """Generates the different 32x32 image patches for the given ratios."""
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
        self.left_cam = tf.placeholder(shape=(None, 240, 320, 3), dtype=tf.float32)
        self.right_cam = tf.placeholder(shape=(None, 240, 320, 3), dtype=tf.float32)
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
        # summary_scale_critic_loss = [tf.summary.scalar("/critic/{}_ratio_{}".format(jn, r), self.scale_loss[jn][r]) for jn, r in product(["tilt", "pan", "vergence"], self.ratios)]
        summary_images = [tf.summary.image("ratio_{}".format(r), self.scale_tensorboard_images[r], max_outputs=1) for r in self.ratios]
        self.summary = tf.summary.merge(
            [summary_loss, summary_critic_loss] +
            summary_per_joint +
            # summary_scale_critic_loss +
            summary_images
        )
        ### Ops
        self.update_count = tf.Variable(0, dtype=tf.int32)
        self.update_count_inc = self.update_count.assign_add(1, use_locking=True)
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
        self.pipe.send("{} going idle".format(self.name))
        cmd = self.pipe.recv()
        while not cmd == "done":
            try:
                print("{} got command {}".format(self.name, cmd))
                self.__getattribute__(cmd[0])(*cmd[1:])
                cmd = self.pipe.recv()
            except KeyboardInterrupt as e:
                print("{} caught a keyboard interrupt".format(self.name))
            except Exception as e:
                print("{} caught exception in worker".format(self.name))
                self.environment.close()
                raise e
        self.environment.close()

    def save_model(self, path):
        """Save a checkpoint on the hard-drive under path
        The Experiment class calls this methode on the worker 0 only
        """
        save_path = self.saver.save(self.sess, path + "/network.ckpt")
        self.pipe.send("{} saved model to {}".format(self.name, save_path))

    def get_current_update_count(self):
        """Returns the current model training step (number of updates applied) in the pipe
        The Experiment class calls this methode on the worker 0 only
        """
        self.pipe.send(self.sess.run(self.update_count))

    def get_current_episode_count(self):
        """Returns the current model training step (number of episodes simulated) in the pipe
        The Experiment class calls this methode on the worker 0 only
        """
        self.pipe.send(self.sess.run(self.episode_count))

    def restore_model(self, path):
        """Restores from a checkpoint
        The Experiment class calls this methode on the worker 0 only
        """
        self.saver.restore(self.sess, os.path.normpath(path + "/network.ckpt"))
        self.pipe.send("{} variables restored from {}".format(self.name, path))

    def test_chunks(self, proper_display_of_chunk_of_test_cases):
        """Performs tests in the simulator, using the greedy policy.
        proper_display_of_chunk_of_test_cases is a wrapper around a chunk_of_test_cases (for pretty printing)
        chunk_of_test_cases contains all informations about the tests that must be performed in the simulator
        Each worker gets a few test cases to process, enabling parallel computations
        The resulting measures are sent back to the Experiment class (the server)
        """
        chunk_of_test_cases = proper_display_of_chunk_of_test_cases.data
        ret = []
        fetches = {
            "total_reconstruction_error": self.autoencoder_loss,
            "critic_value": self.critic_values,
            "action_index": self.greedy_actions_indices
        }
        for test_case in chunk_of_test_cases:
            vergence_init = to_angle(test_case["object_distance"]) - test_case["vergence_error"]
            self.environment.robot.set_position([0, 0, vergence_init], joint_limit_type="none")
            self.environment.screen.set_texture(test_case["stimulus"])
            self.environment.screen.set_trajectory(
                test_case["object_distance"],
                test_case["speed_error"][0],
                test_case["speed_error"][1])
            test_data = np.zeros(test_case["n_iterations"], dtype=dttest_data)
            for i in range(test_case["n_iterations"]):
                left_image, right_image = self.environment.robot.get_vision()
                data = self.sess.run(fetches, feed_dict={self.left_cam: [left_image], self.right_cam: [right_image]})
                action_value = self.actions_indices_to_values(data["action_index"])
                test_data[i]["action_index"] = [data["action_index"][jn] for jn in ["tilt", "pan", "vergence"]]
                test_data[i]["action_value"] = action_value
                test_data[i]["critic_value_tilt"] = data["critic_value"]["tilt"]
                test_data[i]["critic_value_pan"] = data["critic_value"]["pan"]
                test_data[i]["critic_value_vergence"] = data["critic_value"]["vergence"]
                test_data[i]["total_reconstruction_error"] = data["total_reconstruction_error"]
                test_data[i]["eye_position"] = self.environment.robot.position
                test_data[i]["eye_speed"] = self.environment.robot.speed
                test_data[i]["speed_error"] = test_data[i]["eye_speed"] - test_case["speed_error"]
                test_data[i]["vergence_error"] = self.environment.robot.get_vergence_error(test_case["object_distance"])
                self.environment.robot.set_action(action_value, joint_limit_type="none")
                self.environment.step()
            ret.append((test_case, test_data))
        self.pipe.send(ret)
        return ret

    def playback_one_episode(self, greedy=False):
        """Performs one episode of fake training (for visualizing a trained agent in the simulator, see playback.py)
        The greedy boolean specifies wether the greedy or sampled policy should be used
        """
        fetches = (self.greedy_actions_indices if greedy else self.sampled_actions_indices), self.rewards__partial
        self.environment.episode_reset()
        mean = 0
        total_reward__partial = 0
        for iteration in range(self.episode_length):
            left_image, right_image = self.environment.robot.get_vision()
            feed_dict = {self.left_cam: [left_image], self.right_cam: [right_image]}
            ret, total_reward__partial_new = self.sess.run(fetches, feed_dict)
            reward = total_reward__partial - total_reward__partial_new[0]
            total_reward__partial = total_reward__partial_new[0]
            object_distance = self.environment.screen.distance
            vergence_error = self.environment.robot.get_vergence_error(object_distance)
            mean += np.abs(vergence_error)
            print("vergence error: {:.4f}\treward: {:.4f}     {}".format(vergence_error, reward, self.actions_indices_to_values(ret)[-1]), end="\n")
            self.environment.robot.set_action(self.actions_indices_to_values(ret))
            self.environment.step()
        print("mean abs vergence error: {:.4f}".format(mean / self.episode_length))

    def get_episode(self):
        """Get the training data that the RL algorithm needs,
        and stores training infos in a buffer that must be flushed regularly
        See the help guide about data formats
        """
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
        self.environment.episode_reset()
        episode_number, epsilon = self.sess.run([self.episode_count_inc, self.epsilon])
        print("{} simulating episode {}\tepsilon {:.2f}".format(self.name, episode_number, epsilon))
        scale_reward__partial = np.zeros(shape=(len(self.ratios),), dtype=np.float32)
        total_reward__partial = np.zeros(shape=(), dtype=np.float32)
        for iteration in range(self.episode_length):
            left_image, right_image = self.environment.robot.get_vision()
            feed_dict = {self.left_cam: [left_image], self.right_cam: [right_image]}
            ret = self.sess.run(fetches_store, feed_dict)
            ### Emulate reward computation (reward is (rec_err_i - rec_err_i+1) / 0.01)
            scale_reward__partial_new = np.array([ret["scale_reward__partial"][r][0] for r in self.ratios])
            scale_reward = scale_reward__partial - scale_reward__partial_new
            scale_reward__partial = scale_reward__partial_new
            total_reward__partial_new = ret["total_reward__partial"][0]
            total_reward = total_reward__partial - total_reward__partial_new
            total_reward__partial = total_reward__partial_new
            ###
            object_distance = self.environment.screen.distance
            eyes_position = self.environment.robot.position
            eyes_speed = self.environment.robot.speed
            if episode_number % 10 == 0:  # spare some disk space (90%)
                self.store_data(ret, object_distance, eyes_position, eyes_speed, iteration, episode_number, scale_reward, total_reward)
            states.append({r: ret["scales_inp"][r][0] for r in self.ratios})
            actions.append(ret["sampled_actions_indices"])
            # actions.append(ret["greedy_actions_indices"])
            self.environment.robot.set_action(self.actions_indices_to_values(ret["sampled_actions_indices"]))
            self.environment.step()
        self.buffer.incorporate((states, actions))
        return episode_number

    def store_data(self, ret, object_distance, eyes_position, eyes_speed, iteration, episode_number, scale_reward, total_reward):
        """Constructs a dictionary from data and store it in a buffer
        See the help guide about data formats
        """
        data = {
            "worker": np.squeeze(np.array(self.task_index)),
            "episode_number": np.squeeze(np.array(episode_number)),
            "iteration": np.squeeze(np.array(iteration)),
            "global_iteration": np.squeeze(np.array(episode_number * self.episode_length + iteration)),
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
        """Reformats the training data buffer and dumps it onto the hard drive
        This function must be called regularly.
        The frequency at which it is called can be specified in the command line with the option --flush-every
        See the help guide about data formats
        """
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

    def define_actions_sets(self):
        """Defines the pan/tilt/vergence action sets
        At the moment, pan and tilt are comprised only of zeros"""
        n = self.n_actions_per_joint // 2
        # tilt
        self.action_set_tilt = np.zeros(self.n_actions_per_joint)
        # pan
        half_pixel_in_angle = 90 / 320 / 2
        mini = half_pixel_in_angle
        maxi = half_pixel_in_angle * 2 ** (n - 1)
        positive = np.logspace(np.log2(mini), np.log2(maxi), n, base=2)
        negative = -positive[::-1]
        self.action_set_pan = np.concatenate([negative, [0], positive])
        # vergence
        half_pixel_in_angle = 90 / 320 / 2
        mini = half_pixel_in_angle
        maxi = half_pixel_in_angle * 2 ** (n - 1)
        positive = np.logspace(np.log2(mini), np.log2(maxi), n, base=2)
        negative = -positive[::-1]
        self.action_set_vergence = np.concatenate([negative, [0], positive])

    def actions_indices_to_values(self, indices_dict):
        return [self.action_set_tilt[indices_dict["tilt"]],
                self.action_set_pan[indices_dict["pan"]],
                self.action_set_vergence[indices_dict["vergence"]]]

    def train_one_episode(self, states, actions):
        """Updates the networks weights according to the transitions states and actions"""
        fetches = {
            "ops": [self.train_op, self.epsilon_update, self.update_count_inc],
            "summary": self.summary,
            "episode_count": self.episode_count
        }
        feed_dict = {self.scales_inp[r]: [s[r] for s in states] for r in self.ratios}
        for jn in ["tilt", "pan", "vergence"]:
            feed_dict[self.picked_actions[jn]] = [a[jn][0] for a in actions]
        ret = self.sess.run(fetches, feed_dict=feed_dict)
        if self._update_number % 200 == 0:
            self.summary_writer.add_summary(ret["summary"], global_step=ret["episode_count"])
        self._update_number += 1
        return ret["episode_count"]

    def train(self, n_episodes):
        """Train until n_episodes have been simulated
        See the update_factor parameter"""
        before_n_episodes = self.sess.run(self.episode_count)
        current_n_episode = before_n_episodes
        after_n_episode = before_n_episodes + n_episodes
        print("{} debug -- train sees before_n_episodes = {} after_n_episode = {}".format(self.name, before_n_episodes, after_n_episode))
        while current_n_episode < after_n_episode:
            current_n_episode = self.get_episode()
            transitions = self.buffer.batch(self.update_factor)
            for states, actions in transitions:
                current_n_episode = self.train_one_episode(states, actions)
                if current_n_episode >= after_n_episode:
                    break
        self.pipe.send("{} going IDLE".format(self.name))

    def playback(self, n_episodes, greedy=False):
        """Calls the playback_one_episode methode n_episodes times
        """
        for i in range(n_episodes):
            self.playback_one_episode(greedy=greedy)
            print("{} episode {}/{}".format(self.name, i, n_episodes))
        self.pipe.send("{} going IDLE".format(self.name))

    def make_video(self, path, n_episodes, training=False):
        """Generates image collage, to be stored under path, consisting of n_episodes"""
        fetches = self.greedy_actions_indices if not training else self.sampled_actions_indices
        rectangles = [(160 - 16 * r, 120 - 16 * r, 160 + 16 * r, 120 + 16 * r) for r in self.ratios]
        print("{} will store the images under {}".format(self.name, path))
        with get_writer(path, fps=25, format="mp4") as writer:
            for episode_number in range(n_episodes):
                print("{} episode {}/{}".format(self.name, episode_number + 1, n_episodes))
                self.environment.episode_reset()
                for iteration in range(self.episode_length):
                    left_image, right_image = self.environment.robot.get_vision()
                    object_distance = self.environment.screen.distance
                    vergence_error = self.environment.robot.get_vergence_error(object_distance)
                    frame = make_frame(left_image, right_image, object_distance, vergence_error, episode_number + 1, n_episodes, rectangles)
                    writer.append_data(frame)
                    if iteration == 0 or iteration == (self.episode_length-1):
                        for i in range(24):
                            writer.append_data(frame)
                    feed_dict = {self.left_cam: [left_image], self.right_cam: [right_image]}
                    ret = self.sess.run(fetches, feed_dict)
                    self.environment.robot.set_action(self.actions_indices_to_values(ret))
                    self.environment.step()
                    print("vergence error: {:.4f}".format(vergence_error))
        self.pipe.send("{} going IDLE".format(self.name))

    def make_images(self, path, n_episodes, training=False):
        """Generates a video, to be stored under path, consisting of n_episodes"""
        fetches = self.greedy_actions_indices if not training else self.sampled_actions_indices
        rectangles = [(160 - 16 * r, 120 - 16 * r, 160 + 16 * r, 120 + 16 * r) for r in self.ratios]
        print("{} will store the video under {}".format(self.name, path))
        for episode_number in range(n_episodes):
            print("{}: {} episode {}/{}".format("make_images", self.name, episode_number + 1, n_episodes))
            self.environment.episode_reset()
            for iteration in range(self.episode_length):
                left_image, right_image = self.environment.robot.get_vision()
                if iteration == 0 or iteration == (self.episode_length-1):
                    object_distance = self.environment.screen.distance
                    vergence_error = self.environment.robot.get_vergence_error(object_distance)
                    frame = make_frame(left_image, right_image, object_distance, vergence_error, episode_number + 1,
                                       n_episodes, rectangles, str(iteration))
                    image = Image.fromarray(frame)
                    filepath = path + "/{}_{}.jpg".format(episode_number, iteration)
                    image.save(filepath)
                    if iteration == 0:
                        zero_frame = frame
                    else:
                        stacked_image = np.hstack((zero_frame,frame))
                        image = Image.fromarray(stacked_image)
                        filepath = path + "/{}_stacked.jpg".format(episode_number)
                        image.save(filepath)
                feed_dict = {self.left_cam: [left_image], self.right_cam: [right_image]}
                ret = self.sess.run(fetches, feed_dict)
                self.environment.robot.set_action(self.actions_indices_to_values(ret))
                self.environment.step()
                print("vergence error: {:.4f}".format(vergence_error))
        self.pipe.send("{} going IDLE".format(self.name))


def get_n_ports(n, start_port=19000):
    """Returns a list of n usable ports, one for each worker"""
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
    logging.info(ports)
    return ports


class Experiment:
    """An experiment object allows to control clients / workers that train a model
    It starts one process per worker, and one process per parameter server
    It also constructs the filesystem tree for storing all results / data"""

    def __init__(self, n_parameter_servers, n_workers, experiment_dir, worker_conf, worker0_display=False):
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
        print("EXPERIMENT: all processes started. Waiting for answer...")
        for p in self.here_pipes:
            print(p.recv())

    def mktree(self):
        self.logdir = self.experiment_dir + "/log"
        self.checkpointsdir = self.experiment_dir + "/checkpoints"
        self.videodir = self.experiment_dir + "/video"
        self.imagedir = self.experiment_dir + "/images"
        self.datadir = self.experiment_dir + "/data"
        self.testdatadir = self.experiment_dir + "/test_data"
        self.confdir = self.experiment_dir + "/conf"
        if not os.path.exists(self.experiment_dir) or os.listdir(self.experiment_dir) == ['log']:
            os.makedirs(self.experiment_dir, exist_ok=True)
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.videodir)
            os.makedirs(self.imagedir)
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
        # self.episode_length = args.episode_length
        # self.update_factor = args.update_factor
        # self.buffer_size = 20
        ### WORKER INIT
        # model_lr, critic_lr, actor_lr,
        # discount_factor, entropy_coef, entropy_coef_decay,
        # episode_length,
        # model_buffer_size,
        # update_factor,
        # worker0_display=False
        worker = Worker(
            self.cluster,
            task_index,
            self.there_pipes[task_index],
            self.logdir,
            self.ports[task_index],
            self.worker_conf.mlr,
            self.worker_conf.clr,
            self.worker_conf.discount_factor,
            self.worker_conf.epsilon,
            self.worker_conf.epsilon_decay,
            self.worker_conf.episode_length,
            self.worker_conf.buffer_size,
            self.worker_conf.update_factor,
            self.worker0_display
            )
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

    def train(self, n_updates):
        """Sending string "train" to remote workers via pipes to start training"""
        for p in self.here_pipes:
            p.send(("train", n_updates))
        for p in self.here_pipes:
            p.recv()

    def get_current_update_count(self):
        self.here_pipes[0].send(("get_current_update_count", ))
        return self.here_pipes[0].recv()

    def get_current_episode_count(self):
        self.here_pipes[0].send(("get_current_episode_count", ))
        return self.here_pipes[0].recv()

    def test(self, test_conf_path, chunks_size=10, outpath=None):
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
        current_episode_count = self.get_current_episode_count()
        # store the data
        path = self.testdatadir if outpath is None else outpath
        test_conf_basename = os.path.basename(test_conf_path)
        test_conf_name = os.path.splitext(test_conf_basename)[0]
        path = path + "/{}_{}.pkl".format(current_episode_count, test_conf_name)
        with open(path, "wb")as f:
            pickle.dump(test_data_summary, f)

    def playback(self, n_episodes, greedy=False):
        for p in self.here_pipes:
            p.send(("playback", n_episodes, greedy))
        for p in self.here_pipes:
            p.recv()

    def flush_data(self):
        for p in self.here_pipes:
            p.send(("flush_data", self.datadir))
        for p in self.here_pipes:
            p.recv()

    def save_model(self, name=None):
        name = "{:08d}".format(self.get_current_episode_count()) if name is None else name
        path = self.checkpointsdir + "/{}/".format(name)
        os.mkdir(path)
        self.here_pipes[0].send(("save_model", path))
        print(self.here_pipes[0].recv())

    def make_video(self, name, n_episodes, training=False, outpath=None):
        path = self.videodir if outpath is None else outpath
        path += "/{}.mp4".format(name)
        self.here_pipes[0].send(("make_video", path, n_episodes, training))
        print(self.here_pipes[0].recv())

    def make_images(self, name, n_episodes, training=False, outpath=None):
        path = self.imagedir if outpath is None else outpath
        #path += "/{}.mp4".format(name)
        self.here_pipes[0].send(("make_images", path, n_episodes, training))
        print(self.here_pipes[0].recv())

    def restore_model(self, path):
        self.here_pipes[0].send(("restore_model", path))
        print(self.here_pipes[0].recv())

    def close_workers(self):
        for p in self.here_pipes:
            p.send("done")

    def close(self):
        print("EXPERIMENT: closing ...")
        self.close_tensorboard()
        self.close_workers()
        self.close_parameter_servers()
        print("EXPERIMENT: closing ... done.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting Experiment context: ", exc_type, exc_value)
        if exc_type is KeyboardInterrupt:
            print("EXPERIMENT: caught a KeyboardInterrupt. Emptying current command queues and exiting...")
        self.close()

default_mlr = 1e-4
default_clr = 1e-4


def make_experiment_path(date=None, mlr=None, clr=None, description=None):
    date = date if date else time.strftime("%Y_%m_%d-%H.%M.%S", time.localtime())
    mlr = mlr if mlr else default_mlr
    clr = clr if clr else default_clr
    #description = ("__" + description) if description else ""
    description = description if description else ""
    experiment_dir = "../experiments/{}_mlr{:.2e}_clr{:.2e}_{}".format(
        description,
        mlr,
        clr,
        date)
    return experiment_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-ep', '--experiment-path',
        type=str,
        action='store',
        default="",
        help="Path to the experiment directory. Results are stored here."
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
        help="Flush every N simulated episode.",
        default=5000
    )

    parser.add_argument(
        '-ne', '--n-episodes',
        type=int,
        help="Number of episodes to be simulated.",
        default=100000
    )

    parser.add_argument(
        '-el', '--episode-length',
        type=int,
        help="Length of an episode.",
        default=10
    )

    parser.add_argument(
        '-u', '--update-factor',
        type=int,
        help="Number of updates per simulated episode.",
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

    test_at = np.array([5000, 10000, 20000, 30000, 50000, 75000, 100000, 150000]) // args.flush_every

    with Experiment(args.n_parameter_servers, args.n_workers, experiment_dir, worker_conf) as exp:
        if args.restore_from != "none":
            exp.restore_model(args.restore_from)
        if args.tensorboard:
            exp.start_tensorboard()
        for i in range(args.n_episodes // args.flush_every):
            if i in test_at:
                exp.save_model()
                exp.test("../test_conf/vergence_trajectory_4_distances.pkl")
            exp.train(args.flush_every)
            exp.flush_data()
        exp.save_model()
        exp.test("../test_conf/vergence_trajectory_4_distances.pkl")
        exp.make_images("final", 10)
        exp.make_video("final", 100)
