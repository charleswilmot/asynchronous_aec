import tensorflow as tf
from algorithm.replay_buffer import Buffer
from algorithm.environment import Environment
import tensorflow.contrib.layers as tl
import numpy as np
from algorithm.returns import to_return
import time
import os
from helper.utils import to_angle, actions_set_values_tilt, actions_set_values_pan, actions_set_values_vergence
from imageio import get_writer
from PIL import ImageDraw, Image #, ImageFont
import queue


anaglyph_matrix = np.array([
    [0.299, 0    , 0    ],
    [0.587, 0    , 0    ],
    [0.114, 0    , 0    ],
    [0    , 0.299, 0.299],
    [0    , 0.587, 0.587],
    [0    , 0.114, 0.114],
    ])


# TODO: Fix this 2 frames
def make_frame(left_image, right_image, object_distance, vergence_error, episode_number, total_episode_number, rectangles):
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
        object_distance, vergence_error, episode_number, total_episode_number
    )
    drawer.text((20,15), string, fill=(255,255,0))
    return np.array(image, dtype=np.uint8)


def lrelu(x):
    """Tensorflow activation function (slope 0.2 for x < 0, slope 1 for x > 0)"""
    alpha = 0.2
    return tf.nn.relu(x) * (1 - alpha) + x * alpha


class Worker:
    """A worker is a client in a subprocess waiting for instruction.
    It first defines a model according to the model's parameters, then goes to the idle mode, and waits for instructions
    todo: see the Conf object: pass a Conf instance to the Worker constructor.
    """
    def __init__(self, cluster, task_index, pipe_and_queues, logdir, simulator_port, worker_conf, worker0_display=False):
        ### configuration of distributed TF
        self.task_index = task_index
        self.cluster = cluster
        self._n_workers = self.cluster.num_tasks("worker") - 1
        self.server = tf.train.Server(cluster, "worker", task_index)
        task_index_str = "{}".format(task_index)
        task_index_str = task_index_str + " " * (3 - len(task_index_str))
        self.name = "/job:worker/task:{}".format(task_index_str)
        self.device = tf.train.replica_device_setter(worker_device=self.name, cluster=cluster)
        ### copy worker_conf.
        self.discount_factor = worker_conf.discount_factor
        self.epsilon_init = worker_conf.epsilon
        self.epsilon_decay = worker_conf.epsilon_decay
        self.episode_length = worker_conf.episode_length
        self.update_factor = worker_conf.update_factor
        self.model_lr = worker_conf.mlr
        self.critic_lr = worker_conf.clr
        self.reward_scaling_factor = worker_conf.reward_scaling_factor
        self.buffer_size = worker_conf.buffer_size
        self.batch_size = worker_conf.batch_size
        self.batch_norm_decay = worker_conf.batch_norm_decay
        ### communication to server
        self.pipe = pipe_and_queues["pipe"]
        self.summary_queue = pipe_and_queues["summary_queue"]
        self.training_data_queue = pipe_and_queues["training_data_queue"]
        self.testing_data_queue = pipe_and_queues["testing_data_queue"]
        self.test_cases_queue = pipe_and_queues["test_cases_queue"]
        self.logdir = logdir
        self.define_actions_sets()
        self.ratios = worker_conf.ratios
        self.n_scales = len(self.ratios)
        self.turn_2_frames_vergence_on = worker_conf.turn_2_frames_vergence_on
        self.n_encoders = 2 if self.turn_2_frames_vergence_on else 1
        self.n_joints = 3
        self.episode_count = tf.Variable(0)
        self.episode_count_inc = self.episode_count.assign_add(1)
        self.is_training = tf.placeholder_with_default(False, shape=())
        self.define_networks()
        ### some profiling stuff
        self._tsimulation = 0.0
        self._ttrain = 0.0
        self._trun = 0.0
        self._n_time_measurements = 0

        self.update_fetches = {
            "ops": [self.train_op, self.epsilon_update],
            "episode_count": self.update_count_inc
        }
        self.training_behaviour_fetches = {
            "scales_inp_4_frames": self.scales_inp_4_frames,
            "patch_recerrs_4_frames": self.patch_recerrs_4_frames,
            "scale_recerrs_4_frames": self.scale_recerrs_4_frames,
            "total_recerrs_4_frames": self.total_recerrs_4_frames,
            "sampled_actions_indices": self.sampled_actions_indices,
            "greedy_actions_indices": self.greedy_actions_indices,
        }
        if self.turn_2_frames_vergence_on:
            complement_2_frames = {
                "patch_recerrs_2_frames": self.patch_recerrs_2_frames,
                "scale_recerrs_2_frames": self.scale_recerrs_2_frames,
                "total_recerrs_2_frames": self.total_recerrs_2_frames
            }
            self.training_behaviour_fetches.update(complement_2_frames)
        n_patches = self.scale_rec_4_frames[self.ratios[0]].get_shape()[1]
        self.behaviour_data_type = np.dtype([
            ("scales_inp", np.float32, (self.n_scales, self.crop_side_length, self.crop_side_length, 12)),
            ("sampled_actions_indices", np.int32, (self.n_joints)),
            ("total_target_return", np.float32, (self.n_joints,))
        ])
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
        testing_data_type_description = [
            ("action_index", (np.int32, 3)),
            ("action_value", (np.float32, 3)),
            ("critic_value_tilt", (np.float32, 9)),
            ("critic_value_pan", (np.float32, 9)),
            ("critic_value_vergence", (np.float32, 9)),
            ("total_recerrs_4_frames", (np.float32)),
            ("scale_recerrs_4_frames", (np.float32, self.n_scales)),
            # ("total_reconstruction_error", np.float32),
            ("object_distance", np.float32),
            ("eye_position", (np.float32, 3)),
            ("eye_speed", (np.float32, 2)),
            ("speed_error", (np.float32, 2)),
            ("vergence_error", np.float32)
        ]
        if self.turn_2_frames_vergence_on:
            testing_data_type_description += [
                ("total_recerrs_2_frames", (np.float32)),
                ("scale_recerrs_2_frames", (np.float32, self.n_scales))
            ]
        self.testing_data_type = np.dtype(testing_data_type_description)
        self.levels_data_type = np.dtype([
            ("patch", np.float32, (self.n_encoders, self.n_scales, n_patches, n_patches)),
            ("scale", np.float32, (self.n_encoders, self.n_scales)),
            ("total", np.float32, (self.n_encoders,))
        ])
        self.buffer = Buffer(size=self.buffer_size, dtype=self.behaviour_data_type, batch_size=self.batch_size)

        # + 1 for the additional / sacrificial iteration
        self._behaviour_data = np.zeros(shape=self.episode_length + 1, dtype=self.behaviour_data_type)
        self._training_data = np.zeros(shape=self.episode_length + 1, dtype=self.training_data_type)  # passed to the logger
        self._recerrs_data = np.zeros(shape=(self.episode_length + 1, self.n_encoders), dtype=np.float32)

        # Manage Data Logging
        self.saver = tf.train.Saver()  # tf.trainable_variables() + list_of_extra_variables_for_batchnorm
        self.model_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model"))
        self.sess = tf.Session(target=self.server.target)
        # todo: variable initialization can be done in the experiment constructor, would be more elegent
        if task_index == 0 and len(self.sess.run(tf.report_uninitialized_variables())) > 0:  # todo: can be done in Experiment
            self.sess.run(tf.global_variables_initializer())
            print("{} variables initialized".format(self.name))
        # starting VREP
        print("{} starting V-Rep ...".format(self.name))
        self.environment = Environment(headless=task_index != 0 or not worker0_display)
        print("{} starting V-Rep ... done.".format(self.name))

    def add_summary(self, summary, global_step):
        try:
            self.summary_queue.put((summary, global_step), block=False)
        except queue.Full:
            print("{} could not register it's summary. (Queue is full)")

    def register_training_data(self):
        try:
            # don't register the last "sacrificial" iteration
            self.training_data_queue.put(np.copy(self._training_data[:-1]), block=False)
        except queue.Full:
            print("{} could not register training data. (Queue is full)")

    def define_critic_joint(self, joint_name):
        """Defines the critic merging every scales, for one joint"""
        self.picked_actions[joint_name] = tf.placeholder(shape=(None,), dtype=tf.int32, name="picked_action_{}".format(joint_name))
        imputs = []
        _2_frames = self.turn_2_frames_vergence_on and joint_name == "vergence"
        if _2_frames:
            inputs = [tf.stop_gradient(self.scale_latent_conv_2_frames[ratio]) for ratio in self.ratios]
        else:
            inputs = [tf.stop_gradient(self.scale_latent_conv_4_frames[ratio]) for ratio in self.ratios]
        flats = []
        for inp in inputs:
            conv1 = tl.conv2d(inp, 32 if _2_frames else 64, 2, 1, "valid", activation_fn=lrelu)
            pool1 = tl.max_pool2d(conv1, 2, 2, "valid")
            size = np.prod(pool1.get_shape()[1:])
            flats.append(tf.reshape(pool1, (-1, size)))
        flat = tf.concat(flats, axis=-1)
        fc1 = tl.fully_connected(flat, 200, activation_fn=lrelu)
        critic_values = tl.fully_connected(fc1, self.n_actions_per_joint, activation_fn=None)
        self.critic_values[joint_name] = critic_values
        self.returns[joint_name] = tf.placeholder(shape=critic_values.get_shape()[:1], dtype=tf.float32, name="return_{}".format(joint_name))
        actions = self.picked_actions[joint_name]
        indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
        self.critic_values_picked_actions[joint_name] = tf.gather_nd(critic_values, indices)
        self.critic_loss[joint_name] = tf.losses.huber_loss(
            self.returns[joint_name] * self.reward_scaling_factor,
            self.critic_values_picked_actions[joint_name],
            delta=0.5
        )
        ### ACTIONS:
        self.greedy_actions_indices[joint_name] = tf.cast(tf.argmax(self.critic_values[joint_name], axis=1), dtype=tf.int32)
        shape = tf.shape(self.greedy_actions_indices[joint_name])
        condition = tf.greater(tf.random_uniform(shape=shape), self.epsilon)
        random = tf.random_uniform(shape=shape, maxval=self.n_actions_per_joint, dtype=tf.int32)
        self.sampled_actions_indices[joint_name] = tf.where(condition, x=self.greedy_actions_indices[joint_name], y=random)
        # summaries
        summaries = []
        mean_abs_return = tf.reduce_mean(tf.abs(self.returns[joint_name] * self.reward_scaling_factor))
        summaries.append(tf.summary.scalar("/joint/{}/mean_abs_return".format(joint_name), mean_abs_return))
        mean, var = tf.nn.moments(tf.abs(self.critic_values_picked_actions[joint_name] - self.returns[joint_name] * self.reward_scaling_factor), axes=[0])
        summaries.append(tf.summary.scalar("/joint/{}/mean_abs_distance".format(joint_name), mean))
        summaries.append(tf.summary.scalar("/joint/{}/std_abs_distance".format(joint_name), tf.sqrt(var)))
        summaries.append(tf.summary.histogram("return_{}".format(joint_name), self.returns[joint_name]))
        self.joint_summary[joint_name] = tf.summary.merge(summaries)

    def define_critic(self):
        """Defines the critics for each joints"""
        # done once for every joint
        self.picked_actions = {}
        self.critic_values = {}
        self.returns = {}
        self.critic_loss = {}
        self.greedy_actions_indices = {}
        self.sampled_actions_indices = {}
        self.joint_summary = {}
        self.critic_values_picked_actions = {}
        for joint_name in ["tilt", "pan", "vergence"]:
            # done once per joint
            self.define_critic_joint(joint_name)

    def define_inps(self):
        """Generates the different 32x32 image patches for the given ratios."""
        self.scales_inp_2_frames = {}
        self.scales_inp_4_frames = {}
        for ratio in self.ratios:
            self.define_scale_inp(ratio)

    def define_scale_inp(self, ratio):
        """Crops, downscales and converts a central region of the camera images
        Input: Image patch with size 16*ratio x 16*ratio --> Output: Image patch with 16 x 16"""
        self.crop_side_length = 32  # should be defined in the command line
        height_slice = slice(
            (240 - (self.crop_side_length * ratio)) // 2,
            (240 + (self.crop_side_length * ratio)) // 2)
        width_slice = slice(
            (320 - (self.crop_side_length * ratio)) // 2,
            (320 + (self.crop_side_length * ratio)) // 2)
        # CROP
        scale_4_frames = self.stacked_4_frames[:, height_slice, width_slice, :]
        # DOWNSCALE
        scale_4_frames = tf.image.resize_bilinear(scale_4_frames, [self.crop_side_length, self.crop_side_length]) * 2 - 1
        self.scales_inp_4_frames[ratio] = tf.placeholder_with_default(scale_4_frames, shape=scale_4_frames.get_shape(), name="4_frames_input_at_scale_{}".format(ratio))
        if self.turn_2_frames_vergence_on:
            scale_2_frames = scale_4_frames[..., 6:]
            self.scales_inp_2_frames[ratio] = tf.placeholder_with_default(scale_2_frames, shape=scale_2_frames.get_shape(), name="2_frames_input_at_scale_{}".format(ratio))

    def define_2_frames_autoencoder(self, filter_size=8, stride=4):
        self.scale_latent_2_frames = {}
        self.scale_latent_conv_2_frames = {}
        self.scale_rec_2_frames = {}
        self.patch_recerrs_2_frames = {}
        self.scale_recerrs_2_frames = {}
        self.scale_recerr_2_frames = {}
        for ratio in self.ratios:
            self.define_2_frames_autoencoder_scale(ratio, filter_size=filter_size, stride=stride)
        self.latent_2_frames = tf.concat([self.scale_latent_2_frames[r] for r in self.ratios], axis=1)
        self.autoencoder_loss_2_frames = sum([self.scale_recerr_2_frames[r] for r in self.ratios])
        self.total_recerrs_2_frames = self.scale_recerrs_2_frames[self.ratios[0]]
        for ratio in self.ratios[1:]:
            self.total_recerrs_2_frames += self.scale_recerrs_2_frames[ratio]
        self.total_recerrs_2_frames /= self.n_scales

    def define_2_frames_autoencoder_scale(self, ratio, filter_size=4, stride=2):
        """Defines an autoencoder that operates at one scale (one downscaling ratio)"""
        inp = self.scales_inp_2_frames[ratio]
        batch_size = tf.shape(inp)[0]
        conv1 = tl.conv2d(inp + 0, filter_size ** 2 * 3 * 2 // 4, filter_size, stride, "valid", activation_fn=lrelu)
        # conv2 = tl.conv2d(conv1, filter_size ** 2 * 3 * 2 // 4, 1, 1, "valid", activation_fn=lrelu)
        # conv3 = tl.conv2d(conv2, filter_size ** 2 * 3 * 2 // 8, 1, 1, "valid", activation_fn=lrelu)
        # bottleneck = tl.conv2d(conv3, filter_size ** 2 * 3 * 2 // 8, 1, 1, "valid", activation_fn=lrelu)
        bottleneck = tl.conv2d(conv1, filter_size ** 2 * 3 * 2 // 16, 1, 1, "valid", activation_fn=lrelu)
        # conv5 = tl.conv2d(bottleneck, filter_size ** 2 * 3 * 2 // 4, 1, 1, "valid", activation_fn=lrelu)
        # conv6 = tl.conv2d(conv5, filter_size ** 2 * 3 * 2 // 2, 1, 1, "valid", activation_fn=lrelu)
        # reconstruction = tl.conv2d(conv6, filter_size ** 2 * 3 * 2, 1, 1, "valid", activation_fn=None)
        reconstruction = tl.conv2d(bottleneck, filter_size ** 2 * 3 * 2, 1, 1, "valid", activation_fn=None)
        target = tf.extract_image_patches(
            inp, [1, filter_size, filter_size, 1], [1, stride, stride, 1], [1, 1, 1, 1], 'VALID'
        )
        size = np.prod(bottleneck.get_shape()[1:])
        self.scale_latent_conv_2_frames[ratio] = bottleneck
        self.scale_latent_2_frames[ratio] = tf.reshape(bottleneck, (-1, size))
        self.scale_rec_2_frames[ratio] = reconstruction
        self.patch_recerrs_2_frames[ratio] = tf.reduce_mean((self.scale_rec_2_frames[ratio] - target) ** 2, axis=3)
        self.scale_recerrs_2_frames[ratio] = tf.reduce_mean(self.patch_recerrs_2_frames[ratio], axis=[1, 2])
        self.scale_recerr_2_frames[ratio] = tf.reduce_sum(self.scale_recerrs_2_frames[ratio])  # sum on the batch dimension

    def define_4_frames_autoencoder(self, filter_size=8, stride=4):
        self.scale_latent_4_frames = {}
        self.scale_latent_conv_4_frames = {}
        self.scale_rec_4_frames = {}
        self.patch_recerrs_4_frames = {}
        self.scale_recerrs_4_frames = {}
        self.scale_recerr_4_frames = {}
        self.scale_tensorboard_images = {}
        for ratio in self.ratios:
            self.define_4_frames_autoencoder_scale(ratio, filter_size=filter_size, stride=stride)
        self.latent_4_frames = tf.concat([self.scale_latent_4_frames[r] for r in self.ratios], axis=1)
        self.autoencoder_loss_4_frames = sum([self.scale_recerr_4_frames[r] for r in self.ratios])
        self.total_recerrs_4_frames = self.scale_recerrs_4_frames[self.ratios[0]]
        for ratio in self.ratios[1:]:
            self.total_recerrs_4_frames += self.scale_recerrs_4_frames[ratio]
        self.total_recerrs_4_frames /= self.n_scales

    def define_4_frames_autoencoder_scale(self, ratio, filter_size=4, stride=2):
        """Defines an autoencoder that operates at one scale (one downscaling ratio)"""
        inp = self.scales_inp_4_frames[ratio]
        batch_size = tf.shape(inp)[0]
        conv1 = tl.conv2d(inp + 0, filter_size ** 2 * 3 * 4 // 4, filter_size, stride, "valid", activation_fn=lrelu)
        # conv2 = tl.conv2d(conv1, filter_size ** 2 * 3 * 4 // 4, 1, 1, "valid", activation_fn=lrelu)
        # conv3 = tl.conv2d(conv2, filter_size ** 2 * 3 * 4 // 8, 1, 1, "valid", activation_fn=lrelu)
        # bottleneck = tl.conv2d(conv3, filter_size ** 2 * 3 * 4 // 8, 1, 1, "valid", activation_fn=lrelu)
        bottleneck = tl.conv2d(conv1, filter_size ** 2 * 3 * 4 // 16, 1, 1, "valid", activation_fn=lrelu)
        # conv5 = tl.conv2d(bottleneck, filter_size ** 2 * 3 * 4 // 4, 1, 1, "valid", activation_fn=lrelu)
        # conv6 = tl.conv2d(conv5, filter_size ** 2 * 3 * 4 // 2, 1, 1, "valid", activation_fn=lrelu)
        # reconstruction = tl.conv2d(conv6, filter_size ** 2 * 3 * 4, 1, 1, "valid", activation_fn=None)
        reconstruction = tl.conv2d(bottleneck, filter_size ** 2 * 3 * 4, 1, 1, "valid", activation_fn=None)
        target = tf.extract_image_patches(
            inp, [1, filter_size, filter_size, 1], [1, stride, stride, 1], [1, 1, 1, 1], 'VALID'
        )
        size = np.prod(bottleneck.get_shape()[1:])
        self.scale_latent_conv_4_frames[ratio] = bottleneck
        self.scale_latent_4_frames[ratio] = tf.reshape(bottleneck, (-1, size))
        self.scale_rec_4_frames[ratio] = reconstruction
        self.patch_recerrs_4_frames[ratio] = tf.reduce_mean((self.scale_rec_4_frames[ratio] - target) ** 2, axis=3)
        self.scale_recerrs_4_frames[ratio] = tf.reduce_mean(self.patch_recerrs_4_frames[ratio], axis=[1, 2])
        self.scale_recerr_4_frames[ratio] = tf.reduce_sum(self.scale_recerrs_4_frames[ratio])  # sum on the batch dimension
        # ### Images for tensorboard:
        # CONTENT:
        # input                 reconstruction
        # left_t-1   left_t     left_t-1   left_t
        # right_t-1  right_t    right_t-1  right_t
        # NAMES:
        # input                 reconstruction
        # q          w          e          r
        # a          s          d          f
        n_patches = (inp.get_shape()[1] - filter_size + stride) // stride
        reconstructions = tf.reshape(reconstruction[-1], (n_patches, n_patches, filter_size, filter_size, 12))
        reconstructions = tf.transpose(reconstructions, perm=[0, 2, 1, 3, 4])
        reconstructions = tf.reshape(reconstructions, (n_patches * filter_size, n_patches * filter_size, 12))
        e = reconstructions[..., 0:3]
        r = reconstructions[..., 3:6]
        d = reconstructions[..., 6:9]
        f = reconstructions[..., 9:12]
        inputs = tf.reshape(target[-1], (n_patches, n_patches, filter_size, filter_size, 12))
        inputs = tf.transpose(inputs, perm=[0, 2, 1, 3, 4])
        inputs = tf.reshape(inputs, (n_patches * filter_size, n_patches * filter_size, 12))
        q = inputs[..., 0:3]
        w = inputs[..., 3:6]
        a = inputs[..., 6:9]
        s = inputs[..., 9:12]
        top_row = tf.concat([q, w, e, r], axis=1)
        bottom_row = tf.concat([a, s, d, f], axis=1)
        img = tf.concat([top_row, bottom_row], axis=0)
        self.scale_tensorboard_images[ratio] = tf.expand_dims(img, axis=0)

    def define_epsilon(self):
        self.epsilon = tf.Variable(self.epsilon_init)
        self.epsilon_update = self.epsilon.assign(self.epsilon * self.epsilon_decay)

    def define_networks(self):
        self.left_cam = tf.placeholder(shape=(None, 240, 320, 3), dtype=tf.float32, name="left_cam")
        self.left_cam_before = tf.placeholder(shape=(None, 240, 320, 3), dtype=tf.float32, name="left_cam_before")
        self.right_cam = tf.placeholder(shape=(None, 240, 320, 3), dtype=tf.float32, name="right_cam")
        self.right_cam_before = tf.placeholder(shape=(None, 240, 320, 3), dtype=tf.float32, name="right_cam_before")
        self.stacked_4_frames = tf.concat([
            self.left_cam_before,
            self.right_cam_before,
            self.left_cam,
            self.right_cam
        ], axis=-1)  # (None, 240, 320, 12)
        ### graph definitions:
        self.define_inps()
        with tf.variable_scope("model"):
            self.define_4_frames_autoencoder(filter_size=8, stride=4)
            if self.turn_2_frames_vergence_on:
                self.define_2_frames_autoencoder(filter_size=8, stride=4)
        self.define_epsilon()
        self.define_critic()
        ### summaries
        batch_size = tf.cast(tf.shape(self.total_recerrs_4_frames)[0], tf.float32)
        summary_loss_4_frames = tf.summary.scalar("/autoencoder/loss_4_frames", self.autoencoder_loss_4_frames / batch_size / self.n_scales)
        if self.turn_2_frames_vergence_on:
            summary_loss_2_frames = tf.summary.scalar("/autoencoder/loss_2_frames", self.autoencoder_loss_2_frames / batch_size / self.n_scales)
        summary_per_joint = [tf.summary.scalar("/critic/{}".format(jn), self.critic_loss[jn] / batch_size) for jn in ["tilt", "pan", "vergence"]]
        summary_images = [tf.summary.image("ratio_{}".format(r), self.scale_tensorboard_images[r], max_outputs=1) for r in self.ratios]
        summaries = []
        for joint_name in ["tilt", "pan", "vergence"]:
            summaries.append(self.joint_summary[joint_name])
        summaries += summary_per_joint
        summaries += summary_images
        summaries += [summary_loss_4_frames]
        if self.turn_2_frames_vergence_on:
            summaries += [summary_loss_2_frames]
        self.summary = tf.summary.merge(summaries)
        ### Ops
        self.update_count = tf.Variable(0, dtype=tf.int32)
        use_locking = True
        self.update_count_inc = self.update_count.assign_add(1, use_locking=use_locking)
        self.optimizer_autoencoder = tf.train.AdamOptimizer(self.model_lr, use_locking=use_locking)
        self.optimizer_critic = tf.train.AdamOptimizer(self.critic_lr, use_locking=use_locking)
        if self.turn_2_frames_vergence_on:
            self.train_op_autoencoder = self.optimizer_autoencoder.minimize(self.autoencoder_loss_4_frames + self.autoencoder_loss_2_frames)
        else:
            self.train_op_autoencoder = self.optimizer_autoencoder.minimize(self.autoencoder_loss_4_frames)
        self.train_op_critic = self.optimizer_critic.minimize(sum(self.critic_loss.values()))
        self.train_op = tf.group([self.train_op_autoencoder, self.train_op_critic] + tf.get_collection(tf.GraphKeys.UPDATE_OPS))

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
        self.model_saver.restore(self.sess, os.path.normpath(path + "/network.ckpt"))
        self.pipe.send("{} model restored from {}".format(self.name, path))

    def restore_all(self, path):
        """Restores from a checkpoint
        The Experiment class calls this methode on the worker 0 only
        """
        self.saver.restore(self.sess, os.path.normpath(path + "/network.ckpt"))
        self.pipe.send("{} variables restored from {}".format(self.name, path))

    def test_fast(self):
        fetches = {
            "total_recerrs_4_frames": self.total_recerrs_4_frames,
            "scale_recerrs_4_frames": self.scale_recerrs_4_frames,
            "critic_value": self.critic_values,
            "action_index": self.greedy_actions_indices
        }
        if self.turn_2_frames_vergence_on:
            fetches["total_recerrs_2_frames"] = self.total_recerrs_2_frames
            fetches["scale_recerrs_2_frames"] = self.scale_recerrs_2_frames
        while not self.test_cases_queue.empty():
            try:
                test_cases_chunk, policy_independent_inputs = self.test_cases_queue.get(timeout=30)
            except queue.Empty:
                print("{} found an empty queue".format(self.name))
                break
            test_data = np.zeros(shape=(len(test_cases_chunk), 1), dtype=self.testing_data_type)
            feed_dict = {
                self.left_cam_before: policy_independent_inputs[:, 0],
                self.right_cam_before: policy_independent_inputs[:, 1],
                self.left_cam: policy_independent_inputs[:, 2],
                self.right_cam: policy_independent_inputs[:, 3]
            }
            data = self.sess.run(fetches, feed_dict=feed_dict)
            action_value = self.actions_indices_to_values(data["action_index"])
            test_data[:, 0]["action_index"] = np.array([data["action_index"][jn] for jn in ["tilt", "pan", "vergence"]]).T
            test_data[:, 0]["action_value"] = (0, 0, 0)  # useless
            test_data[:, 0]["critic_value_tilt"] = data["critic_value"]["tilt"]
            test_data[:, 0]["critic_value_pan"] = data["critic_value"]["pan"]
            test_data[:, 0]["critic_value_vergence"] = data["critic_value"]["vergence"]
            test_data[:, 0]["object_distance"] = 0  # set to zero as well
            test_data[:, 0]["eye_position"] = (0, 0, 0)  # useless
            test_data[:, 0]["eye_speed"] = (0, 0)  # useless
            test_data[:, 0]["speed_error"] = test_cases_chunk["speed_error"]
            test_data[:, 0]["vergence_error"] = test_cases_chunk["vergence_error"]
            test_data[:, 0]["total_recerrs_4_frames"] = data["total_recerrs_4_frames"]
            test_data[:, 0]["scale_recerrs_4_frames"] = np.stack([data["scale_recerrs_4_frames"][r] for r in self.ratios], axis=-1)
            if self.turn_2_frames_vergence_on:
                test_data[:, 0]["total_recerrs_2_frames"] = data["total_recerrs_2_frames"]
                test_data[:, 0]["scale_recerrs_2_frames"] = np.stack([data["scale_recerrs_2_frames"][r] for r in self.ratios], axis=-1)
            self.testing_data_queue.put((test_cases_chunk, test_data))
        print("{} queue is empty, no more quick tests".format(self.name))
        self.pipe.send("{} no more fast test cases, going IDLE".format(self.name))

    def test(self):
        """Performs tests in the environment, using the greedy policy.
        The resulting measures are sent back to the Experiment class (the server)
        """
        # import imageio  # temporary
        fetches = {
            "total_recerrs_4_frames": self.total_recerrs_4_frames,
            "scale_recerrs_4_frames": self.scale_recerrs_4_frames,
            "critic_value": self.critic_values,
            "action_index": self.greedy_actions_indices
        }
        if self.turn_2_frames_vergence_on:
            fetches["total_recerrs_2_frames"] = self.total_recerrs_2_frames
            fetches["scale_recerrs_2_frames"] = self.scale_recerrs_2_frames
        while not self.test_cases_queue.empty():
            try:
                test_case = self.test_cases_queue.get(timeout=1)
            except queue.Empty:
                print("{} found an empty queue".format(self.name))
                break
            test_data = np.zeros(test_case["n_iterations"], dtype=self.testing_data_type)
            # vergence_error = eyes_vergence - correct_vergence
            # eyes_vergence = correct_vergence + vergence_error
            vergence_init = to_angle(test_case["object_distance"]) + test_case["vergence_error"]
            # speed_error = eyes_speed - correct_speed
            # eyes_speed = 0
            # correct_speed = screen_speed = - speed_error
            screen_speed = -test_case["speed_error"]
            ### initialize environment, step simulation, take pictures, move screen
            self.environment.robot.reset_speed()
            self.environment.robot.set_position([0, 0, vergence_init], joint_limit_type="none")
            self.environment.screen.set_texture(test_case["stimulus"])
            self.environment.screen.set_trajectory(
                test_case["object_distance"],
                screen_speed[0],
                screen_speed[1],
                test_case["depth_speed"],
                preinit=True)
            self.environment.step()
            left_image_before, right_image_before = self.environment.robot.get_vision()
            ###
            for i in range(test_case["n_iterations"]):
                self.environment.step()
                left_image, right_image = self.environment.robot.get_vision()
                feed_dict = {
                    self.left_cam: [left_image],
                    self.left_cam_before: [left_image_before],
                    self.right_cam: [right_image],
                    self.right_cam_before: [right_image_before]
                }
                data = self.sess.run(fetches, feed_dict=feed_dict)
                action_value = self.actions_indices_to_values(data["action_index"])
                test_data[i]["action_index"] = [data["action_index"][jn] for jn in ["tilt", "pan", "vergence"]]
                test_data[i]["action_value"] = action_value
                test_data[i]["critic_value_tilt"] = data["critic_value"]["tilt"]
                test_data[i]["critic_value_pan"] = data["critic_value"]["pan"]
                test_data[i]["critic_value_vergence"] = data["critic_value"]["vergence"]
                test_data[i]["object_distance"] = self.environment.screen.distance
                test_data[i]["eye_position"] = self.environment.robot.position
                test_data[i]["eye_speed"] = self.environment.robot.speed
                test_data[i]["speed_error"] = test_data[i]["eye_speed"] - screen_speed
                test_data[i]["vergence_error"] = self.environment.robot.get_vergence_error(test_case["object_distance"])
                test_data[i]["total_recerrs_4_frames"] = data["total_recerrs_4_frames"]
                test_data[i]["scale_recerrs_4_frames"] = np.stack([data["scale_recerrs_4_frames"][r] for r in self.ratios], axis=-1)
                if self.turn_2_frames_vergence_on:
                    test_data[i]["total_recerrs_2_frames"] = data["total_recerrs_2_frames"]
                    test_data[i]["scale_recerrs_2_frames"] = np.stack([data["scale_recerrs_2_frames"][r] for r in self.ratios], axis=-1)
                if i + 1 != test_case["n_iterations"]:
                    self.environment.robot.set_action(action_value, joint_limit_type="none")
                left_image_before = left_image
                right_image_before = right_image
            self.testing_data_queue.put((test_case, test_data))
        self.pipe.send("{} no more test cases, going IDLE".format(self.name))

    def get_episode(self):
        """Get the training data that the RL algorithm needs,
        and stores training infos in a buffer that must be flushed regularly
        See the help guide about data formats
        """
        episode_number, epsilon = self.sess.run([self.episode_count_inc, self.epsilon])
        log_data = episode_number % 10 == 0
        if episode_number % 100 == 0:
            print("{} simulating episode {}\tepsilon {:.2f}".format(self.name, episode_number, epsilon))
        if episode_number % 1000 == 0:
            if self._n_time_measurements > 0:
                print("{} simulation time: {:.2f}sec/episode\tof which {:.2f} is forward pass\ttraining time: {:.2f}sec/episode".format(self.name, self._tsimulation / self._n_time_measurements, self._trun / self._n_time_measurements, self._ttrain / self._n_time_measurements))
                self._tsimulation = 0
                self._trun = 0
                self._ttrain = 0
                self._n_time_measurements = 0

        self.environment.episode_reset(preinit=True)
        self.environment.step()  # moves the screen
        left_image_before, right_image_before = self.environment.robot.get_vision()
        for iteration in range(self.episode_length + 1):  # + 1 for the additional / sacrificial iteration
            self.environment.step()  # moves the screen
            left_image, right_image = self.environment.robot.get_vision()
            feed_dict = {
                self.left_cam: [left_image],
                self.left_cam_before: [left_image_before],
                self.right_cam: [right_image],
                self.right_cam_before: [right_image_before]
            }
            trun_before = time.time()
            data = self.sess.run(self.training_behaviour_fetches, feed_dict)
            trun_after = time.time()
            self._trun += trun_after - trun_before
            self.fill_behaviour_data(iteration, data)  # missing return targets
            if log_data:
                self.fill_training_data(iteration, data)
            self.fill_recerrs_data(iteration, data)    # for computation of return targets
            self.environment.robot.set_action(self.to_action(data))
            left_image_before = left_image
            right_image_before = right_image

        ### COMPUTE TARGET RETURN  --> for every joint / every level / every ratio
        for joint_name in ["tilt", "pan", "vergence"]:
            feed_dict[self.picked_actions[joint_name]] = data["sampled_actions_indices"][joint_name]
        all_start = self.sess.run(self.critic_values_picked_actions, feed_dict=feed_dict)

        for joint_index, joint_name in enumerate(["tilt", "pan", "vergence"]):
            encoder_index = 1 if self.turn_2_frames_vergence_on and joint_name == "vergence" else 0
            all_rewards = self._recerrs_data[:-1, encoder_index] - self._recerrs_data[1:, encoder_index]
            all_bootstrap = all_start[joint_name][0]
            all_returns = to_return(all_rewards, start=all_bootstrap, discount_factor=self.discount_factor)
            self._behaviour_data["total_target_return"][:-1, joint_index] = all_returns

        # store in buffer
        self.buffer.incorporate(self._behaviour_data[:-1])
        if log_data:
            self._training_data["episode_number"] = episode_number
            encoder_index = 0
            self._training_data["all_rewards"][:-1, encoder_index] = all_rewards
            self._training_data["total_target_return"][:-1, encoder_index] = all_returns
            if self.turn_2_frames_vergence_on:
                encoder_index = 1  # 1 stands for 2_frames encoder
                self._training_data["all_rewards"][:-1, encoder_index] = all_rewards
                self._training_data["total_target_return"][:-1, encoder_index] = all_returns
            self.register_training_data()
        return episode_number

    def fill_training_data(self, iteration, data):
        self._training_data[iteration]["sampled_actions_indices"] = [data["sampled_actions_indices"][joint_name] for joint_name in ["tilt", "pan", "vergence"]]
        self._training_data[iteration]["greedy_actions_indices"] = [data["greedy_actions_indices"][joint_name] for joint_name in ["tilt", "pan", "vergence"]]
        encoder_index = 0  # 0 stands for 4_frames encoder
        self._training_data[iteration]["patch_recerrs"][encoder_index] = [data["patch_recerrs_4_frames"][r] for r in self.ratios]
        self._training_data[iteration]["scale_recerrs"][encoder_index] = [data["scale_recerrs_4_frames"][r] for r in self.ratios]
        self._training_data[iteration]["total_recerrs"][encoder_index] = data["total_recerrs_4_frames"]
        if self.turn_2_frames_vergence_on:
            encoder_index = 1  # 1 stands for 2_frames encoder
            self._training_data[iteration]["patch_recerrs"][encoder_index] = [data["patch_recerrs_2_frames"][r] for r in self.ratios]
            self._training_data[iteration]["scale_recerrs"][encoder_index] = [data["scale_recerrs_2_frames"][r] for r in self.ratios]
            self._training_data[iteration]["total_recerrs"][encoder_index] = data["total_recerrs_2_frames"]
        self._training_data[iteration]["object_distance"] = self.environment.screen.distance
        self._training_data[iteration]["object_speed"] = self.environment.screen.tilt_pan_speed
        self._training_data[iteration]["eyes_position"] = self.environment.robot.position
        self._training_data[iteration]["eyes_speed"] = self.environment.robot.speed

    def fill_behaviour_data(self, iteration, data):
        self._behaviour_data[iteration]["scales_inp"] = np.concatenate([data["scales_inp_4_frames"][r] for r in self.ratios], axis=0)
        self._behaviour_data[iteration]["sampled_actions_indices"] = [data["sampled_actions_indices"][joint_name] for joint_name in ["tilt", "pan", "vergence"]]

    def fill_recerrs_data(self, iteration, data):
        encoder_index = 0  # 0 stands for 4_frames encoder
        self._recerrs_data[iteration, encoder_index] = data["total_recerrs_4_frames"][0]
        if self.turn_2_frames_vergence_on:
            encoder_index = 1  # 1 stands for 2_frames encoder
            self._recerrs_data[iteration, encoder_index] = data["total_recerrs_2_frames"][0]

    # def define_actions_sets(self, tilt=True, pan=True, vergence=True):
    def define_actions_sets(self):
        """Defines the pan/tilt/vergence action sets
        At the moment, pan and tilt are comprised only of zeros"""
        self.n_actions_per_joint = len(actions_set_values_vergence)
        self.actions_set_tilt = actions_set_values_tilt
        self.actions_set_pan = actions_set_values_pan
        self.actions_set_vergence = actions_set_values_vergence

    def actions_indices_to_values(self, indices_dict):
        return [self.actions_set_tilt[indices_dict["tilt"][0]],
                self.actions_set_pan[indices_dict["pan"][0]],
                self.actions_set_vergence[indices_dict["vergence"][0]]]

    def to_action(self, data):
        return [self.actions_set_tilt[data["sampled_actions_indices"]["tilt"][0]],
                self.actions_set_pan[data["sampled_actions_indices"]["pan"][0]],
                self.actions_set_vergence[data["sampled_actions_indices"]["vergence"][0]]]

    def train_one_episode(self):
        """Updates the networks weights according to the transitions states and actions"""
        data = self.buffer.random_batch
        feed_dict = {self.scales_inp_4_frames[r]: data["scales_inp"][:, i] for i, r in enumerate(self.ratios)}
        feed_dict[self.is_training] = True
        if self.turn_2_frames_vergence_on:
            feed_dict_2 = {self.scales_inp_2_frames[r]: data["scales_inp"][:, i, ..., 6:] for i, r in enumerate(self.ratios)}
            feed_dict.update(feed_dict_2)
        for i, joint_name in enumerate(["tilt", "pan", "vergence"]):
            feed_dict[self.picked_actions[joint_name]] = data["sampled_actions_indices"][:, i]
            feed_dict[self.returns[joint_name]] = data["total_target_return"][:, i]
        ret = self.sess.run(self.update_fetches, feed_dict=feed_dict)
        if ret["episode_count"] % 100 == 0:
            self.add_summary(self.sess.run(self.summary, feed_dict=feed_dict), global_step=ret["episode_count"])
        return ret["episode_count"]

    def train(self, n_episodes):
        """Train until n_episodes have been simulated
        See the update_factor parameter"""
        before_n_episodes = self.sess.run(self.episode_count)
        current_n_episode = before_n_episodes
        after_n_episode = before_n_episodes + n_episodes
        while current_n_episode < after_n_episode:
            tstart = time.time()
            current_n_episode = self.get_episode()
            tsimulation = time.time()
            current_n_episode = self.train_one_episode()
            ttrain = time.time()
            self._tsimulation += tsimulation - tstart
            self._ttrain += ttrain - tsimulation
            self._n_time_measurements += 1
            if current_n_episode >= after_n_episode:
                break
        self.pipe.send("{} going IDLE".format(self.name))

    # ToDo: Implement random subsample and sorting if needed, additional: write test parameters into video frame
    def make_video_test_cases(self, path, test_cases, training=False, test_case_limit=100, sort_by=None, rand_subsample=False):
        actions_indices = self.greedy_actions_indices if not training else self.sampled_actions_indices
        rectangles = [(
            160 - self.crop_side_length / 2 * r,
            120 - self.crop_side_length / 2 * r,
            160 + self.crop_side_length / 2 * r,
            120 + self.crop_side_length / 2 * r
        ) for r in self.ratios]
        print("{} will store the video under {}".format(self.name, path))
        with get_writer(path, fps=25, format="mp4") as writer:
            for pos, test_case in enumerate(test_cases):
                if pos > test_case_limit:
                    break
                print("{}: Test Case {}/{} ({} test cases in total)".format(self.name, pos + 1, min(
                    test_case_limit, len(test_cases)), len(test_cases)))
                print("\t Parameters for this test case: {}".format(test_case))
                vergence_init = to_angle(test_case["object_distance"]) + test_case["vergence_error"]
                screen_speed = -test_case["speed_error"]
                ### initialize environment, step simulation, take pictures, move screen
                self.environment.robot.reset_speed()
                self.environment.robot.set_position([0, 0, vergence_init], joint_limit_type="none")
                self.environment.screen.set_texture(test_case["stimulus"])
                self.environment.screen.set_trajectory(
                    test_case["object_distance"],
                    screen_speed[0],
                    screen_speed[1],
                    test_case["depth_speed"],
                    preinit=True)
                self.environment.step()
                left_image_before, right_image_before = self.environment.robot.get_vision()
                for iteration in range(self.episode_length + 1):  # + 1 for the additional / sacrificial iteration
                    self.environment.step()  # moves the screen
                    left_image, right_image = self.environment.robot.get_vision()
                    feed_dict = {
                        self.left_cam: [left_image],
                        self.left_cam_before: [left_image_before],
                        self.right_cam: [right_image],
                        self.right_cam_before: [right_image_before]
                    }
                    action = self.sess.run(actions_indices, feed_dict)
                    self.environment.robot.set_action(self.actions_indices_to_values(action))
                    left_image_before = left_image
                    right_image_before = right_image
                    object_distance = self.environment.screen.distance
                    vergence_error = self.environment.robot.get_vergence_error(object_distance)
                    eyes_speed = self.environment.robot.speed
                    screen_speed = self.environment.screen.tilt_pan_speed
                    tilt_speed_error = eyes_speed[0] - screen_speed[0]
                    pan_speed_error = eyes_speed[1] - screen_speed[1]
                    print("vergence error: {: .4f}    tilt speed error: {: .4f}    pan speed error: {: .4f}".format(
                        vergence_error, tilt_speed_error, pan_speed_error))
                    frame = make_frame(left_image, right_image, object_distance, vergence_error, pos + 1,
                                       test_case_limit, rectangles)
                    writer.append_data(frame)
                    if iteration == 0 or iteration == self.episode_length:
                        for i in range(12):
                            writer.append_data(frame)
        self.pipe.send("{} going IDLE".format(self.name))

    def make_video(self, path, n_episodes, training=False):
        actions_indices = self.greedy_actions_indices if not training else self.sampled_actions_indices
        rectangles = [(
            160 - self.crop_side_length / 2 * r,
            120 - self.crop_side_length / 2 * r,
            160 + self.crop_side_length / 2 * r,
            120 + self.crop_side_length / 2 * r
            ) for r in self.ratios]
        print("{} will store the video under {}".format(self.name, path))
        with get_writer(path, fps=25, format="mp4") as writer:
            for episode_number in range(n_episodes):
                print("{} episode {}/{}".format(self.name, episode_number + 1, n_episodes))
                self.environment.episode_reset(preinit=True)
                self.environment.step()  # moves the screen
                left_image_before, right_image_before = self.environment.robot.get_vision()
                for iteration in range(self.episode_length + 1):  # + 1 for the additional / sacrificial iteration
                    self.environment.step()  # moves the screen
                    left_image, right_image = self.environment.robot.get_vision()
                    feed_dict = {
                        self.left_cam: [left_image],
                        self.left_cam_before: [left_image_before],
                        self.right_cam: [right_image],
                        self.right_cam_before: [right_image_before]
                    }
                    action = self.sess.run(actions_indices, feed_dict)
                    self.environment.robot.set_action(self.actions_indices_to_values(action))
                    left_image_before = left_image
                    right_image_before = right_image
                    object_distance = self.environment.screen.distance
                    vergence_error = self.environment.robot.get_vergence_error(object_distance)
                    eyes_speed = self.environment.robot.speed
                    screen_speed = self.environment.screen.tilt_pan_speed
                    tilt_speed_error = eyes_speed[0] - screen_speed[0]
                    pan_speed_error = eyes_speed[1] - screen_speed[1]
                    print("vergence error: {: .4f}    tilt speed error: {: .4f}    pan speed error: {: .4f}".format(vergence_error, tilt_speed_error, pan_speed_error))
                    frame = make_frame(left_image, right_image, object_distance, vergence_error, episode_number + 1, n_episodes, rectangles)
                    writer.append_data(frame)
                    if iteration == 0 or iteration == self.episode_length:
                        for i in range(12):
                            writer.append_data(frame)
        self.pipe.send("{} going IDLE".format(self.name))
