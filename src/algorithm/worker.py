import tensorflow as tf
from algorithm.replay_buffer import Buffer
from log.logging import DataLogger
from enviroment.environment import Environment
import tensorflow.contrib.layers as tl
import numpy as np
from algorithm.returns import tf_returns
import time
import os
from helper.utils import to_angle
from imageio import get_writer
from PIL import ImageDraw, Image #, ImageFont

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
        self.ratios = list(range(1, 4)) # Last number is excluded: list(range(1, 3)) -> [1, 2]
        #self.ratios = [1, 2, 3]
        self.define_networks()
        self.define_actions_sets()

        # Manage Data Logging
        self.data_logger = DataLogger()
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
        scale = self.left_cams_stacked[:, height_slice, width_slice, :]
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
        left_before = left_right[..., 3:]
        image_left = tf.concat([left, left_before], axis=0)
        left_right = tf.reshape(target[-1], (n_patches, n_patches, filter_size, filter_size, 6))
        left_right = tf.transpose(left_right, perm=[0, 2, 1, 3, 4])
        left_right = tf.reshape(left_right, (n_patches * filter_size, n_patches * filter_size, 6))
        left = left_right[..., :3]
        left_before = left_right[..., 3:]
        image_right = tf.concat([left, left_before], axis=0)
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
        self.left_cam_before = tf.placeholder(shape=(None, 240, 320, 3), dtype=tf.float32)
        self.left_cams_stacked = tf.concat([self.left_cam, self.left_cam_before], axis=-1)  # (None, 240, 320, 12)
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
        """Performs tests in the enviroment, using the greedy policy.
        proper_display_of_chunk_of_test_cases is a wrapper around a chunk_of_test_cases (for pretty printing)
        chunk_of_test_cases contains all informations about the tests that must be performed in the enviroment
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
            left_image, right_image = self.environment.robot.get_vision()
            for i in range(test_case["n_iterations"]):
                left_image_before, right_image_before = left_image, right_image
                left_image, right_image = self.environment.robot.get_vision()
                feed_dict = {self.left_cam: [left_image], self.left_cam_before: [left_image_before]}
                data = self.sess.run(fetches, feed_dict=feed_dict)
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
        """Performs one episode of fake training (for visualizing a trained agent in the enviroment, see playback.py)
        The greedy boolean specifies wether the greedy or sampled policy should be used
        """
        fetches = (self.greedy_actions_indices if greedy else self.sampled_actions_indices), self.rewards__partial
        self.environment.episode_reset()
        mean = 0
        total_reward__partial = 0
        left_image, right_image = self.environment.robot.get_vision()
        for iteration in range(self.episode_length):
            left_image_before, right_image_before = left_image, right_image
            left_image, right_image = self.environment.robot.get_vision()
            feed_dict = {self.left_cam: [left_image], self.left_cam_before: [left_image_before]}
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
        self.environment.screen.set_episode_iteration(-2)
        self.environment.step()
        left_image, right_image = self.environment.robot.get_vision()
        self.environment.step()
        for iteration in range(self.episode_length):
            left_image_before, right_image_before = left_image, right_image
            left_image, right_image = self.environment.robot.get_vision()
            feed_dict = {self.left_cam: [left_image], self.left_cam_before: [left_image_before]}
            ret = self.sess.run(fetches_store, feed_dict)
            ### Emulate reward computation (reward is (rec_err_i - rec_err_i+1) / 0.01)
            scale_reward__partial_new = np.array([ret["scale_reward__partial"][r][0] for r in self.ratios])
            scale_reward = scale_reward__partial - scale_reward__partial_new
            scale_reward__partial = scale_reward__partial_new
            total_reward__partial_new = ret["total_reward__partial"][0]
            total_reward = total_reward__partial - total_reward__partial_new
            total_reward__partial = total_reward__partial_new
            ###
            # Log Data
            ###
            object_distance = self.environment.screen.distance
            object_speed = self.environment.screen.tilt_pan_speed
            object_position = self.environment.screen.position
            eyes_position = self.environment.robot.position
            eyes_speed = self.environment.robot.speed
            if episode_number % 10 == 0:  # spare some disk space (90%)
                self.data_logger.store_data(ret, object_distance, object_speed, object_position, eyes_position, eyes_speed, iteration, episode_number, scale_reward, total_reward, self.task_index, self.episode_length, self.ratios)
            ###
            #
            ###
            states.append({r: ret["scales_inp"][r][0] for r in self.ratios})
            actions.append(ret["sampled_actions_indices"])
            # actions.append(ret["greedy_actions_indices"])
            self.environment.robot.set_action(self.actions_indices_to_values(ret["sampled_actions_indices"]))
            self.environment.step()
        self.buffer.incorporate((states, actions))
        return episode_number

    def flush_data(self, path):
        """Reformats the training data buffer and dumps it onto the hard drive
        This function must be called regularly.
        The frequency at which it is called can be specified in the command line with the option --flush-every
        See the help guide about data formats
        """
        self.data_logger.flush_data(self.task_index, path)
        self.pipe.send("{} flushed data on the hard drive".format(self.name))

    def define_actions_sets(self):
        """Defines the pan/tilt/vergence action sets
        At the moment, pan and tilt are comprised only of zeros"""
        n = self.n_actions_per_joint // 2
        # tilt
        # positive = [0.5, 0.9, 1.1, 1.3]
        # negative = [-0.5, -0.9, -1.1, -1.2]
        # self.action_set_tilt = np.concatenate([negative, [0], positive])
        #self.action_set_tilt = np.zeros(self.n_actions_per_joint)
        half_pixel_in_angle = 90 / 240 / 2
        mini = half_pixel_in_angle
        maxi = half_pixel_in_angle * 2 ** (n - 1)
        positive = np.logspace(np.log2(mini), np.log2(maxi), n, base=2)
        negative = -positive[::-1]
        self.action_set_tilt = np.concatenate([negative, [0], positive])
        # pan
        #self.action_set_pan = np.zeros(self.n_actions_per_joint)
        half_pixel_in_angle = 90 / 320 / 2
        mini = half_pixel_in_angle
        maxi = half_pixel_in_angle * 2 ** (n - 1)
        positive = np.logspace(np.log2(mini), np.log2(maxi), n, base=2)
        negative = -positive[::-1]
        #positive = [0.5, 0.9, 1.1, 1.3]
        #negative = [-0.5, -0.9, -1.1, -1.2]
        self.action_set_pan = np.concatenate([negative, [0], positive])
        # vergence
        self.action_set_vergence = np.zeros(self.n_actions_per_joint)

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
        """Generates a video, to be stored under path, concisting of n_episodes
        """
        fetches = self.greedy_actions_indices if not training else self.sampled_actions_indices
        rectangles = [(160 - 16 * r, 120 - 16 * r, 160 + 16 * r, 120 + 16 * r) for r in self.ratios]
        print("{} will store the video under {}".format(self.name, path))
        with get_writer(path, fps=25, format="mp4") as writer:
            for episode_number in range(n_episodes):
                print("{} episode {}/{}".format(self.name, episode_number + 1, n_episodes))
                self.environment.episode_reset()
                left_image, right_image = self.environment.robot.get_vision()
                for iteration in range(self.episode_length):
                    left_image_before, right_image_before = left_image, right_image
                    left_image, right_image = self.environment.robot.get_vision()
                    object_distance = self.environment.screen.distance
                    vergence_error = self.environment.robot.get_vergence_error(object_distance)
                    eyes_speed = self.environment.robot.speed
                    screen_speed = self.environment.screen.tilt_pan_speed
                    tilt_speed_error = eyes_speed[0] - screen_speed[0]
                    pan_speed_error = eyes_speed[1] - screen_speed[1]
                    frame = make_frame(left_image, right_image, object_distance, vergence_error, episode_number + 1, n_episodes, rectangles)
                    writer.append_data(frame)
                    if iteration == 0:
                        for i in range(24):
                            writer.append_data(frame)
                    feed_dict = {self.left_cam: [left_image], self.left_cam_before: [left_image_before]}
                    ret = self.sess.run(fetches, feed_dict)
                    self.environment.robot.set_action(self.actions_indices_to_values(ret))
                    self.environment.step()
                    print("vergence error: {:.4f}".format(vergence_error))
        self.pipe.send("{} going IDLE".format(self.name))