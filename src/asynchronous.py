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
        spec["ps"].append("localhost:{}".format(i + port))
    for i in range(n_workers):
        if "worker" not in spec:
            spec["worker"] = []
        spec["worker"].append("localhost:{}".format(i + port + n_parameter_servers))
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
        self.mlr, self.clr, self.alr = args.model_learning_rate, args.critic_learning_rate, args.actor_learning_rate
        self.softmax_temperature = args.softmax_temperature
        self.softmax_temperature_decay = args.softmax_temperature_decay
        self.entropy_reg = args.entropy_reg
        self.entropy_reg_decay = args.entropy_reg_decay
        self.discount_factor = args.discount_factor
        self.sequence_length = args.sequence_length
        self.update_per_episode = args.update_per_episode
        self.buffer_size = 20


class Worker:
    def __init__(self, cluster, task_index, pipe, logdir, simulator_port,
                 model_lr, critic_lr, actor_lr, discount_factor,
                 entropy_coef, entropy_coef_decay,
                 softmax_temperature, softmax_temperature_decay,
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
        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay
        self.softmax_temperature = softmax_temperature
        self.softmax_temperature_decay = softmax_temperature_decay
        self.sequence_length = sequence_length
        self.update_per_episode = update_per_episode
        self.model_lr = model_lr
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.buffer = Buffer(size=model_buffer_size)
        self.pipe = pipe
        self.logdir = logdir
        self.n_actions_per_joint = 9
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

    def define_autoencoder(self, ratio, filter_size=4, stride=2):
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
        self.scale_losses_patch[ratio] = tf.reduce_mean(sub ** 2, axis=3)
        self.scale_losses[ratio] = tf.reduce_mean(self.scale_losses_patch[ratio], axis=[1, 2])
        self.scale_rewards_patch[ratio] = -(self.scale_losses_patch[ratio] - 0.03) / 0.03
        self.scale_rewards[ratio] = -(self.scale_losses[ratio] - 0.03) / 0.03
        self.scale_loss[ratio] = tf.reduce_mean(self.scale_losses[ratio])
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

    def define_critic(self, ratio):
        ### PATCH LEVEL
        inp = tf.stop_gradient(self.scale_latent_conv[ratio])
        conv1 = tl.conv2d(inp, 20, 1, 1, "valid", activation_fn=lrelu)
        per_patch_critic = tl.conv2d(inp, 1, 1, 1, "valid", activation_fn=None)
        reward = self.scale_rewards_patch[ratio][1:]
        with tf.device(self.device):
            returns = tf_returns(reward, self.discount_factor, start=per_patch_critic[-1, :, :, 0][tf.newaxis], axis=0)
        self.scale_returns_patch[ratio] = returns
        self.scale_critic_losses_patch[ratio] = tf.reduce_mean((per_patch_critic[:-1, :, :, 0] - returns) ** 2, axis=1)
        self.scale_critic_loss_patch[ratio] = tf.reduce_mean(self.scale_critic_losses_patch[ratio])
        ### SCALE LEVEL
        size = np.prod(per_patch_critic.get_shape()[1:])
        inp = tf.concat([tf.stop_gradient(self.scale_latent[ratio]), tf.reshape(per_patch_critic, (-1, size))], axis=1)
        # inp = self.scale_latent[ratio]
        fc1 = tl.fully_connected(inp, 200, activation_fn=lrelu)
        # fc2 = tl.fully_connected(fc1, 200, activation_fn=lrelu)
        # fc3 = tl.fully_connected(fc2, 200, activation_fn=lrelu)
        # fc4 = tl.fully_connected(fc3, 1, activation_fn=None)
        fc4 = tl.fully_connected(fc1, 1, activation_fn=None)
        self.scale_critic_values[ratio] = fc4
        reward = self.scale_rewards[ratio][1:]
        with tf.device(self.device):
            returns = tf_returns(reward, self.discount_factor, start=fc4[-1][tf.newaxis], axis=0)
        self.scale_returns[ratio] = returns
        self.scale_critic_losses[ratio] = tf.reduce_mean((fc4[:-1] - returns) ** 2, axis=1)
        self.scale_critic_loss[ratio] = tf.reduce_mean(self.scale_critic_losses[ratio])
        self.scale_critic_loss_total[ratio] = self.scale_critic_loss[ratio] + self.scale_critic_loss_patch[ratio]

    def define_actor(self):
        inp = tf.stop_gradient(self.latent)
        # inp = self.latent
        self.picked_actions = {}
        self.logits = {}
        self.probs = {}
        self.distributions = {}
        self.sampled_actions_indices = {}
        self.greedy_actions_indices = {}
        self.greedy_actions_prob = {}
        self.log_prob_picked_actions = {}
        self.entropies = {}
        self.actors_losses = {}
        self.entropy_coef = tf.Variable(self.entropy_coef, trainable=False)
        self.softmax_temperature = tf.Variable(self.softmax_temperature, trainable=False)
        self.entropy_coef_update = self.entropy_coef.assign(self.entropy_coef * self.entropy_coef_decay)
        self.softmax_temperature_update = self.softmax_temperature.assign(self.softmax_temperature * self.softmax_temperature_decay)
        self.actors_targets = tf.stop_gradient(self.rewards[1:] - self.critic_values[:-1, 0])
        for joint_name in ["tilt", "pan", "vergence"]:
            fc1 = tl.fully_connected(inp, 200, activation_fn=tf.tanh)
            # fc2 = tl.fully_connected(fc1, 200, activation_fn=lrelu)
            # fc3 = tl.fully_connected(fc2, 200, activation_fn=lrelu)
            # fc4 = tl.fully_connected(fc3, self.n_actions_per_joint, activation_fn=lrelu)
            # fc4 = tl.fully_connected(fc1, self.n_actions_per_joint, activation_fn=tf.tanh)
            fc4 = tl.fully_connected(fc1, self.n_actions_per_joint, activation_fn=None)
            self.picked_actions[joint_name] = tf.placeholder(shape=(None, 1), dtype=tf.int32)
            self.logits[joint_name] = fc4 / self.softmax_temperature
            self.probs[joint_name] = tf.nn.softmax(self.logits[joint_name])
            self.distributions[joint_name] = tf.distributions.Categorical(probs=self.probs[joint_name])
            # self.distributions[joint_name] = tf.distributions.Categorical(logits=self.logits[joint_name])
            self.sampled_actions_indices[joint_name] = self.distributions[joint_name].sample()
            self.greedy_actions_indices[joint_name] = tf.argmax(self.logits[joint_name], axis=1)
            self.greedy_actions_prob[joint_name] =  \
                self.distributions[joint_name].prob(self.greedy_actions_indices[joint_name])
            self.log_prob_picked_actions[joint_name] = \
                self.distributions[joint_name].log_prob(self.picked_actions[joint_name][:, 0])[:-1]
            self.entropies[joint_name] = self.distributions[joint_name].entropy()[:-1]
            self.actors_losses[joint_name] = tf.reduce_mean(
                -self.log_prob_picked_actions[joint_name] * self.actors_targets -
                self.entropy_coef * self.entropies[joint_name])
        # self.actor_loss = \
        #     sum([self.actors_losses[joint_name] for joint_name in ["tilt", "pan", "vergence"]])
        self.actor_loss = sum(self.actors_losses.values())

    def define_networks(self):
        self.left_cam = tf.placeholder(shape=(None, 240, 320, 3), dtype=tf.uint8)
        self.right_cam = tf.placeholder(shape=(None, 240, 320, 3), dtype=tf.uint8)
        self.cams = tf.concat([self.left_cam, self.right_cam], axis=-1)  # (None, 240, 320, 6)
        ### autoencoder
        self.scales_inp = {}
        self.scale_latent = {}
        self.scale_latent_conv = {}
        self.scale_rec = {}
        self.scale_losses_patch = {}
        self.scale_rewards_patch = {}
        self.scale_losses = {}
        self.scale_rewards = {}
        self.scale_loss = {}
        self.scale_tensorboard_images = {}
        self.ratios = list(range(1, 9))
        for ratio in self.ratios:
            self.define_scale_inp(ratio)
            self.define_autoencoder(ratio, filter_size=4, stride=2)
        self.latent = tf.concat([self.scale_latent[r] for r in self.ratios], axis=1)
        self.autoencoder_loss = sum([self.scale_loss[r] for r in self.ratios])
        self.rewards = sum([self.scale_rewards[r] for r in self.ratios]) / len(self.ratios)
        ### critic
        self.scale_critic_losses_patch = {}
        self.scale_critic_loss_patch = {}
        self.scale_returns_patch = {}
        self.scale_critic_values = {}
        self.scale_critic_losses = {}
        self.scale_critic_loss = {}
        self.scale_returns = {}
        self.scale_critic_loss_total = {}
        for ratio in self.ratios:
            self.define_critic(ratio)
        inp = tf.concat([tf.stop_gradient(self.latent)] + [self.scale_critic_values[r] for r in self.ratios], axis=-1)
        # inp = tf.stop_gradient(self.latent)
        fc1 = tl.fully_connected(inp, 200, activation_fn=lrelu)
        fc2 = tl.fully_connected(fc1, 200, activation_fn=lrelu)
        self.critic_values = tl.fully_connected(fc2, 1, activation_fn=None)
        with tf.device(self.device):
            self.returns = tf_returns(self.rewards[1:], self.discount_factor, start=self.critic_values[-1][tf.newaxis], axis=0)
        self.critic_loss_global = tf.reduce_mean((self.critic_values[:-1] - self.returns) ** 2)
        self.critic_loss = sum([self.scale_critic_loss_total[r] for r in self.ratios]) + self.critic_loss_global
        ### actor
        self.define_actor()
        ### summaries
        summary_loss = tf.summary.scalar("/autoencoders/loss", self.autoencoder_loss)
        summary_critic_loss = tf.summary.scalar("/critic/loss", self.critic_loss)
        summary_scale_critic_loss = [tf.summary.scalar("/critic/ratio_{}".format(r), self.scale_critic_loss[r]) for r in self.ratios]
        summary_actor_loss = tf.summary.scalar("/actor/loss", self.actor_loss)
        summary_per_actor_loss = [tf.summary.scalar("/actor/loss_{}".format(jn), self.actors_losses[jn]) for jn in ["tilt", "pan", "vergence"]]
        summary_per_actor_logits = [tf.summary.scalar("/actor/logits_{}".format(jn), self.logits[jn][0, 4]) for jn in ["tilt", "pan", "vergence"]]
        summary_per_actor_entropy = [tf.summary.scalar("/actor/entropy_{}".format(jn), tf.reduce_mean(self.entropies[jn])) for jn in ["tilt", "pan", "vergence"]]
        summary_per_actor_max_prob = [tf.summary.scalar("/actor/max_prob_{}".format(jn), tf.reduce_mean(self.greedy_actions_prob[jn])) for jn in ["tilt", "pan", "vergence"]]
        summary_images = [tf.summary.image("ratio_{}".format(r), self.scale_tensorboard_images[r], max_outputs=1) for r in self.ratios]
        self.summary = tf.summary.merge(
            [summary_loss, summary_critic_loss, summary_actor_loss] +
            summary_images +
            summary_scale_critic_loss +
            summary_per_actor_loss +
            summary_per_actor_logits +
            summary_per_actor_entropy +
            summary_per_actor_max_prob)
        self.train_step = tf.Variable(0, dtype=tf.int32)
        self.train_step_inc = self.train_step.assign_add(1, use_locking=True)
        self.optimizer_autoencoder = tf.train.AdamOptimizer(self.model_lr, use_locking=True)
        self.optimizer_critic = tf.train.AdamOptimizer(self.critic_lr, use_locking=True)
        self.optimizer_actor = tf.train.AdamOptimizer(self.actor_lr, use_locking=True)
        # self.optimizer_actor = tf.train.GradientDescentOptimizer(self.actor_lr, use_locking=True)
        # self.optimizer_actor = tf.train.RMSPropOptimizer(self.actor_lr / 100, decay=0.99, use_locking=True)
        # self.train_op_autoencoder = self.optimizer_autoencoder.minimize(self.autoencoder_loss + 10.0 * self.actor_loss + .025 * self.critic_loss)
        self.train_op_autoencoder = self.optimizer_autoencoder.minimize(self.autoencoder_loss)
        self.train_op_critic = self.optimizer_critic.minimize(self.critic_loss)
        self.train_op_actor = self.optimizer_actor.minimize(self.actor_loss)
        self.train_op = tf.group([self.train_op_autoencoder, self.train_op_critic, self.train_op_actor])

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

    def restore(self, path):
        self.saver.restore(self.sess, os.path.normpath(path + "/network.ckpt"))
        self.pipe.send("{} variables restored from {}".format(self.name, path))

    def play_trajectory(self, greedy=False):
        fetches = self.greedy_actions_indices if greedy else self.sampled_actions_indices
        self.reset_env()
        mean = 0
        for iteration in range(self.sequence_length):
            left_image, right_image = self.robot.receiveImages()
            feed_dict = {self.left_cam: [left_image], self.right_cam: [right_image]}
            ret = self.sess.run(fetches, feed_dict)
            self.apply_action([ret[jn] for jn in ["tilt", "pan", "vergence"]])
            object_distance = self.screen.distance
            vergence = self.robot.getEyePositions()[-1]
            vergence_error = - np.degrees(np.arctan2(RESSOURCES.Y_EYES_DISTANCE, 2 * object_distance)) * 2 - vergence
            mean += np.abs(vergence_error)
            print("vergence error: {:.4f}".format(vergence_error), end="\n")
        # print("")
        print("mean abs vergence error: {:.4f}".format(mean / self.sequence_length))

    def get_trajectory(self):
        states = []
        actions = []
        rewards = []
        fetches = {
            "scales_inp": self.scales_inp,
            "greedy_actions_indices": self.greedy_actions_indices
        }
        fetches_store = {
            "scales_inp": self.scales_inp,
            "scale_reward": self.scale_rewards,
            "total_reward": self.rewards,
            "greedy_actions_indices": self.greedy_actions_indices,
            "sampled_actions_indices": self.sampled_actions_indices,
            "scale_critic_values": self.scale_critic_values,
            "critic_values": self.critic_values
        }
        self.reset_env()
        trajectory = self.sess.run(self.trajectory_count)
        for iteration in range(self.sequence_length):
            left_image, right_image = self.robot.receiveImages()
            feed_dict = {self.left_cam: [left_image], self.right_cam: [right_image]}
            if iteration < self.sequence_length:
                ret = self.sess.run(fetches_store, feed_dict)
                object_distance = self.screen.distance
                eyes_position = self.robot.getEyePositions()
                eyes_speed = (self.tilt_delta, self.pan_delta)
                self.store_data(ret, object_distance, eyes_position, eyes_speed, iteration, trajectory)
            else:
                ret = self.sess.run(fetches, feed_dict)
            states.append({r: ret["scales_inp"][r][0] for r in self.ratios})
            actions.append(ret["sampled_actions_indices"])
            # actions.append(ret["greedy_actions_indices"])
            self.apply_action([ret["sampled_actions_indices"][jn] for jn in ["tilt", "pan", "vergence"]])
        self.buffer.incorporate((states, actions))
        return states, actions

    def store_data(self, ret, object_distance, eyes_position, eyes_speed, iteration, trajectory):
        data = {
            "worker": np.squeeze(np.array(self.task_index)),
            "trajectory": np.squeeze(np.array(trajectory)),
            "iteration": np.squeeze(np.array(iteration)),
            "global_iteration": np.squeeze(np.array(trajectory * self.sequence_length + iteration)),
            "total_reward": np.squeeze(np.array(ret["total_reward"])),
            "scale_rewards": np.squeeze(np.array([ret["scale_reward"][ratio] for ratio in self.ratios])),
            "greedy_actions_indices": np.squeeze(np.array([ret["greedy_actions_indices"][k] for k in ["tilt", "pan", "vergence"]])),
            "sampled_actions_indices": np.squeeze(np.array([ret["sampled_actions_indices"][k] for k in ["tilt", "pan", "vergence"]])),
            "scale_critic_values": np.squeeze(np.array([ret["scale_critic_values"][ratio] for ratio in self.ratios])),
            "critic_values": np.squeeze(np.array(ret["critic_values"])),
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
                "ops": [self.train_op, self.entropy_coef_update, self.softmax_temperature_update],
                "summary": self.summary,
                "step": self.train_step_inc,
                "autoencoder_loss": self.autoencoder_loss,
                "softmax_temperature": self.softmax_temperature,
                "entropy_coef": self.entropy_coef
            }
            feed_dict = {self.scales_inp[r]: [s[r] for s in states] for r in self.ratios}
            for jn in ["tilt", "pan", "vergence"]:
                feed_dict[self.picked_actions[jn]] = [a[jn] for a in actions]
            ret = self.sess.run(fetches, feed_dict=feed_dict)
            if self._update_number % 100 == 0:
                self.summary_writer.add_summary(ret["summary"], global_step=ret["step"])
            print("{} update {}\tautoencoder loss:  {:6.3f}\t\ttemperature {:.2f}\tentropy {:.2f}".format(
                self.name,
                ret["step"],
                ret["autoencoder_loss"],
                ret["softmax_temperature"],
                ret["entropy_coef"]))
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
        self.confdir = self.experiment_dir + "/conf"
        if not os.path.exists(self.experiment_dir) or os.listdir(self.experiment_dir) == ['log']:
            os.makedirs(self.experiment_dir, exist_ok=True)
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.videodir)
            os.makedirs(self.datadir)
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
            self.worker_conf.alr,
            self.worker_conf.discount_factor,
            self.worker_conf.entropy_reg,
            self.worker_conf.entropy_reg_decay,
            self.worker_conf.softmax_temperature,
            self.worker_conf.softmax_temperature_decay,
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

    # def save_video(self, name, n_sequences, training=True):
    #     path = self.videodir + "/{}.mp4".format(name)
    #     self.here_pipes[0].send(("run_video", path, n_sequences, training))
    #     print(self.here_pipes[0].recv())

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
default_alr = 1e-4
default_er = 2e-1


def make_experiment_path(date=None, mlr=None, clr=None, alr=None, er=None, description=None):
    date = date if date else time.strftime("%Y_%m_%d-%H:%M:%S", time.localtime())
    mlr = mlr if mlr else default_mlr
    clr = clr if clr else default_clr
    alr = alr if alr else default_alr
    er = er if er else default_er
    description = ("__" + description) if description else ""
    experiment_dir = "../experiments/{}_mlr{:.2e}_clr{:.2e}_alr{:.2e}_entropy{:.2e}{}".format(
        date, mlr, clr, alr, er, description)
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
        default=2
    )

    parser.add_argument(
        '-nw', '--n-workers',
        type=int,
        help="Number of workers.",
        default=16
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
        default=5000
    )

    parser.add_argument(
        '-nt', '--n-trajectories',
        type=int,
        help="Number of trajectories to be simulated.",
        default=100000
    )

    parser.add_argument(
        '-sl', '--sequence-length',
        type=int,
        help="Length of an episode.",
        default=40
    )

    parser.add_argument(
        '-u', '--update-per-episode',
        type=int,
        help="Number of updates per gathered trajectory.",
        default=5
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
        '-alr', '--actor-learning-rate',
        type=float,
        default=default_alr,
        help="actor learning rate."
    )

    parser.add_argument(
        '-df', '--discount-factor',
        type=float,
        default=0,
        help="Discount factor."
    )

    parser.add_argument(
        '-er', '--entropy-reg',
        type=float,
        default=default_er,
        help="Entropy regularizer."
    )

    parser.add_argument(
        '-erd', '--entropy-reg-decay',
        type=float,
        default=1.0,
        help="Entropy regularizer decay."
    )

    parser.add_argument(
        '-st', '--softmax-temperature',
        type=float,
        default=10.0,
        help="Temperature of the softmax sampling."
    )

    parser.add_argument(
        '-std', '--softmax-temperature-decay',
        type=float,
        default=1.0,
        help="Decay for the temperature of the softmax sampling."
    )

    args = parser.parse_args()

    if not args.experiment_path:
        experiment_dir = make_experiment_path(
            mlr=args.model_learning_rate,
            clr=args.critic_learning_rate,
            alr=args.actor_learning_rate,
            er=args.entropy_reg,
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
