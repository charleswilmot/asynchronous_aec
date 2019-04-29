from tempfile import TemporaryDirectory
import png
from replay_buffer import Buffer
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


# def lrelu(x):
#     alpha = 0.2
#     return tf.nn.relu(x) * (1 - alpha) + x * alpha


def lrelu(x):
    return tf.tanh(x)


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


class Worker:
    def __init__(self, cluster, task_index, pipe, logdir,
                 model_lr, critic_lr, actor_lr,
                 discount_factor, entropy_coef,
                 sequence_length,
                 model_buffer_size):
        self.task_index = task_index
        self.cluster = cluster
        self._n_workers = self.cluster.num_tasks("worker") - 1
        self.server = tf.train.Server(cluster, "worker", task_index)
        self.name = "/job:worker/task:{}".format(task_index)
        self.device = tf.train.replica_device_setter(worker_device=self.name, cluster=cluster)
        self.discount_factor = discount_factor
        self.entropy_coef = entropy_coef
        self.sequence_length = sequence_length
        self.model_lr = model_lr
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.buffer = Buffer(size=model_buffer_size)
        self.pipe = pipe
        self.logdir = logdir
        self.define_networks()
        self.define_actions_sets()
        self.end_episode_data = []
        self._flush_id = 0
        graph = tf.get_default_graph() if task_index == 0 else None
        self.summary_writer = tf.summary.FileWriter(self.logdir + "/worker{}".format(task_index), graph=graph)
        self.saver = tf.train.Saver()
        self.sess = tf.Session(target=self.server.target)
        if task_index == 0 and len(self.sess.run(tf.report_uninitialized_variables())) > 0:  # todo: can be done in Experiment
            self.sess.run(tf.global_variables_initializer())
            print("{}  variables initialized".format(self.name))
        lock = filelock.FileLock("/home/wilmot/Documents/code/asynchronous_aec/experiments/lock_universe")
        lock.acquire()
        self.universe = vb.Universe(task_index == -1)
        lock.release()
        self.robot = self.universe.robot
        self.textures = get_textures("/home/aecgroup/aecdata/Textures/mcgillManMade_600x600_bmp_selection/")
        self.screen = vb.ConstantSpeedScreen(self.universe.sim, (0.5, 5), 0.0, 0, self.textures)
        self.screen.set_positions(-2, 0, 0)

    def define_networks(self):
        self.left_cam = tf.placeholder(shape=(None, 240, 320, 3), dtype=tf.uint8)
        self.right_cam = tf.placeholder(shape=(None, 240, 320, 3), dtype=tf.uint8)
        cams = tf.concat([self.left_cam, self.right_cam], axis=-1)  # (None, 240, 320, 6)
        crop_side_length = 32
        coarse_scale_ratio = 4
        height_slice = slice(
            (240 - crop_side_length) // 2,
            (240 + crop_side_length) // 2)
        width_slice = slice(
            (320 - crop_side_length) // 2,
            (320 + crop_side_length) // 2)
        fine_scale_uint8 = cams[:, height_slice, width_slice, :]
        self.fine_scale = tf.placeholder_with_default(
            tf.cast(fine_scale_uint8, tf.float32) / 127.5 - 1,
            shape=fine_scale_uint8.get_shape())
        height_slice = slice(
            (240 - crop_side_length * coarse_scale_ratio) // 2,
            (240 + crop_side_length * coarse_scale_ratio) // 2)
        width_slice = slice(
            (320 - crop_side_length * coarse_scale_ratio) // 2,
            (320 + crop_side_length * coarse_scale_ratio) // 2)
        coarse_scale_uint8 = cams[:, height_slice, width_slice, :]
        coarse_scale_uint8 = tf.image.resize_bilinear(coarse_scale_uint8, [crop_side_length, crop_side_length])
        self.coarse_scale = tf.placeholder_with_default(
            tf.cast(coarse_scale_uint8, tf.float32) / 127.5 - 1,
            shape=coarse_scale_uint8.get_shape())

        with tf.variable_scope("fine_scale_encoder"):
            conv1_fine = tl.conv2d(self.fine_scale + 0, 128, 3, 2, "same", activation_fn=lrelu)  # 16, 16, 16
            conv2_fine = tl.conv2d(conv1_fine, 128, 3, 2, "same", activation_fn=lrelu)       # 32,  8,  8
        with tf.variable_scope("coarse_scale_encoder"):
            conv1_coarse = tl.conv2d(self.coarse_scale + 0, 128, 3, 2, "same", activation_fn=lrelu)  # 16, 16, 16
            conv2_coarse = tl.conv2d(conv1_coarse, 128, 3, 2, "same", activation_fn=lrelu)       # 32,  8,  8
        with tf.variable_scope("joint_encoder_decoder"):
            batch_size = tf.shape(conv1_fine)[0]
            flat_fine = tf.reshape(conv2_fine, (batch_size, -1))
            flat_coarse = tf.reshape(conv2_coarse, (batch_size, -1))
            combined = tf.concat([flat_fine, flat_coarse], axis=-1)
            latent_size = int(np.prod(conv2_fine.get_shape()[1:]) + np.prod(conv2_coarse.get_shape()[1:]))
            combined = tf.reshape(combined, (-1, latent_size))
            self.latent = tl.fully_connected(combined, latent_size, activation_fn=None)
            latent_fine = tf.reshape(self.latent[:, :latent_size // 2], [-1] + list(conv2_fine.get_shape())[1:])
            latent_coarse = tf.reshape(self.latent[:, latent_size // 2:], [-1] + list(conv2_coarse.get_shape())[1:])
        with tf.variable_scope("fine_scale_decoder"):
            deconv1_fine = tl.conv2d_transpose(latent_fine, 128, 3, 2, "same", activation_fn=lrelu)
            deconv2_fine = tl.conv2d_transpose(deconv1_fine, 6, 3, 2, "same", activation_fn=None)
        with tf.variable_scope("coarse_scale_decoder"):
            deconv1_coarse = tl.conv2d_transpose(latent_coarse, 128, 3, 2, "same", activation_fn=lrelu)
            deconv2_coarse = tl.conv2d_transpose(deconv1_coarse, 6, 3, 2, "same", activation_fn=None)
        with tf.variable_scope("model_auxiliary_ops"):
            self.error_fine = tf.reduce_mean((self.fine_scale - deconv2_fine) ** 2, axis=[1, 2, 3])
            self.error_coarse = tf.reduce_mean((self.coarse_scale - deconv2_coarse) ** 2, axis=[1, 2, 3])
            self.batch_error_fine = tf.reduce_mean((self.fine_scale - deconv2_fine) ** 2)
            self.batch_error_coarse = tf.reduce_mean((self.coarse_scale - deconv2_coarse) ** 2)
            self.batch_error = self.batch_error_fine + self.batch_error_coarse
            self.error = self.error_fine + self.error_coarse

        with tf.variable_scope("actor"):
            self.n_actions_per_joint = 9
            n_actions_per_joint = self.n_actions_per_joint
            fc1_actor = tl.fully_connected(tf.stop_gradient(self.latent), 200, activation_fn=lrelu)
            fc2_actor = tl.fully_connected(fc1_actor, n_actions_per_joint * 3, activation_fn=None)
            self.tilt_actor = fc2_actor[:, n_actions_per_joint * 0: n_actions_per_joint * 1]
            self.pan_actor = fc2_actor[:, n_actions_per_joint * 1: n_actions_per_joint * 2]
            self.verg_actor = fc2_actor[:, n_actions_per_joint * 2: n_actions_per_joint * 3]
            # distributions
            # tilt_dist = tf.distributions.Categorical(probs=tf.nn.softmax(self.tilt_actor))
            # pan_dist = tf.distributions.Categorical(probs=tf.nn.softmax(self.pan_actor))
            # verg_dist = tf.distributions.Categorical(probs=tf.nn.softmax(self.verg_actor))
            tilt_dist = tf.distributions.Categorical(logits=self.tilt_actor)
            pan_dist = tf.distributions.Categorical(logits=self.pan_actor)
            verg_dist = tf.distributions.Categorical(logits=self.verg_actor)
            # action indices
            self.action_indices = tf.placeholder(shape=(None, 3), dtype=tf.int32)
            tilt_action_indices = self.action_indices[:, 0]
            pan_action_indices = self.action_indices[:, 1]
            verg_action_indices = self.action_indices[:, 2]
            # action log probabilities
            tilt_log_picked_probs = tilt_dist.log_prob(tilt_action_indices)
            pan_log_picked_probs = pan_dist.log_prob(pan_action_indices)
            verg_log_picked_probs = verg_dist.log_prob(verg_action_indices)
            # action probabilities
            tilt_picked_probs = tilt_dist.prob(tilt_action_indices)
            pan_picked_probs = pan_dist.prob(pan_action_indices)
            verg_picked_probs = verg_dist.prob(verg_action_indices)
            # greedy action indices
            tilt_greedy_actions_indices = tf.argmax(self.tilt_actor, axis=-1)
            pan_greedy_actions_indices = tf.argmax(self.pan_actor, axis=-1)
            verg_greedy_actions_indices = tf.argmax(self.verg_actor, axis=-1)
            # samples
            self.tilt_sample = tilt_dist.sample()
            self.pan_sample = pan_dist.sample()
            self.verg_sample = verg_dist.sample()
            # entropies
            tilt_entropy = tilt_dist.entropy()
            pan_entropy = pan_dist.entropy()
            verg_entropy = verg_dist.entropy()

        with tf.variable_scope("critic"):
            fc1_critic = tl.fully_connected(tf.stop_gradient(self.latent), 200, activation_fn=lrelu)
            fc2_critic = tl.fully_connected(fc1_critic, 2, activation_fn=None)
            self.critic_values = fc2_critic
            critic_value = tf.reduce_sum(fc2_critic, axis=1)

        with tf.variable_scope("rl_auxiliary_ops"):
            self.train_step = tf.Variable(0, dtype=tf.int32)
            self.train_step_inc = self.train_step.assign_add(1)
            # critic
            mean = 0.055
            std = 0.055
            self.rewards = tf.stop_gradient((mean - tf.stack([self.error_fine, self.error_coarse], axis=1)) / std)
            # self.rewards = tf.stop_gradient(normalize(tf.stack([-self.error_fine, -self.error_coarse], axis=1), 0.999))
            critic_diff = fc2_critic - self.rewards
            self.critic_loss = tf.reduce_mean(critic_diff ** 2)
            critic_quality = tf.clip_by_value(tf.reduce_mean(critic_diff / (tf.reduce_mean(self.rewards, axis=1, keepdims=True) - self.rewards)), 0, 20)
            # actor
            targets = tf.reduce_sum(self.rewards, axis=1) - critic_value
            targets = tf.stop_gradient(targets)
            tilt_actor_loss = -tf.maximum(tilt_log_picked_probs, -4) * targets  # - self.entropy_coef * tilt_entropy
            pan_actor_loss = -tf.maximum(pan_log_picked_probs, -4) * targets - self.entropy_coef * pan_entropy
            verg_actor_loss = -tf.maximum(verg_log_picked_probs, -4) * targets - self.entropy_coef * verg_entropy
            self.actor_loss = tf.reduce_mean(verg_actor_loss)  # tilt_actor_loss  # + pan_actor_loss + verg_actor_loss
            self.optimizer = tf.train.GradientDescentOptimizer(self.model_lr)
            self.train_op = self.optimizer.minimize(self.batch_error + 5e-4 * self.critic_loss)  # + 1e-0 * self.actor_loss)

        summary_error_fine = tf.summary.scalar("/model/error_fine", self.error_fine[0])
        summary_error_coarse = tf.summary.scalar("/model/error_coarse", self.error_coarse[0])
        summary_error_total = tf.summary.scalar("/model/error_total", self.error[0])
        summary_error_batch = tf.summary.scalar("/model/error_batch", self.batch_error)
        summary_error_critic = tf.summary.scalar("/rl/critic_loss", self.critic_loss)
        summary_quality_critic = tf.summary.scalar("/rl/critic_quality", critic_quality)
        summary_error_actor = tf.summary.scalar("/rl/actor_loss", self.actor_loss)
        summary_reward_fine = tf.summary.scalar("/rl/reward_fine", self.rewards[0, 0])
        summary_reward_coarse = tf.summary.scalar("/rl/reward_coarse", self.rewards[0, 1])
        fine_scale_image_left = tf.concat([self.fine_scale[:, :, :, :3], deconv2_fine[:, :, :, :3]], axis=2)
        fine_scale_image_right = tf.concat([self.fine_scale[:, :, :, 3:], deconv2_fine[:, :, :, 3:]], axis=2)
        fine_scale_image = tf.concat([fine_scale_image_left, fine_scale_image_right], axis=1)
        summary_fine_reconstruction = tf.summary.image("fine_scale", fine_scale_image, max_outputs=1)
        coarse_scale_image_left = tf.concat([self.coarse_scale[:, :, :, :3], deconv2_coarse[:, :, :, :3]], axis=2)
        coarse_scale_image_right = tf.concat([self.coarse_scale[:, :, :, 3:], deconv2_coarse[:, :, :, 3:]], axis=2)
        coarse_scale_image = tf.concat([coarse_scale_image_left, coarse_scale_image_right], axis=1)
        summary_coarse_reconstruction = tf.summary.image("coarse_scale", coarse_scale_image, max_outputs=1)
        summary_tilt_entropy = tf.summary.scalar("/rl/tilt_entropy", tilt_entropy[0])
        summary_pan_entropy = tf.summary.scalar("/rl/pan_entropy", pan_entropy[0])
        summary_verg_entropy = tf.summary.scalar("/rl/verg_entropy", verg_entropy[0])
        # self.model_summary = tf.summary.scalar("/dummy", tf.reduce_mean(self.error_coarse))
        self.model_summary = tf.summary.merge([
            summary_error_fine,
            summary_error_coarse,
            summary_error_total,
            summary_error_batch,
            summary_error_critic,
            summary_quality_critic,
            summary_error_actor,
            summary_reward_fine,
            summary_reward_coarse,
            summary_fine_reconstruction,
            summary_coarse_reconstruction,
            summary_tilt_entropy,
            summary_pan_entropy,
            summary_verg_entropy
        ])

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

    def get_trajectory(self):
        states = []
        actions = []
        rewards = []
        fetches = {
            "fine_scale": self.fine_scale,
            "coarse_scale": self.coarse_scale,
            "reward": self.rewards,
            "actions": [self.tilt_sample, self.pan_sample, self.verg_sample]
        }
        fetches_store = {
            "train_step": self.train_step,
            "fine_scale": self.fine_scale,
            "coarse_scale": self.coarse_scale,
            "reward": self.rewards,
            "actions": [self.tilt_sample, self.pan_sample, self.verg_sample],
            "critic_values": self.critic_values
        }
        self.reset_env()
        for _ in range(self.sequence_length):
            left_image, right_image = self.robot.receiveImages()
            feed_dict = {self.left_cam: [left_image], self.right_cam: [right_image]}
            if _ == self.sequence_length - 1:
                ret = self.sess.run(fetches_store, feed_dict)
                object_distance = self.screen.distance
                eyes_position = self.robot.getEyePositions()
                eyes_speed = (self.tilt_delta, self.pan_delta)
                self.store_data(ret, object_distance, eyes_position, eyes_speed)
            else:
                ret = self.sess.run(fetches, feed_dict)
            states.append((ret["fine_scale"][0], ret["coarse_scale"][0]))
            actions.append([a[0] for a in ret["actions"]])
            rewards.append(ret["reward"])
            self.apply_action(ret["actions"])
        self.buffer.incorporate_multiple(zip(states, actions, rewards))
        return states, actions, rewards

    def store_data(self, ret, object_distance, eyes_position, eyes_speed):
        data = {
            "train_step": ret["train_step"],
            "rewards": ret["reward"],
            "actions": ret["actions"],
            "critic_values": ret["critic_values"],
            "object_distance": object_distance,
            "eyes_position": eyes_position,
            "eyes_speed": eyes_speed
        }
        self.end_episode_data.append(data)

    def flush_data(self, path):
        with open(path + "/worker_{:04d}_flush_{:04d}.pkl".format(self.task_index, self._flush_id), "wb") as f:
            pickle.dump(self.end_episode_data, f)
        self.end_episode_data = []
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
        verg_new_pos = np.clip(verg_pos + self.action_set_vergence[verg], -8, 0)
        self.robot.setEyePositions((tilt_new_pos, pan_new_pos, verg_new_pos))
        self.screen.iteration_init()
        self.universe.sim.stepSimulation()

    def reset_env(self):
        self.screen.episode_init()
        # random_distance = np.random.uniform(low=0.5, high=5)
        random_distance = self.screen.distance
        random_vergence = -np.arctan(RESSOURCES.Y_EYES_DISTANCE / (2 * random_distance)) * 360 / np.pi
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
        mini = 0.28
        maxi = 0.28 * 2 ** (n - 1)
        positive = np.logspace(np.log2(mini), np.log2(maxi), n, base=2)
        negative = -positive[::-1]
        self.action_set_vergence = np.concatenate([negative, [0], positive])

    def train(self):
        # states, actions, rewards = self.get_trajectory()
        self.get_trajectory()
        for _ in range(20):
            transitions = self.buffer.batch(self.sequence_length)
            states = [s for s, a, r in transitions]
            actions = [a for s, a, r in transitions]
            rewards = [r for s, a, r in transitions]
            fetches = {
                "ops": [self.train_op],
                "summary": self.model_summary,
                "step": self.train_step_inc,
                "errors": [self.batch_error_fine, self.batch_error_coarse, self.critic_loss, self.actor_loss]
            }
            feed_dict = {
                self.fine_scale: [a for a, b in states],
                self.coarse_scale: [b for a, b in states],
                self.action_indices: actions
            }
            ret = self.sess.run(fetches, feed_dict=feed_dict)
            self.summary_writer.add_summary(ret["summary"], global_step=ret["step"])
            # print(ret["errors"])
            print("{} episode {}\tfine  {:6.3f} coarse  {:6.3f} critic  {:6.3f} actor  {:6.3f}".format(self.name, ret["step"], ret["errors"][0], ret["errors"][1], ret["errors"][2], ret["errors"][3]))
        return ret["step"]

    def start_training(self, n_updates):
        step = self.sess.run(self.train_step)
        n_updates += step
        while step < n_updates - self._n_workers:
            step = self.train()
        self.pipe.send("{} going IDLE".format(self.name))

    # def rewards_to_return(self, rewards, prev_return=0):
    #     returns = np.zeros_like(rewards)
    #     for i in range(len(rewards) - 1, -1, -1):
    #         r = rewards[i]
    #         prev_return = r + self.discount_factor * prev_return
    #         returns[i] = prev_return
    #     return returns


class Experiment:
    def __init__(self, n_parameter_servers, n_workers, experiment_dir):
        lock = filelock.FileLock("/home/wilmot/Documents/code/asynchronous_aec/experiments/lock")
        lock.acquire()
        self.n_parameter_servers = n_parameter_servers
        self.n_workers = n_workers
        self.experiment_dir = experiment_dir
        self.mktree()
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
        os.mkdir(self.experiment_dir)
        os.mkdir(self.logdir)
        os.mkdir(self.videodir)
        os.mkdir(self.datadir)
        os.mkdir(self.checkpointsdir)

    def parameter_server_func(self, task_index):
        server = tf.train.Server(self.cluster, "ps", task_index)
        server.join()

    def worker_func(self, task_index):
        np.random.seed(task_index)
        worker = Worker(self.cluster, task_index, self.there_pipes[task_index], self.logdir, 2e-1, 0, 0, 0, 5e-3, 40, 40 * 500)
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



if __name__ == "__main__":
    n_parameter_servers = 4
    n_workers = 28
    experiment_dir = "../experiments/tmp2/"

    with Experiment(n_parameter_servers, n_workers, experiment_dir) as exp:
        # exp.start_tensorboard()
        for i in range(100):
            exp.asynchronously_train(1000)
            exp.flush_data()
