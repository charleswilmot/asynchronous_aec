import logging
import pickle
import socket
import sys
import os
import multiprocessing
import tensorflow as tf
import numpy as np
from algorithm.worker import Worker
import time
import subprocess
from itertools import cycle, islice
from helper.proper_display import ProperDisplay


def repeatlist(it, count):
    return islice(cycle(it), count)


def _chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def chunks(l, n):
    return list(_chunks(l, n))


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