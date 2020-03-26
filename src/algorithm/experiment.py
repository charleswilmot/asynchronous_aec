import logging
import pickle
import socket
import sys
import os
import multiprocessing
import tensorflow as tf
import numpy as np
from algorithm.worker import Worker
from helper.generate_test_conf import TestConf
import time
import subprocess
from itertools import cycle, islice


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


def collect_summaries(queue, path):
    total_time_getting = 0
    total_time_writing = 0
    last_time_printing = time.time()
    with tf.summary.FileWriter(path) as writer:
        while True:
            t0 = time.time()
            summary, global_step = queue.get()
            t1 = time.time()
            writer.add_summary(summary, global_step=global_step)
            t2 = time.time()
            total_time_getting += t1 - t0
            total_time_writing += t2 - t1
            if t2 - last_time_printing > 120:
                total = total_time_getting + total_time_writing
                # print("SUMMARY COLLECTOR: {:.2f}% getting, {:.2f}% writing. Size: {}".format(
                #     100 * total_time_getting / total, 100 * total_time_writing / total, queue.qsize())
                # )
                last_time_printing = t2


def collect_training_data(queue, path):
    total_time_getting = 0
    total_time_writing = 0
    last_time_printing = time.time()
    with open(path, "wb") as f:
        data = queue.get()
        serialized_dtype = pickle.dumps(data.dtype)
        f.write(np.int32(len(serialized_dtype)))  # 4 bytes for serialized_dtype size
        f.write(serialized_dtype)                 # n bytes for serialized_dtype
        episode_length = data.shape[0]
        f.write(np.int32(episode_length))         # 4 bytes for episode_length
        f.write(data.tobytes())
        while True:
            t0 = time.time()
            data = queue.get()
            t1 = time.time()
            f.write(data.tobytes())
            t2 = time.time()
            total_time_getting += t1 - t0
            total_time_writing += t2 - t1
            if t2 - last_time_printing > 120:
                total = total_time_getting + total_time_writing
                # print("TRAINING DATA COLLECTOR: {:.2f}% getting, {:.2f}% writing. Size: {}".format(
                #     100 * total_time_getting / total, 100 * total_time_writing / total, queue.qsize())
                # )
                last_time_printing = t2


class Experiment:
    """An experiment object allows to control clients / workers that train a model
    It starts one process per worker, and one process per parameter server
    It also constructs the filesystem tree for storing all results / data"""

    def __init__(self, n_parameter_servers, n_workers, experiment_dir, worker_conf, test_conf_path=None, worker0_display=False):
        self.n_parameter_servers = n_parameter_servers
        self.n_workers = n_workers
        self.experiment_dir = experiment_dir
        self.worker_conf = worker_conf
        self.test_conf_path = test_conf_path
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
        self.summary_queue = multiprocessing.Queue(maxsize=1000)
        self.summary_collector_process = multiprocessing.Process(
            target=collect_summaries,
            args=(self.summary_queue, self.logdir),
            daemon=True
        )
        self.training_data_queue = multiprocessing.Queue(maxsize=1000)
        self.training_data_collector_process = multiprocessing.Process(
            target=collect_training_data,
            args=(self.training_data_queue, self.datadir + "/training.data"),
            daemon=True
        )
        self.testing_data_queue = multiprocessing.Queue(maxsize=10000)
        self.test_cases_queue = multiprocessing.Queue(maxsize=10000)
        ### start all processes ###
        all_processes = self.parameter_servers_processes + self.workers_processes + [self.summary_collector_process, self.training_data_collector_process]
        for p in all_processes:
            p.start()
        print("EXPERIMENT: all processes started. Waiting for answer...")
        for p in self.here_pipes:
            print(p.recv())
        print("EXPERIMENT: all processes started. Waiting for answer... received")
        print("EXPERIMENT: loading test_conf from hard drive")
        self.test_conf = None if test_conf_path is None else TestConf.load(test_conf_path)
        print("EXPERIMENT: loading test_conf from hard drive ... done")

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
        pipe_and_queues = {
            "pipe": self.there_pipes[task_index],
            "summary_queue": self.summary_queue,
            "training_data_queue": self.training_data_queue,
            "testing_data_queue": self.testing_data_queue,
            "test_cases_queue": self.test_cases_queue
        }
        worker = Worker(self.cluster, task_index, pipe_and_queues, self.logdir, self.ports[task_index], self.worker_conf, self.worker0_display)
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

    def test(self, chunks_size=30, outpath=None):
        # policy independent test cases:
        print("EXPERIMENT: policy independent testing")
        # send
        list_of_test_cases = self.test_conf.data["test_cases_policy_independent"]
        policy_independent_inputs = self.test_conf.policy_independent_inputs
        n_sent = 0
        for index in range(0, len(list_of_test_cases), chunks_size):
            self.test_cases_queue.put(
                (list_of_test_cases[index:index + chunks_size], policy_independent_inputs[index:index + chunks_size])
            )
            n_sent += 1
        for pipe in self.here_pipes:
            pipe.send(("test_fast",))
        # receive
        n_received = 0
        res = []
        while n_received != n_sent:
            test_cases_chunk, test_data_chunk = self.testing_data_queue.get()
            n_received += 1
            for item in zip(test_cases_chunk, test_data_chunk):
                res.append(item)
        for pipe in self.here_pipes:
            print(pipe.recv())  # make sure all workers are done
        print("EXPERIMENT: policy independent testing ... done")

        # policy dependent test cases:
        print("EXPERIMENT: policy dependent testing")
        # send
        list_of_test_cases = self.test_conf.data["test_cases_policy_dependent"]
        n_test_cases = len(list_of_test_cases)
        for test_case in list_of_test_cases:
            self.test_cases_queue.put(test_case)
        for pipe in self.here_pipes:
            pipe.send(("test",))
        # receive
        n_received = 0
        while n_received < n_test_cases:
            test_case, test_data = self.testing_data_queue.get()
            n_received += 1
            res.append((test_case, test_data))
            if n_received % 100 == 0:
                print("EXPERIMENT: tested {: 5d} out of {: 5d} cases".format(n_received, n_test_cases))
        for pipe in self.here_pipes:
            print(pipe.recv())  # make sure all workers are done
        # get the current iteration...
        current_episode_count = self.get_current_episode_count()
        if current_episode_count == 0:
            current_episode_count = np.random.randint(0, 100000000)
        # store the data
        path = self.testdatadir if outpath is None else outpath
        test_conf_basename = os.path.basename(self.test_conf_path)
        test_conf_name = os.path.splitext(test_conf_basename)[0]
        path = path + "/{:07d}_{}.pkl".format(current_episode_count, test_conf_name)
        with open(path, "wb")as f:
            pickle.dump(res, f)
        print("EXPERIMENT: policy dependent testing ... done")
        print("EXPERIMENT: saved testing data under {}".format(path))

    def playback(self, n_episodes, greedy=False):
        for p in self.here_pipes:
            p.send(("playback", n_episodes, greedy))
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

    def restore_all(self, path):
        self.here_pipes[0].send(("restore_all", path))
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
