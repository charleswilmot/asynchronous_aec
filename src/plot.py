import argparse
import pickle
from helper.utils import *
import os
import numpy as np
import re
from visualization.plot_log_data import *
from algorithm.conf import Conf


def get_data(path, flush_id=None):
    return merge_flush(get_data_nonmerged(path, flush_id=flush_id))


def get_data_nonmerged(path, flush_id=None):
    if flush_id is None:
        worker_ids = get_worker_and_flush_ids(path)
        flush_id = list(worker_ids[0])
        flush_id.sort()
        return get_data_nonmerged(path, flush_id)
    elif type(flush_id) is int:
        worker_ids = get_worker_and_flush_ids(path)
        if flush_id < 0:
            flush_ids = list(worker_ids[0])
            flush_ids.sort()
            return get_data_nonmerged(path, flush_ids[flush_id:]) #TODO: Why from 'flush_id:'
        data = []
        for worker_id in worker_ids:
            filename = "/worker_{:04d}_flush_{:04d}.pkl".format(worker_id, flush_id)
            with open(path + filename, "rb") as f:
                data.append(pickle.load(f))
        return data
    else:
        data = []
        for i in flush_id:
            data += get_data_nonmerged(path, i)
        return data


def get_worker_and_flush_ids(path):
    ids = {}
    for f in os.listdir(path):
        m = re.match("worker_([0-9]+)_flush_([0-9]+).pkl", f)
        if m:
            worker_id = int(m.group(1))
            flush_id = int(m.group(2))
            if worker_id not in ids:
                ids[worker_id] = set()
            ids[worker_id].add(flush_id)
    return ids


def merge_flush(flush_list):
    dummy_key = list(flush_list[0].keys())[0]
    total_length = sum(len(f[dummy_key]) for f in flush_list)
    data = {k: np.zeros([total_length] + list(flush_list[0][k].shape[1:]), dtype=flush_list[0][k].dtype) for k in flush_list[0]}
    index = 0
    for f in flush_list:
        index_new = index + len(f[dummy_key])
        for k, v in f.items():
            data[k][index:index_new] = v
        index = index_new
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path', metavar="PATH",
        type=str,
        nargs="+",
        action='store',
        help="Path to the data."
    )

    parser.add_argument(
        '-s', '--save',
        action='store_true',
        help="If turned on, saves the plots."
    )

    parser.add_argument(
        '-o', '--overwrite',
        action='store_true',
        help="If turned on, overwrites given folder if exists."
    )

    parser.add_argument(
        '-n', '--name',
        type=str,
        help="Outdir name",
        default="default"
    )

    args = parser.parse_args()
    paths = args.path


    for path in paths:
        print("\n", path, "\n")
        ### CONSTANTS
        plotpath = path + "/../plots_{}/".format(args.name)
        with open(path + "/../conf/worker_conf.pkl", "rb") as f:
            conf = pickle.load(f)
            #discount_factor = conf.discount_factor
        ratios = list(range(1, 4))
        n_actions_per_joint = 9
        action_set = define_actions_set(n_actions_per_joint)

        data = get_data(path)
        # data = get_data(path, -5)
        #print(data)
        if args.save:
            if os.path.exists(plotpath) and not args.overwrite:
                print("[ERROR] Path exists. Turn on overwrite option with -o")
                exit()
            elif not os.path.exists(plotpath):
                os.mkdir(plotpath)

        # reward_wrt_vergence_all_scales(data, 0.5, 5, args.save)
        # reward_wrt_vergence(data, 0.5, 5, args.save)
        # critic_wrt_vergence_all_scales(data, 1, 4, args.save)
        # critic_wrt_vergence(data, 2, 3, args.save)

        # for i in range(100):
        #     data = get_data(path, i)
        #     print("flush  ", i)
        #     target_wrt_delta_vergence(data, args.save)

        # print("return_wrt_critic_value:")
        # return_wrt_critic_value(data, discount_factor, args.save)

        # print("target_wrt_delta_vergence:")
        # target_wrt_delta_vergence(data, discount_factor, args.save)



        # print("preference for correct action")
        # for i in range(3):
        #     preference_for_correct_action(data, scale=i, save=args.save)
        # preference_for_correct_action(data, save=args.save)
        # print("action_wrt_vergence:")
        # action_wrt_vergence(data, 0.5, 5, greedy=False, save=args.save)
        # print("action_wrt_vergence:")
        # action_wrt_vergence(data, 0.5, 5, greedy=True, save=args.save)
        # print("vergence_wrt_object_distance:")
        # vergence_wrt_object_distance(data, args.save)
        # for i in range(3):
        #     action_wrt_vergence_based_on_critic(data, 0.5, 5, i, save=args.save)

        # print("v shape")

        for pantilt in ['pan', 'tilt', 'pantilt']:
            mean_speed_error_episode_end_wrt_episode(data, plotpath, pantilt=pantilt, save=args.save)
            speed_error_episode_end_wrt_episode(data, plotpath, pantilt=pantilt, save=args.save)
        delta_reward_wrt_delta_speed(data, plotpath, save=args.save)
        for scale in [1, 2, 3]:
            delta_reward_wrt_delta_speed_one_scale(data, plotpath, scale, save=args.save)


        #data = generate_sample_data()
        #circular_plot(data, plotpath, save=
        #circular_polar_movement_plot_2(data, plotpath, save=args.save)
        #circular_xy_movement_plot(data, plotpath, save=args.save)
        #movement(data)

        #check_data(data, args.save)

        #data = get_data(path)

        # print("vergence_error_episode_end_wrt_episode:")
        # vergence_error_episode_end_wrt_episode(data, args.save)
        # print("vergence_episode_end_wrt_episode:")
        # vergence_episode_end_wrt_episode(data, args.save)
        # print("mean_abs_vergence_error_episode_end_wrt_episode:")
        # mean_abs_vergence_error_episode_end_wrt_episode(data, args.save)
