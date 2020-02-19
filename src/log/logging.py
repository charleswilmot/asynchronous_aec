import numpy as np
import pickle


class DataLogger:

    def __init__(self):
        self.end_episode_data = []
        self._flush_id = 0

    def store_data(self, ret, object_distance, object_speed, object_position, eyes_position, eyes_speed, iteration,
                   episode_number, scale_reward, total_reward, task_index, episode_length, ratios):
        """Constructs a dictionary from data and store it in a buffer
        See the help guide about data formats
        """
        data = {
            "worker": np.squeeze(np.array(task_index)),
            "episode_number": np.squeeze(np.array(episode_number)),
            "iteration": np.squeeze(np.array(iteration)),
            "global_iteration": np.squeeze(np.array(episode_number * episode_length + iteration)),
            "total_reward": np.squeeze(total_reward),
            "scale_rewards": np.squeeze(scale_reward),
            "greedy_actions_indices": np.squeeze(np.array([ret["greedy_actions_indices"][k] for k in ["tilt", "pan",
                                                                                                      "vergence"]])),
            "sampled_actions_indices": np.squeeze(np.array([ret["sampled_actions_indices"][k] for k in ["tilt", "pan",
                                                                                                        "vergence"]])),
            "scale_values_tilt": np.squeeze(np.array([ret["scale_values"]["tilt"][ratio] for ratio in ratios])),
            "scale_values_pan": np.squeeze(np.array([ret["scale_values"]["pan"][ratio] for ratio in ratios])),
            "scale_values_vergence": np.squeeze(np.array([ret["scale_values"]["vergence"][ratio] for ratio in ratios])),
            "critic_values_tilt": np.squeeze(np.array(ret["critic_values"]["tilt"])),
            "critic_values_pan": np.squeeze(np.array(ret["critic_values"]["pan"])),
            "critic_values_vergence": np.squeeze(np.array(ret["critic_values"]["vergence"])),
            "object_distance": np.squeeze(np.array(object_distance)),
            "object_speed": np.squeeze(np.array(object_speed)),
            "object_position": np.squeeze(np.array(object_position)),
            "eyes_position": np.squeeze(np.array(eyes_position)),
            "eyes_speed": np.squeeze(np.array(eyes_speed))
        }
        self.end_episode_data.append(data)

    def flush_data(self, task_index, path):
        """Reformats the training data buffer and dumps it onto the hard drive
        This function must be called regularly.
        The frequency at which it is called can be specified in the command line with the option --flush-every
        See the help guide about data formats
        """
        length = len(self.end_episode_data)
        data = {k: np.zeros([length] + list(self.end_episode_data[0][k].shape), dtype=self.end_episode_data[0][k].dtype)
                for k in self.end_episode_data[0]}
        for i, d in enumerate(self.end_episode_data):
            for k, v in d.items():
                data[k][i] = v
        with open(path + "/worker_{:04d}_flush_{:04d}.pkl".format(task_index, self._flush_id), "wb") as f:
            pickle.dump(data, f)
        self.end_episode_data.clear()
        self._flush_id += 1
