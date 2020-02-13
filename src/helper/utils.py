import numpy as np

# interocular distance [m]
Y_EYES_DISTANCE = 0.034000 + 0.034000

deg = np.rad2deg
rad = np.deg2rad

def to_angle(other_distance):
    return deg(2 * np.arctan2(Y_EYES_DISTANCE, 2 * other_distance))

def define_actions_set(n_actions_per_joint, image_size = 320):
    n = n_actions_per_joint // 2
    mini = 90 / image_size / 2
    maxi = 90 / image_size / 2 * 2 ** (n - 1)
    positive = np.logspace(np.log2(mini), np.log2(maxi), n, base=2)
    negative = -positive[::-1]
    return np.concatenate([negative, [0], positive])