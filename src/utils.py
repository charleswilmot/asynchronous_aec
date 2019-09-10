import os
import getpass
import numpy as np


# interocular distance [m]
Y_EYES_DISTANCE = 0.034000 + 0.034000


deg = np.rad2deg
rad = np.deg2rad


def to_angle(other_distance):
    return deg(2 * np.arctan2(Y_EYES_DISTANCE, 2 * other_distance))
