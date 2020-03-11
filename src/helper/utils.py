import numpy as np

# interocular distance [m]
Y_EYES_DISTANCE = 0.034000 + 0.034000

deg = np.rad2deg
rad = np.deg2rad

def to_angle(other_distance):
    return deg(2 * np.arctan2(Y_EYES_DISTANCE, 2 * other_distance))


def vergence_error(eyes_positions, object_distances):
    vergences = eyes_positions[..., -1]
    return to_angle(object_distances) - vergences


actions_in_semi_pixels = np.array([-8, -4, -2, -1, 0, 1, 2, 4, 8])
one_pixel = 90 / 320
actions_set_values = actions_in_semi_pixels * one_pixel / 2
actions_set_values_tilt = actions_set_values
actions_set_values_pan = actions_set_values
actions_set_values_vergence = actions_set_values
