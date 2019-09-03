### Help about the data formats
## Training data
# Description


The training data is gathered in the `get_trajectory` function, and passed to the `store_data` function.
Regularly, the `flush_data` function will dump the gathered data on the hard drive.

```
get_trajectory --> store_data --> flush_data
```

the `store_data` function represents the data as a list of dictionaries, while the `flush_data` function converts it to a dictionary of arrays.

The data available is the following:

```
{
    "worker":                  shape: (),                              type: np.int32,   description: id of the worker,
    "trajectory":              shape: (),                              type: np.int32,   description: trajectory number,
    "iteration":               shape: (),                              type: np.int32,   description: iteration within the trajectory,
    "global_iteration":        shape: (),                              type: np.int32,   description: iteration number "in total",
    "total_reward":            shape: (),                              type: np.float32, description: reward obtained *IN THE PREVIOUS ITERATION*,
    "scale_rewards":           shape: (n_scales, ),                    type: np.float32, description: reward obtained *IN THE PREVIOUS ITERATION* for each scale,
    "greedy_actions_indices":  shape: (3, ),                           type: np.int32,   description: index of the greedy action in the action set (order = tilt, pan, vergence),
    "sampled_actions_indices": shape: (3, ),                           type: np.int32,   description: index of the sampled action in the action set (order = tilt, pan, vergence),
    "scale_values_tilt":       shape: (n_scales, n_actions_per_joint), type: np.float32, description: critic value for each action in the tilt action set and for each scale,
    "scale_values_pan":        shape: (n_scales, n_actions_per_joint), type: np.float32, description: critic value for each action in the pan action set and for each scale,
    "scale_values_vergence":   shape: (n_scales, n_actions_per_joint), type: np.float32, description: critic value for each action in the vergence action set and for each scale,
    "critic_values_tilt":      shape: (n_actions_per_joint, ),         type: np.float32, description: critic value for each action in the tilt action set,
    "critic_values_pan":       shape: (n_actions_per_joint, ),         type: np.float32, description: critic value for each action in the pan action set,
    "critic_values_vergence":  shape: (n_actions_per_joint, ),         type: np.float32, description: critic value for each action in the vergence action set,
    "object_distance":         shape: (),                              type: np.float32, description: object distance,
    "eyes_position":           shape: (3, ),                           type: np.float32, description: tilt, pan, vergence positions,
    "eyes_speed":              shape: (2, ),                           type: np.float32, description: tilt, pan speeds
}
```

The shape is given before the dictionaries are stacked on the first dimension.

# Issues

There can be collisions (rarely) of the trajectory and the global_iteration fields between different workers, ie.

worker_2 and worker_7 can both have a trajectory number 1234 and therefore will have all global_iterations
1234 * sequence_length + 0
1234 * sequence_length + 1
1234 * sequence_length + 2
...
1234 * sequence_length + sequence_length - 1

_______________________________________________________________________________________________

Too much data is stored, it takes to much place on the hard drive and is not convenient for plotting

_______________________________________________________________________________________________



## Testing data
# Description

The test cases are pregenerated and stored in a pickle file.
This pickle file is then loaded, and the test cases are dispatched amongst workers.
Finaly, the data is gather, and pickled into the `/test_data` directory.

# Test configuration files


The testing configuration files contains a dictionary with the fields:

"test_descriptions": a dictionary of anchors
example of one anchor:
{
    "stimulus": [0, 1, 2],
    "object_distances": [0.5, 2, 4],
    "vergence_errors": [-3, -1, 1, 3],
    "speed_errors": [(0, 0)],
    "n_iterations": [20]
}
This anchor engenders all test_cases with stimulus 0, 1, 2 at every distances 0.5, 2, 4, for every initial vergence errors ......


"test_cases": a numpy array of type `dttest_case`
dttest_case = np.dtype([
    ("stimulus", np.int32),            # the id of the stimulus that must be used
    ("object_distance", np.float32),   # the distance at which the stimulus is placed
    ("vergence_error", np.float32),    # the initial vergence error of the eyes wrt the scren
    ("speed_error", (np.float32, 2)),  # the initial speed error wrt the screen
    ("n_iterations", np.int32)         # the number of iterations that must be simulated
])

______________________________________________________________________

Example of a test_conf dictionary:

test_conf = {
  "test_descriptions": {"test_name_1": test_anchors_lists, "test_name_2": test_anchors_lists},
  "test_cases": big_array_of_type_dttest_case
}

______________________________________________________________________


# Test data

```
dttest_data = np.dtype([
    ("action_index", (np.int32, 3)),             # index of the chosen action in the action set (tilt, pan, vergence)
    ("action_value", (np.float32, 3)),           # corresponding value in degrees
    ("critic_value_tilt", (np.float32, 9)),      # critic value for each action (tilt)
    ("critic_value_pan", (np.float32, 9)),       # critic value for each action (pan)
    ("critic_value_vergence", (np.float32, 9)),  # critic value for each action (vergence)
    ("total_reconstruction_error", np.float32),  # total reconstruction error
    ("eye_position", (np.float32, 3)),           # eyes positions
    ("eye_speed", (np.float32, 2)),              # eyes speeds
    ("speed_error", (np.float32, 2)),            # speed errors
    ("vergence_error", np.float32)               # vergence error
])
```
