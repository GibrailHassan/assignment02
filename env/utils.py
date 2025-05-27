import math
import numpy as np


# discretize available distance in the interval [-distance_discrete_range, distance_discrete_range]
def discretize_distance(dist, screen_size, distance_discrete_range, factor=1.0):
    if distance_discrete_range == -1:
        return discretize_distance_float(dist, screen_size, factor)
    else:
        return math.ceil(round(dist / screen_size, 2) / distance_discrete_range * 100)


def discretize_distance_float(dist, screen_size, factor=1.0):
    return dist / (screen_size * factor)


# DISCRETE ACTIONS
def calc_target_position(marine_x, marine_y, direction, distance, screen_size):
    if direction in DIRECTION_FUNCTIONS.keys():
        return DIRECTION_FUNCTIONS[direction](marine_x, marine_y, distance, screen_size)
    else:
        raise TypeError(f"no function is found for {direction}")


def calc_direction_and_distance_from_action(
    action_discrete, distance_range, distance_delta
):
    # {0, 1, 2, 3, 4, 5, 6, 7}
    direction = math.floor(action_discrete / distance_range)
    # {distance_delta, 2 * distance_delta, ..., distance_range * distance_delta},
    # say we set distance_range = 5, and screen_size = 64 then distance_delta = 12 and the range: {12, 24, 36, 48, 60}
    distance = (action_discrete % 8 + 1) * distance_delta
    return direction, distance


def up(x, y, distance, screen_size):
    return check_borders(x, y - distance, screen_size)


def right(x, y, distance, screen_size):
    return check_borders(x + distance, y, screen_size)


def down(x, y, distance, screen_size):
    return check_borders(x, y + distance, screen_size)


def left(x, y, distance, screen_size):
    return check_borders(x - distance, y, screen_size)


def up_right(x, y, distance, screen_size):
    distance = distance / math.sqrt(2)
    return check_borders(x + distance, y - distance, screen_size)


def up_left(x, y, distance, screen_size):
    distance = distance / math.sqrt(2)
    return check_borders(x - distance, y - distance, screen_size)


def down_right(x, y, distance, screen_size):
    distance = distance / math.sqrt(2)
    return check_borders(x + distance, y + distance, screen_size)


def down_left(x, y, distance, screen_size):
    distance = distance / math.sqrt(2)
    return check_borders(x - distance, y + distance, screen_size)


def check_borders(x, y, screen_size):
    if y < 0:
        y = 0
    elif y > screen_size - 1:
        y = screen_size - 1

    if x < 0:
        x = 0
    elif x > screen_size - 1:
        x = screen_size - 1

    return x, y


DIRECTION_FUNCTIONS = {
    "up": up,
    "up_right": up_right,
    "right": right,
    "down_right": down_right,
    "down": down,
    "down_left": down_left,
    "left": left,
    "up_left": up_left,
}


# there are 17 channels max in pysc2, if one selects channels > 2, they have to be preprosessed to fit into observation space
def preprocess_channels(obs):
    channels = obs.observation.feature_screen
    state_size = channels.shape

    data = np.ndarray(shape=(state_size[0], state_size[1], state_size[2]))

    c = s1 = s2 = 0

    while c < state_size[0]:
        while s1 < state_size[1]:
            while s2 < state_size[2]:
                data[c, s1, s2] = channels[c, s1, s2]
                s2 += 1
            s1 += 1
        c += 1

    return data
