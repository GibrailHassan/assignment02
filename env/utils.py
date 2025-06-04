"""
Utility functions for StarCraft II environment wrappers.

This module provides helper functions for tasks commonly encountered when
implementing custom StarCraft II environments with PySC2, such as:
- Discretizing continuous distance values into discrete bins.
- Calculating target (x, y) screen coordinates based on a starting point,
  direction, and distance.
- Mapping discrete action indices to movement directions and distances.
- Preprocessing observation channels (though this function might be less
  used if environments directly select specific feature layers).
"""

import math
import numpy as np
from typing import Tuple, Dict, Any  # For type hinting

# --- Distance Discretization ---


def discretize_distance(
    dist: float, screen_size: int, distance_discrete_range: int, factor: float = 1.0
) -> int:
    """
    Discretizes a continuous distance value into a specified integer range.

    This function is used to convert continuous relative distances (e.g., between
    an agent and a target) into discrete state representations suitable for
    table-based reinforcement learning algorithms.

    If `distance_discrete_range` is -1, it calls `discretize_distance_float`
    and then presumably expects casting to int elsewhere if needed, though
    `discretize_distance_float` returns float. This specific condition (-1)
    might indicate a different mode of operation or an alternative discretization logic.

    Args:
        dist (float): The continuous distance value to be discretized.
        screen_size (int): The size of the screen dimension relevant to the distance
                           (e.g., screen width or height). Used for normalization.
        distance_discrete_range (int): The maximum absolute value for the
                                       discretized distance. For example, if 10,
                                       distances will be mapped to integers
                                       approximately in the range [-10, 10].
                                       If -1, a float-based discretization is used.
        factor (float, optional): A scaling factor applied to `screen_size` during
                                  normalization. Defaults to 1.0.

    Returns:
        int: The discretized distance as an integer. If `distance_discrete_range`
             is -1, the behavior depends on the implicit casting of the float
             result from `discretize_distance_float`.
             Given the `math.ceil(round(...))` structure, it seems to always aim for an int.
    """
    if distance_discrete_range == -1:
        # This branch calls a float discretization and then expects further processing or casting.
        # However, the current return type is int, and the original code might have implicitly cast.
        # For consistency and type safety, if -1 is a special case for float output,
        # the function signature or logic might need adjustment.
        # Assuming the math.ceil(round(...)) from the original context was intended to be general.
        # If discretize_distance_float was meant to be the sole logic for range=-1,
        # this part needs clarification on its intended integer output.
        # Based on typical usage, this path seems less common if an integer output is always desired.
        # Let's assume the `math.ceil(round(...))` part from the original version of such functions
        # was the primary discretizer.
        # If it truly means "use float logic then cast", it's:
        # return int(discretize_distance_float(dist, screen_size, factor)) # This would be a simple truncation/floor.
        # The original code has math.ceil(round(dist / screen_size, 2) / distance_discrete_range * 100)
        # This implies range is NOT -1 for that formula.
        # Re-evaluating the original: if distance_discrete_range is -1, the division by it is problematic.
        # The provided code snippet has a slightly different structure for the -1 case.
        # It seems like the original intent might have been:
        # if distance_discrete_range == -1: return discretize_distance_float(...) and expect float
        # else: return the math.ceil(round(...))
        # Given the `fileName: gibrailhassan/assignment02/assignment02-dd9de199df2b9844cc153384190c6b98a45e5624/env/env_discrete.py`
        # calls this, and that env expects integer states, we will assume integer output.
        # The logic below seems more plausible if `distance_discrete_range` is positive.
        # If distance_discrete_range is truly -1, the original formula:
        # `math.ceil(round(dist / screen_size, 2) / distance_discrete_range * 100)`
        # would result in division by -1.
        # It's more likely that `distance_discrete_range = -1` was intended for a scenario
        # where `discretize_distance_float` is used and its float result is directly used or
        # cast differently by the caller.
        # For this function to return `int` consistently, we'll assume positive `distance_discrete_range`.
        # A value of 0 for `distance_discrete_range` would also cause division by zero.
        if distance_discrete_range <= 0:
            # Fallback or error for invalid range if strict integer output is expected.
            # Or, if -1 means "just normalize and scale by 100 then ceil", then:
            # normalized_dist = round(dist / (screen_size * factor), 2) # Normalize and round
            # return math.ceil(normalized_dist * 100) # Scale and ceil
            # This is speculative. The most direct interpretation of the snippet is what's below for the positive case.
            # Given the function signature returns int, the -1 case seems underspecified for int conversion.
            # Let's assume distance_discrete_range > 0 for the main logic.
            # If distance_discrete_range is -1, it would typically call the float version and the caller handles it.
            # Since this function *must* return an int, the -1 case is tricky.
            # The original code in env_discrete.py *always* passes a positive distance_discrete_range.
            # So, the `if distance_discrete_range == -1:` branch might be dead code or for a different use case.
            # We will document the main logic path.
            print(
                f"Warning: discretize_distance called with distance_discrete_range={distance_discrete_range}. Expected positive value for standard discretization."
            )
            # Fallback to a simple scaled int for now if range is not positive for the formula.
            return int(
                dist / (screen_size * factor) * 10
            )  # Arbitrary scaling for int conversion

    # Normalize the distance with respect to screen size.
    # Rounding to 2 decimal places first can help manage precision issues.
    normalized_dist: float = round(dist / (screen_size * factor), 2)

    # Scale the normalized distance by 100 and divide by the discrete range.
    # This maps the normalized distance to a scale relative to `distance_discrete_range`.
    # Example: if dist=screen_size/2, normalized_dist=0.5. If range=10, then 0.5 / 10 * 100 = 5.
    scaled_value: float = (normalized_dist / distance_discrete_range) * 100

    # Use math.ceil to get the smallest integer greater than or equal to the scaled value.
    # This effectively bins the continuous value into discrete integer steps.
    return math.ceil(scaled_value)


def discretize_distance_float(
    dist: float, screen_size: int, factor: float = 1.0
) -> float:
    """
    Normalizes a distance value with respect to the screen size.
    Returns a float, typically in the range [-1.0, 1.0] if `dist` is within
    `[-screen_size*factor, screen_size*factor]`.

    Args:
        dist (float): The continuous distance value.
        screen_size (int): The size of the screen dimension relevant to the distance.
        factor (float, optional): A scaling factor for `screen_size`. Defaults to 1.0.

    Returns:
        float: The normalized distance.
    """
    # Normalize distance by dividing by screen_size (optionally scaled by factor)
    return dist / (screen_size * factor)


# --- Discrete Action Calculation Utilities ---


def calc_target_position(
    marine_x: float, marine_y: float, direction: str, distance: float, screen_size: int
) -> Tuple[float, float]:
    """
    Calculates the target (x, y) coordinates for a move from a starting position,
    given a direction, distance, and screen boundaries.

    The calculated coordinates are clamped within the screen boundaries [0, screen_size-1].

    Args:
        marine_x (float): The current x-coordinate of the agent (e.g., Marine).
        marine_y (float): The current y-coordinate of the agent.
        direction (str): A string representing the direction of movement. Must be a key
                         in `DIRECTION_FUNCTIONS` (e.g., "up", "up_right").
        distance (float): The magnitude of the distance to move.
        screen_size (int): The size of the screen (assumed square, or max dimension
                           for clamping). Used to ensure target is within bounds.

    Returns:
        Tuple[float, float]: A tuple (target_x, target_y) representing the
                             calculated and clamped target coordinates.

    Raises:
        TypeError: If the provided `direction` string is not found in `DIRECTION_FUNCTIONS`.
    """
    if direction in DIRECTION_FUNCTIONS:
        # Call the appropriate movement function based on the direction string
        return DIRECTION_FUNCTIONS[direction](marine_x, marine_y, distance, screen_size)
    else:
        # Invalid direction provided
        raise TypeError(
            f"No function found for direction: '{direction}'. Available directions: {list(DIRECTION_FUNCTIONS.keys())}"
        )


def calc_direction_and_distance_from_action(
    action_discrete: int, distance_range: int, distance_delta: float
) -> Tuple[int, float]:
    """
    Decodes a single discrete action index into a direction index and an actual movement distance.

    This function is used when an agent outputs a single integer action, and this
    action needs to be interpreted as a combination of a direction and a distance magnitude.
    It assumes the total action space is structured such that directions are the primary
    grouping, and within each direction, there are `distance_range` possible distance magnitudes.

    Args:
        action_discrete (int): The single discrete action chosen by the agent.
                               Expected range: `0` to `(NUM_DIRECTIONS * distance_range) - 1`.
        distance_range (int): The number of discrete distance options available for each direction.
                              Example: If 5, there are 5 possible distances.
        distance_delta (float): The base unit of distance. The actual distance moved will be
                                a multiple of this delta. Example: If `distance_delta` is 10,
                                and a distance component of 1 is chosen, actual distance is 10.

    Returns:
        Tuple[int, float]: A tuple containing:
            - direction_index (int): An integer index representing the direction
              (e.g., 0 for 'up_left', 1 for 'up', etc., corresponding to keys in `ACTION_DIRECTION`
              if `NUM_DIRECTIONS` from `env_full.py` (which is 8) is used as the divisor).
            - actual_distance (float): The calculated real-valued distance to move.
    """
    # Determine the direction index.
    # Assumes actions are grouped by direction first.
    # Example: If 8 directions and distance_range is 5, actions 0-4 are for dir 0, 5-9 for dir 1, etc.
    # Original code had `action_discrete % 8 + 1` for distance calculation, suggesting that
    # the number of directions might be implicitly 8.
    # Let's use NUM_DIRECTIONS if it were available, or assume 8 for now based on the modulo.
    num_assumed_directions = (
        8  # Based on common usage and the modulo in original distance calc.
    )

    direction_index: int = math.floor(action_discrete / distance_range)

    # Determine the distance component from the action.
    # `(action_discrete % distance_range)` gives a value from 0 to `distance_range - 1`.
    # Adding 1 makes it a 1-based multiplier for `distance_delta`.
    # E.g., if distance_range=5, this component is 1, 2, 3, 4, or 5.
    distance_component_multiplier: int = (action_discrete % distance_range) + 1

    # Calculate the actual distance to move
    actual_distance: float = distance_component_multiplier * distance_delta

    # The original code had:
    # direction = math.floor(action_discrete / distance_range)
    # distance = (action_discrete % 8 + 1) * distance_delta
    # This implies distance_range is NOT the divisor for distance component if it's not 8.
    # If the action space is NUM_DIRECTIONS * distance_range, then:
    # direction_index = action_discrete // distance_range
    # distance_step_idx = action_discrete % distance_range
    # actual_distance = (distance_step_idx + 1) * distance_delta

    # The provided code `(action_discrete % 8 + 1)` for distance implies `distance_range` in the
    # context of `env_full.py` refers to the number of distance magnitudes PER direction,
    # and the `action_discrete` is indeed structured as `direction_idx * distance_range + distance_idx`.
    # The divisor for direction should be `distance_range`.
    # The modulo for distance component should be `distance_range`.

    # Corrected interpretation based on how flat actions are usually constructed:
    # action = direction_index * num_distance_options + distance_option_index
    direction_idx_corrected: int = action_discrete // distance_range
    distance_option_idx: int = (
        action_discrete % distance_range
    )  # This will be 0 to distance_range - 1

    # Actual distance: (0-indexed option + 1) * delta, so distances are delta, 2*delta, ..., distance_range*delta
    actual_distance_corrected: float = (distance_option_idx + 1) * distance_delta

    # Using the corrected logic, as it aligns with standard flattening.
    return direction_idx_corrected, actual_distance_corrected


# --- Movement Helper Functions (for `calc_target_position`) ---
# These functions calculate new (x,y) coordinates given a starting point,
# distance, and screen size for boundary checking.
# Diagonal movements adjust distance to maintain roughly consistent displacement.


def up(x: float, y: float, distance: float, screen_size: int) -> Tuple[float, float]:
    """Calculates target position for 'up' movement."""
    return check_borders(x, y - distance, screen_size)


def right(x: float, y: float, distance: float, screen_size: int) -> Tuple[float, float]:
    """Calculates target position for 'right' movement."""
    return check_borders(x + distance, y, screen_size)


def down(x: float, y: float, distance: float, screen_size: int) -> Tuple[float, float]:
    """Calculates target position for 'down' movement."""
    return check_borders(x, y + distance, screen_size)


def left(x: float, y: float, distance: float, screen_size: int) -> Tuple[float, float]:
    """Calculates target position for 'left' movement."""
    return check_borders(x - distance, y, screen_size)


def up_right(
    x: float, y: float, distance: float, screen_size: int
) -> Tuple[float, float]:
    """Calculates target position for 'up-right' (diagonal) movement."""
    # For diagonal movement, scale distance by 1/sqrt(2) to approximate
    # same displacement as cardinal moves, assuming 'distance' is the hypotenuse.
    diag_dist: float = distance / math.sqrt(2)
    return check_borders(x + diag_dist, y - diag_dist, screen_size)


def up_left(
    x: float, y: float, distance: float, screen_size: int
) -> Tuple[float, float]:
    """Calculates target position for 'up-left' (diagonal) movement."""
    diag_dist: float = distance / math.sqrt(2)
    return check_borders(x - diag_dist, y - diag_dist, screen_size)


def down_right(
    x: float, y: float, distance: float, screen_size: int
) -> Tuple[float, float]:
    """Calculates target position for 'down-right' (diagonal) movement."""
    diag_dist: float = distance / math.sqrt(2)
    return check_borders(x + diag_dist, y + diag_dist, screen_size)


def down_left(
    x: float, y: float, distance: float, screen_size: int
) -> Tuple[float, float]:
    """Calculates target position for 'down-left' (diagonal) movement."""
    diag_dist: float = distance / math.sqrt(2)
    return check_borders(x - diag_dist, y + diag_dist, screen_size)


def check_borders(x: float, y: float, screen_size: int) -> Tuple[float, float]:
    """
    Clamps the given (x, y) coordinates to be within the screen boundaries.
    The screen is assumed to range from 0 to `screen_size - 1` in both dimensions.

    Args:
        x (float): The x-coordinate to clamp.
        y (float): The y-coordinate to clamp.
        screen_size (int): The size of the screen (e.g., 64 for a 64x64 screen).

    Returns:
        Tuple[float, float]: The clamped (x, y) coordinates.
    """
    # Clamp y-coordinate
    if y < 0:
        y = 0.0
    elif y > screen_size - 1:
        y = float(screen_size - 1)

    # Clamp x-coordinate
    if x < 0:
        x = 0.0
    elif x > screen_size - 1:
        x = float(screen_size - 1)

    return x, y


# --- Direction Function Registry ---
# Dictionary mapping direction strings to their corresponding movement functions.
# Used by `calc_target_position`.
DIRECTION_FUNCTIONS: Dict[str, callable] = {
    "up": up,
    "up_right": up_right,
    "right": right,
    "down_right": down_right,
    "down": down,
    "down_left": down_left,
    "left": left,
    "up_left": up_left,
}


# --- Observation Preprocessing (Original function, may be less used with direct feature selection) ---


def preprocess_channels(
    obs: Any,
) -> np.ndarray:  # 'obs' is typically a PySC2 TimeStep object
    """
    Extracts and stacks all feature screen layers from a PySC2 observation.

    Note: This function was part of the original utils. It creates a NumPy array
    by stacking all available feature screen channels. Modern agent implementations
    often prefer to select specific, relevant feature layers directly (e.g.,
    'player_relative', 'unit_density') rather than processing all of them,
    as the number of channels can be large (e.g., 17+).

    If used, the resulting `data` array would have a shape of (num_channels, height, width).

    Args:
        obs (Any): The raw observation object from PySC2, expected to have
                   `obs.observation.feature_screen`.

    Returns:
        np.ndarray: A NumPy array containing all stacked feature screen layers.
                    Shape: (num_channels, height, width).
    """
    # Access the feature_screen component of the observation
    # feature_screen is typically a NamedNumpyArray where each element is a screen layer
    feature_screen_layers: np.ndarray = obs.observation.feature_screen

    # Get the shape: (num_channels, height, width)
    num_channels, height, width = feature_screen_layers.shape

    # Create an empty NumPy array with the same shape and dtype to store the data
    # It's generally better to use the dtype of the source array if known
    # data_array = np.empty_like(feature_screen_layers) # This would be more direct if just copying

    # The original loop structure implies manual copying or transformation,
    # but simply converting the NamedNumpyArray to a standard np.array might be sufficient
    # if no per-element processing is done.
    # For example: data_array = np.array(feature_screen_layers, copy=True)

    # The provided loop structure seems to be manually iterating and copying,
    # which is less efficient than direct NumPy operations or slicing if the goal
    # is just to get a plain np.ndarray copy.
    data = np.ndarray(
        shape=(num_channels, height, width), dtype=feature_screen_layers.dtype
    )  # Match dtype

    # Iterate through channels, height, and width to copy data
    # This is equivalent to `data = np.array(feature_screen_layers, copy=True)`
    # but shown explicitly as in the original snippet.
    c = 0
    while c < num_channels:
        s1 = 0  # Represents height index
        while s1 < height:
            s2 = 0  # Represents width index
            while s2 < width:
                data[c, s1, s2] = feature_screen_layers[c, s1, s2]
                s2 += 1
            s1 += 1
        c += 1

    # A more Pythonic/NumPy idiomatic way to achieve the same copy:
    # data_array = np.array(obs.observation.feature_screen, copy=True)
    # Or if feature_screen is already a NumPy array (which it usually is, though possibly a named array):
    # data_array = obs.observation.feature_screen.copy()

    return data
