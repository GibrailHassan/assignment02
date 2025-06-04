"""
A StarCraft II environment wrapper for the 'MoveToBeacon' mini-game
that provides the full visual screen features as the state.

This environment is designed for more advanced, deep RL agents, such as
Deep Q-Networks (DQN) that utilize Convolutional Neural Networks (CNNs)
to process image-like input. Unlike its discrete counterpart (`MoveToBeaconDiscreteEnv`),
this version does not simplify the state to relative coordinates but instead
provides richer, multi-channel screen observations. The action space is also
more granular, typically a flattened combination of direction and distance.
"""

import math
import numpy as np
import gym  # For gym.Env base class and gym.spaces
from pysc2.env import sc2_env  # The core StarCraft II environment
from pysc2.lib import (
    actions,
    features,
)  # For defining actions and accessing screen features
from typing import Tuple, Dict, Any, Optional  # For type hinting

# Import utility functions from the local 'utils' module within the 'env' package
# These utilities help in calculating target positions and mapping actions.
from env.utils import calc_target_position, calc_direction_and_distance_from_action

# --- PySC2 Constants (related to screen features and actions) ---
_PLAYER_RELATIVE: int = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY: int = (
    1  # Value in 'player_relative' layer for friendly units (Marine)
)
_FUNCTIONS = actions.FUNCTIONS  # Collection of available PySC2 actions

# --- Action Mapping ---
# A dictionary to map discrete action indices (for direction component) to direction names.
# These names are then used by `calc_target_position`.
ACTION_DIRECTION: Dict[int, str] = {
    0: "up_left",  # Move towards top-left
    1: "up",  # Move towards top
    2: "up_right",  # Move towards top-right
    3: "right",  # Move towards right
    4: "down_right",  # Move towards bottom-right
    5: "down",  # Move towards bottom
    6: "down_left",  # Move towards bottom-left
    7: "left",  # Move towards left
}
# Number of unique directions defined above
NUM_DIRECTIONS: int = len(ACTION_DIRECTION)


class MoveToBeaconEnv(gym.Env[np.ndarray, int]):  # Generic types for obs and action
    """
    A gym.Env wrapper for the PySC2 "MoveToBeacon" mini-game with a visual state space.

    This environment provides raw screen features (e.g., 'player_relative',
    'unit_density') as observations, suitable for agents employing CNNs.
    The action space is a flattened discrete space combining multiple movement
    directions and multiple movement distances.

    Attributes:
        screen_size (int): The resolution of the screen features (e.g., 32x32).
        distance_range (int): The number of discrete distance options for movement.
        distance_delta (float): The actual distance unit corresponding to one step
                                in the `distance_range`.
        step_mul_value (int): Number of game steps per agent action.
        visualize_game (bool): If True, PySC2 renders the game GUI.
        action_space (gym.spaces.Discrete): Flattened discrete action space.
        observation_space (gym.spaces.Box): Multi-channel visual observation space.
        _env (sc2_env.SC2Env): The underlying PySC2 environment instance.
        _obs (Optional[Any]): The last observation (TimeStep) from PySC2.
        marine_pos (np.ndarray): Current [x, y] coordinates of the Marine.
    """

    def __init__(
        self,
        distance_range: int = 8,  # Number of discrete distances the agent can choose to move
        screen_size: int = 32,  # Screen resolution (height and width)
        step_mul: int = 8,  # Number of game steps per agent action
        is_visualize: bool = False,  # Whether to render the game GUI
    ) -> None:
        """
        Initializes the MoveToBeaconEnv (full visual version).

        Args:
            distance_range (int, optional): The number of discrete distances the
                agent can choose to move in a single action. Defaults to 8.
            screen_size (int, optional): The screen resolution for game features.
                Defaults to 32 (e.g., 32x32 pixels).
            step_mul (int, optional): The number of game steps to simulate for each
                agent action. Defaults to 8.
            is_visualize (bool, optional): If True, PySC2 will render the game.
                Defaults to False.
        """
        super().__init__()  # Initialize the base gym.Env class

        self.screen_size: int = screen_size
        self.distance_range: int = distance_range
        # Calculate the increment of distance per discrete distance step.
        # Example: if screen_size is 64 and distance_range is 8, delta is 8.
        # Actions will correspond to moving 8, 16, ..., 64 units.
        self.distance_delta: float = math.floor(screen_size / distance_range)

        self.step_mul_value: int = step_mul
        self.visualize_game: bool = is_visualize

        # --- Initialize the underlying PySC2 environment ---
        self._env: sc2_env.SC2Env = sc2_env.SC2Env(
            map_name="MoveToBeacon",  # Name of the mini-game map
            players=[sc2_env.Agent(sc2_env.Race.terran)],  # Agent controls Terran units
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(
                    screen=screen_size, minimap=screen_size
                ),
                use_feature_units=True,  # Allows identification of units like the Marine
            ),
            step_mul=self.step_mul_value,
            game_steps_per_episode=0,  # Episode ends based on game conditions (beacon reached)
            visualize=self.visualize_game,
        )

        # --- Define Action Space ---
        # The action space is a composite of directions and distances, flattened into a single discrete space.
        # Total actions = number_of_directions * number_of_distance_options.
        # An agent selects a single integer, which is then un-flattened to a direction and distance.
        self.action_space: gym.spaces.Discrete = gym.spaces.Discrete(
            self.distance_range * NUM_DIRECTIONS
        )

        # --- Define Observation Space ---
        # The observation space is visual, typically composed of multiple screen feature layers.
        # Here, we use two channels: 'player_relative' and 'unit_density'.
        # Shape is (Channels, Height, Width) - common for PyTorch CNNs.
        self.observation_space: gym.spaces.Box = gym.spaces.Box(
            low=0,
            high=max(
                features.SCREEN_FEATURES.player_relative.scale,  # Max possible value for player_relative
                features.SCREEN_FEATURES.unit_density.scale,
            ),  # Max possible value for unit_density
            shape=(
                2,
                screen_size,
                screen_size,
            ),  # 2 channels: player_relative and unit_density
            dtype=np.int32,  # Data type of the observation features
        )

        # Internal state variable for Marine's position
        self.marine_pos: np.ndarray = np.array(
            [0, 0], dtype=np.float32
        )  # Marine's (x,y)
        self._obs: Optional[Any] = None  # Stores the raw PySC2 observation (TimeStep)

    def _get_state(self) -> np.ndarray:
        """
        Extracts and processes the state observation from the current PySC2 timestep.

        The state consists of two screen feature layers: 'player_relative' and
        'unit_density'. These are stacked to form a multi-channel image-like observation.

        Returns:
            np.ndarray: A NumPy array of shape (Channels, Height, Width) representing
                        the current state. Returns a zero array if the timestep is not available.
        """
        if self._obs is None:
            # Should only occur if called before the first reset.
            return np.zeros(
                self.observation_space.shape, dtype=self.observation_space.dtype
            )

        # Extract specified feature layers from the PySC2 observation
        player_relative_map: np.ndarray = (
            self._obs.observation.feature_screen.player_relative
        )
        unit_density_map: np.ndarray = self._obs.observation.feature_screen.unit_density

        # Stack these 2D feature maps to form a multi-channel (2, H, W) state.
        # Ensure dtype matches observation_space.
        state_array = np.array(
            [player_relative_map, unit_density_map], dtype=self.observation_space.dtype
        )
        return state_array

    def _set_marine_position(self) -> None:
        """
        Updates the Marine's current position based on the PySC2 observation.
        It finds pixels corresponding to friendly units and calculates their mean position.
        Requires `self._obs` to be a valid PySC2 TimeStep.
        """
        if self._obs is None:
            return

        player_relative: np.ndarray = self._obs.observation["feature_screen"][
            _PLAYER_RELATIVE
        ]
        marine_y_coords, marine_x_coords = (
            player_relative == _PLAYER_FRIENDLY
        ).nonzero()

        if marine_y_coords.any():  # If Marine is found
            self.marine_pos = (
                np.mean(list(zip(marine_x_coords, marine_y_coords)), axis=0)
                .round()
                .astype(np.float32)
            )
        # else:
        # print("Warning: Marine unit not found in _set_marine_position.")

    def reset(self) -> np.ndarray:
        """
        Resets the environment for a new episode.

        Involves resetting PySC2, selecting the army, updating Marine's position,
        and returning the initial visual state.

        Returns:
            np.ndarray: The initial visual state observation (Channels, Height, Width).
        """
        # Reset the PySC2 environment
        self._obs = self._env.reset()[0]

        # Select the army (the Marine) at the start of the episode
        self._obs = self._env.step([_FUNCTIONS.select_army("select")])[0]

        # Update the Marine's position based on the initial observation
        self._set_marine_position()

        # Return the processed initial state
        return self._get_state()

    def step(self, flat_action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes one agent action in the environment.

        The `flat_action` is an integer that needs to be un-flattened into a
        direction and a distance component for movement.

        Args:
            flat_action (int): A single discrete action index from the flattened
                               action space (0 to N_DIRECTIONS * distance_range - 1).

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: A tuple containing:
                - next_state (np.ndarray): The visual observation after the action.
                - reward (float): The reward received for the action.
                - done (bool): True if the episode has ended, False otherwise.
                - info (Dict[str, Any]): An empty dictionary (no auxiliary info).
        """
        # --- Un-flatten the action ---
        # `flat_action` needs to be converted into a direction and a distance magnitude.
        # `calc_direction_and_distance_from_action` utility handles this.
        # Note: The original `calc_direction_and_distance_from_action` in `env/utils.py` had
        # `action_discrete % 8 + 1` for distance. This assumes directions are the primary divisor.
        # If NUM_DIRECTIONS is used (e.g. 8), it should align.
        # direction_idx = flat_action // self.distance_range
        # distance_step_idx = flat_action % self.distance_range
        # actual_distance = (distance_step_idx + 1) * self.distance_delta
        # direction_str = ACTION_DIRECTION[direction_idx]

        # Using the provided utility function directly:
        # It assumes the total number of actions is NUM_DIRECTIONS * distance_range,
        # and that `flat_action` is structured such that `direction = flat_action // distance_range`
        # and `distance_component = flat_action % distance_range`.
        # The utility then maps `distance_component` to an actual distance.
        direction_idx, actual_distance = calc_direction_and_distance_from_action(
            flat_action, self.distance_range, self.distance_delta
        )
        direction_str: str = ACTION_DIRECTION[direction_idx]

        # Calculate the target (x, y) screen coordinates for the move
        target_x, target_y = calc_target_position(
            self.marine_pos[0],
            self.marine_pos[1],  # Current Marine (x,y)
            direction_str,
            actual_distance,
            self.screen_size,
        )
        # Optimistically update marine_pos; actual position will be confirmed after PySC2 step
        # self.marine_pos = np.array([target_x, target_y], dtype=np.float32) # Commented out in original too

        # --- Perform the move action in PySC2 ---
        if _FUNCTIONS.Move_screen.id in self._obs.observation.available_actions:
            self._obs = self._env.step(
                [_FUNCTIONS.Move_screen("now", (target_x, target_y))]
            )[0]
        else:
            # Fallback: If Move_screen is not available, try to re-select the army.
            # print("Warning: Move_screen not available. Attempting to select army.")
            self._obs = self._env.step([_FUNCTIONS.select_army("select")])[0]

        # Update Marine's actual position based on the observation after the move
        self._set_marine_position()

        # Extract MDP components
        next_state: np.ndarray = self._get_state()
        reward: float = float(self._obs.reward)  # Ensure reward is float
        done: bool = self._obs.last()  # True if episode ended

        info: Dict[str, Any] = {}  # No auxiliary info provided by default

        return next_state, reward, done, info

    def close(self) -> None:
        """
        Closes the environment and releases PySC2 resources.
        """
        if self._env is not None:
            self._env.close()
        super().close()

    # --- Properties for Agent Compatibility ---
    @property
    def state_shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the observation space (visual state).

        Returns:
            Tuple[int, ...]: The shape, e.g., (2, 32, 32) for 2 channels, 32x32 screen.
        """
        return self.observation_space.shape

    @property
    def action_shape(self) -> range:  # Or Tuple[int, ...]
        """
        Returns information about the action space. For discrete spaces,
        this typically returns a range representing the number of actions.

        The fix in the original file `return range(self.action_space.n)` ensures
        compatibility with agents expecting an iterable or a way to get `len()`.

        Returns:
            range: A range object from 0 to `n-1`, where `n` is the total number
                   of discrete actions.
        """
        if isinstance(self.action_space, gym.spaces.Discrete):
            return range(self.action_space.n)
        else:
            # This environment is defined with a Discrete action space.
            raise TypeError(
                "Expected Discrete action space for 'action_shape' property in MoveToBeaconEnv."
            )
