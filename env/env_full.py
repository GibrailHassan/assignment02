# env/env_full.py

"""
A StarCraft II environment wrapper for the 'MoveToBeacon' mini-game
that provides the full visual screen features as the state.

This environment is designed for more advanced, deep RL agents like DQN with CNNs.
"""

import math
import numpy as np
import gym
from pysc2.env import sc2_env
from pysc2.lib import actions, features
from typing import Tuple, Dict, Any

from env.utils import calc_target_position, calc_direction_and_distance_from_action

# PySC2 constants
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1  # Marine
_FUNCTIONS = actions.FUNCTIONS

# A dictionary to map discrete action indices to direction names.
ACTION_DIRECTION: dict[int, str] = {
    0: 'up_left',
    1: 'up',
    2: 'up_right',
    3: 'right',
    4: 'down_right',
    5: 'down',
    6: 'down_left',
    7: 'left'
}

class MoveToBeaconEnv(gym.Env):
    """
    A gym-compatible wrapper for the PySC2 MoveToBeacon mini-game with a visual state.

    This environment provides the raw screen features as a state, making it suitable
    for agents with convolutional layers. The action space is a composite of
    direction and distance.

    Attributes:
        screen_size (int): The resolution of the screen features (e.g., 32x32).
        state_shape (tuple): The shape of the observation space (C, H, W).
        action_shape (range): The range of discrete actions.
    """
    def __init__(self,
                 distance_range: int = 8,
                 screen_size: int = 32,
                 step_mul: int = 8,
                 is_visualize: bool = False):
        """
        Initializes the MoveToBeaconEnv.

        Args:
            distance_range (int): The number of discrete distances the agent can move in.
            screen_size (int): The screen resolution for the game.
            step_mul (int): The number of game steps to take per agent action.
            is_visualize (bool): If True, a GUI of the game will be rendered.
        """
        super().__init__()
        self.screen_size = screen_size
        self.distance_range = distance_range
        self.distance_delta = math.floor(screen_size / distance_range)
        
        # Initialize the underlying PySC2 environment
        self._env = sc2_env.SC2Env(
            map_name="MoveToBeacon",
            players=[sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=screen_size, minimap=screen_size),
                use_feature_units=True
            ),
            step_mul=step_mul,
            game_steps_per_episode=0,
            visualize=is_visualize
        )
        
        # The action space combines directions and distances.
        # DQNs work with a single discrete action space, so we flatten them.
        self.action_space = gym.spaces.Discrete(self.distance_range * len(ACTION_DIRECTION))
        # The observation space is visual, composed of multiple screen feature layers.
        self.observation_space = gym.spaces.Box(
            low=0,
            high=features.SCREEN_FEATURES.player_relative.scale,
            shape=(2, screen_size, screen_size), # 2 channels: player_relative and unit_density
            dtype=np.int32
        )
        self.marine_pos: np.ndarray = np.array([0, 0])

    def _get_state(self) -> np.ndarray:
        """
        Extracts the state from the PySC2 observation.

        Returns:
            np.ndarray: A NumPy array of shape (C, H, W) representing the state.
        """
        # We use two feature layers as input channels for the CNN
        player_relative = self._obs.observation.feature_screen.player_relative
        unit_density = self._obs.observation.feature_screen.unit_density
        return np.array([player_relative, unit_density], dtype=np.int32)

    def _set_marine_position(self) -> None:
        """Finds the mean coordinates of the agent's marine on the screen."""
        player_relative = self._obs.observation["feature_screen"][_PLAYER_RELATIVE]
        marine_y, marine_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
        if marine_y.any():
            self.marine_pos = np.mean(list(zip(marine_x, marine_y)), axis=0).round()

    def reset(self) -> np.ndarray:
        """Resets the environment for a new episode."""
        self._obs = self._env.step([_FUNCTIONS.select_army("select")])[0]
        self._set_marine_position()
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Executes one step in the environment."""
        # Calculate the direction and distance from the flattened action index
        direction, distance = calc_direction_and_distance_from_action(action, self.distance_range, self.distance_delta)
        
        # Get the target coordinates based on the calculated direction and distance
        direction_str = ACTION_DIRECTION[direction]
        new_x, new_y = calc_target_position(self.marine_pos[0], self.marine_pos[1], direction_str, distance, self.screen_size)
        self.marine_pos = np.array([new_x, new_y])
        
        # Perform the move action in the PySC2 environment
        if _FUNCTIONS.Move_screen.id in self._obs.observation.available_actions:
            self._obs = self._env.step([_FUNCTIONS.Move_screen("now", self.marine_pos)])[0]
        else:
            self._obs = self._env.step([_FUNCTIONS.select_army("select")])[0]
        
        # Extract MDP components
        state = self._get_state()
        reward = self._obs.reward
        done = self._obs.last()
        return state, float(reward), done, {}

    def close(self) -> None:
        """Closes the environment."""
        if self._env is not None:
            self._env.close()
        super().close()

    @property
    def state_shape(self) -> Tuple[int, ...]:
        """Returns the shape of the observation space."""
        return self.observation_space.shape

    @property
    def action_shape(self) -> range:
        """
        Returns the range of discrete actions.

        ***** THIS IS THE FIX *****
        Returning a range object ensures that len() returns the correct number
        of actions and random.choice() samples a valid action index.
        """
        return range(self.action_space.n)