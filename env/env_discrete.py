"""
This module defines the MoveToBeaconDiscreteEnv class, a Gym-compatible
environment wrapper for the StarCraft II "MoveToBeacon" mini-game.

This version of the environment simplifies the original mini-game by:
1. Discretizing the state space: The agent's observation is the relative
   x and y distance to the beacon, with these distances binned into a
   configurable number of discrete ranges.
2. Simplifying the action space: The agent can choose from a fixed number of
   discrete movement directions (e.g., 8 directions: N, NE, E, SE, S, SW, W, NW).

This simplified environment is particularly well-suited for table-based
reinforcement learning algorithms like Q-Learning and SARSA, where a discrete
state-action space is required or highly beneficial.
"""

import numpy as np
import gym  # For gym.Env base class and gym.spaces
from pysc2.env import sc2_env  # The core StarCraft II environment
from pysc2.lib import (
    actions,
    features,
)  # For defining actions and accessing screen features
from typing import Tuple, Dict, Any, Optional  # For type hinting

# Import utility functions from the local 'utils' module within the 'env' package
from env.utils import discretize_distance, calc_target_position

# --- PySC2 Constants (related to screen features) ---
# These constants help in identifying units on the screen.
# _PLAYER_RELATIVE: Index of the 'player_relative' feature layer in observations.
#                   This layer indicates unit allegiance (friendly, enemy, neutral).
_PLAYER_RELATIVE: int = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY: int = (
    1  # Value in 'player_relative' layer representing a friendly unit (e.g., the Marine)
)
_PLAYER_NEUTRAL: int = (
    3  # Value in 'player_relative' layer representing a neutral unit (e.g., the Beacon)
)

# _FUNCTIONS: A collection of available actions in PySC2.
_FUNCTIONS = actions.FUNCTIONS

# --- Action Mapping ---
# Defines a mapping from discrete action indices (0-7) to string representations
# of movement directions. These directions are then used by `calc_target_position`.
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


class MoveToBeaconDiscreteEnv(
    gym.Env[np.ndarray, int]
):  # Generic types for obs and action
    """
    A gym.Env wrapper for the PySC2 "MoveToBeacon" mini-game with a discrete state and action space.

    The agent controls a single Marine and aims to reach a Beacon. The state is
    the discretized relative (x, y) distance to the Beacon. Actions are discrete
    movements in one of 8 directions.

    Attributes:
        distance_discrete_range (int): The number of bins or range for discretizing
                                       the x and y distances to the beacon.
        distance_to_move (int): The fixed distance the Marine attempts to move in one step.
                                (Name changed from `distance` to avoid conflict with gym.Env properties).
        screen_size (int): The resolution of the game screen (e.g., 64x64).
        step_mul (int): Number of game steps per agent action.
        visualize (bool): Whether to render the game GUI.
        action_space (gym.spaces.Discrete): Discrete action space (8 directions).
        observation_space (gym.spaces.Box): 2D box representing discretized [dx, dy] to beacon.
        _env (sc2_env.SC2Env): The underlying PySC2 environment instance.
        _obs (Optional[Any]): The last observation received from PySC2. Type is PySC2's TimeStep.
        marine_pos (np.ndarray): Current [x, y] coordinates of the Marine.
        beacon_pos (np.ndarray): Current [x, y] coordinates of the Beacon.
        last_action (Optional[int]): The last action taken by the agent.
        current_reward (float): Reward from the last step. (Name changed from `reward`).
        current_state (np.ndarray): Current discretized state. (Name changed from `state`).
    """

    def __init__(
        self,
        distance_discrete_range: int = 10,  # Range for discretizing distance values
        distance: int = 1,  # How far the marine attempts to move in one step (original param name)
        screen_size: int = 64,  # Screen resolution
        step_mul: int = 8,  # Game steps per agent action
        is_visualize: bool = False,  # Whether to render the game
    ):
        """
        Initializes the MoveToBeaconDiscreteEnv.

        Args:
            distance_discrete_range (int, optional): Defines the range for discretizing
                the x and y distances to the beacon. For example, if 10, distances
                are mapped to bins like [-10, ..., 0, ..., 10]. Defaults to 10.
            distance (int, optional): The magnitude of movement the Marine attempts
                in one action step. Defaults to 1 (a small step).
            screen_size (int, optional): The screen resolution. Defaults to 64.
            step_mul (int, optional): Number of game steps per agent action.
                Defaults to 8.
            is_visualize (bool, optional): If True, PySC2 will render the game.
                Defaults to False.
        """
        super().__init__()  # Initialize the base gym.Env class

        self.distance_discrete_range: int = distance_discrete_range
        self.distance_to_move: int = distance  # Renamed for clarity within the class
        self.screen_size: int = screen_size
        self.step_mul_value: int = step_mul  # Renamed to avoid potential conflicts
        self.visualize: bool = is_visualize

        # --- Initialize the underlying PySC2 environment ---
        self._env: sc2_env.SC2Env = sc2_env.SC2Env(
            map_name="MoveToBeacon",  # Name of the mini-game map
            players=[sc2_env.Agent(sc2_env.Race.terran)],  # Agent controls Terran units
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(
                    screen=self.screen_size, minimap=self.screen_size
                ),
                use_feature_units=True,  # To identify Marine and Beacon units
            ),
            step_mul=self.step_mul_value,
            game_steps_per_episode=0,  # Episode ends when beacon is reached or game ends
            visualize=self.visualize,
        )

        # --- Define Action Space ---
        # Discrete action space corresponding to the number of directions defined in ACTION_DIRECTION.
        self.action_space: gym.spaces.Discrete = gym.spaces.Discrete(
            len(ACTION_DIRECTION)
        )

        # --- Define Observation Space ---
        # A 2D Box space representing the discretized [dx, dy] distance to the beacon.
        # 'dx' and 'dy' can range from -distance_discrete_range to +distance_discrete_range.
        # Shape is (2,) for the [dx, dy] vector.
        self.observation_space: gym.spaces.Box = gym.spaces.Box(
            low=-self.distance_discrete_range,
            high=self.distance_discrete_range,
            shape=(2,),  # A 2-element vector: [discretized_x_dist, discretized_y_dist]
            dtype=np.int32,  # Data type for the state representation
        )

        # Internal state variables for the wrapper
        self._obs: Optional[Any] = None  # Stores the raw PySC2 observation (TimeStep)
        self.marine_pos: np.ndarray = np.array(
            [0, 0], dtype=np.float32
        )  # Marine's (x,y)
        self.beacon_pos: np.ndarray = np.array(
            [0, 0], dtype=np.float32
        )  # Beacon's (x,y)

        # MDP components (initialized in reset, stored for potential inspection)
        self.last_action: Optional[int] = None
        self.current_reward: float = 0.0  # Renamed from 'reward'
        self.current_state: np.ndarray = np.zeros(
            self.observation_space.shape, dtype=self.observation_space.dtype
        )  # Renamed from 'state'

    def _set_marine_position(self) -> None:
        """
        Updates the Marine's current position based on the PySC2 observation.
        It finds pixels corresponding to friendly units and calculates their mean position.
        Requires `self._obs` to be a valid PySC2 TimeStep.
        """
        if self._obs is None:
            return  # Should not happen after reset

        player_relative: np.ndarray = self._obs.observation["feature_screen"][
            _PLAYER_RELATIVE
        ]
        # Find all (y, x) coordinates where player_relative indicates a friendly unit
        marine_y_coords, marine_x_coords = (
            player_relative == _PLAYER_FRIENDLY
        ).nonzero()

        if marine_y_coords.any():  # Check if any friendly units (Marine) are found
            # Calculate the mean position (centroid) of all friendly unit pixels
            self.marine_pos = (
                np.mean(list(zip(marine_x_coords, marine_y_coords)), axis=0)
                .round()
                .astype(np.float32)
            )
        # else:
        # Marine not found (e.g., if it died, though not possible in MoveToBeacon)
        # print("Warning: Marine unit not found in observation.")

    def _set_beacon_position(self) -> None:
        """
        Updates the Beacon's current position based on the PySC2 observation.
        It finds pixels corresponding to neutral units (Beacon) and calculates their mean position.
        Requires `self._obs` to be a valid PySC2 TimeStep.
        """
        if self._obs is None:
            return

        player_relative: np.ndarray = self._obs.observation["feature_screen"][
            _PLAYER_RELATIVE
        ]
        # Find all (y, x) coordinates where player_relative indicates a neutral unit
        beacon_y_coords, beacon_x_coords = (
            player_relative == _PLAYER_NEUTRAL
        ).nonzero()

        if beacon_y_coords.any():  # Check if any neutral units (Beacon) are found
            # Calculate the mean position (centroid) of all Beacon pixels
            self.beacon_pos = (
                np.mean(list(zip(beacon_x_coords, beacon_y_coords)), axis=0)
                .round()
                .astype(np.float32)
            )
        # else:
        # Beacon not found (should not happen unless episode just ended and a new one hasn't fully started)
        # print("Warning: Beacon unit not found in observation.")

    def _get_state(self) -> np.ndarray:
        """
        Calculates the discretized state (relative distance to beacon) from current
        Marine and Beacon positions.

        Returns:
            np.ndarray: A 2D NumPy array `[dx, dy]` representing the discretized
                        relative distance from the Marine to the Beacon.
        """
        # Calculate linear distances in x and y coordinates
        x_distance: float = self.beacon_pos[0] - self.marine_pos[0]
        y_distance: float = self.beacon_pos[1] - self.marine_pos[1]

        # Discretize these distances using the utility function
        discretized_x = discretize_distance(
            x_distance, self.screen_size, self.distance_discrete_range
        )
        discretized_y = discretize_distance(
            y_distance, self.screen_size, self.distance_discrete_range
        )

        return np.array([discretized_x, discretized_y], dtype=np.int32)

    def reset(self) -> np.ndarray:
        """
        Resets the environment for a new episode.

        This involves:
        1. Resetting the underlying PySC2 environment.
        2. Selecting the Marine (issuing a 'select_army' action).
        3. Determining the initial positions of the Marine and Beacon.
        4. Calculating and returning the initial discretized state.

        Returns:
            np.ndarray: The initial discretized state observation.
        """
        # Reset the PySC2 environment and get the first observation
        self._obs = self._env.reset()[0]  # [0] for single-agent environment

        # Initial action: select the army (the Marine)
        # This is necessary for subsequent 'Move_screen' actions to work.
        # We step the PySC2 environment with this selection action.
        self._obs = self._env.step([_FUNCTIONS.select_army("select")])[0]

        # Update internal positions of Marine and Beacon from the new observation
        self._set_marine_position()
        self._set_beacon_position()

        # Reset internal MDP tracking variables
        self.last_action = None
        self.current_reward = 0.0
        self.current_state = self._get_state()  # Calculate initial discretized state

        return self.current_state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes one agent action in the environment.

        Args:
            action (int): The discrete action index (0-7) representing a movement direction.

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: A tuple containing:
                - next_state (np.ndarray): The discretized observation after the action.
                - reward (float): The reward received for the action.
                - done (bool): True if the episode has ended, False otherwise.
                - info (Dict[str, Any]): An empty dictionary (no auxiliary info provided).
        """
        self.last_action = action  # Store the action taken

        # Calculate the target (x, y) coordinates on the screen for the Marine's move
        # based on the chosen action (direction) and `self.distance_to_move`.
        target_x, target_y = calc_target_position(
            self.marine_pos[0],  # Current Marine x
            self.marine_pos[1],  # Current Marine y
            ACTION_DIRECTION[action],  # String representation of the direction
            self.distance_to_move,  # How far to attempt to move
            self.screen_size,  # Screen dimensions for boundary checks
        )
        # Update internal marine position optimistically (PySC2 will handle actual movement)
        # self.marine_pos = np.array([target_x, target_y], dtype=np.float32) # This was potentially problematic if actual pos drifts

        # Perform the 'Move_screen' action in the PySC2 environment
        # Check if 'Move_screen' is an available action in the current observation
        if _FUNCTIONS.Move_screen.id in self._obs.observation.available_actions:
            # Issue the move command. "[0]" for "now" (not queued).
            self._obs = self._env.step(
                [_FUNCTIONS.Move_screen("now", (target_x, target_y))]
            )[0]
        else:
            # If Move_screen is not available (e.g., Marine not selected), try to re-select.
            # This is a fallback; ideally, the Marine should remain selected.
            # print("Warning: Move_screen not available. Attempting to select army.")
            self._obs = self._env.step([_FUNCTIONS.select_army("select")])[0]
            # Note: If army selection itself fails or doesn't make Move_screen available,
            # the agent might get stuck. More robust error handling could be added.

        # Update Marine's position based on the new observation from PySC2 after the move.
        # This gives the actual position rather than the targeted one.
        self._set_marine_position()

        # Extract reward from the PySC2 observation
        self.current_reward = float(self._obs.reward)

        # If a reward of 1 is received, it means the beacon was reached.
        # In this case, a new beacon appears, so its position needs to be updated.
        if self.current_reward == 1.0:
            self._set_beacon_position()  # Update beacon position for the new state calculation

        # Calculate the new discretized state based on updated Marine and Beacon positions
        self.current_state = self._get_state()

        # Check if the episode has ended (PySC2's 'last()' method)
        done: bool = self._obs.last()

        # info dictionary (currently empty, can be extended)
        info: Dict[str, Any] = {}

        return self.current_state, self.current_reward, done, info

    def roll_to_next_state(self, action: int) -> np.ndarray:
        """
        Calculates a hypothetical next state (raw, non-discretized distances)
        if the given action were taken, without actually stepping the environment.

        This method is intended for use by simple heuristic agents (like BasicAgent)
        that might want to preview the outcome of actions without affecting the
        true environment state. It does not discretize the resulting distances.

        Args:
            action (int): The discrete action index (0-7) representing a movement direction.

        Returns:
            np.ndarray: A 2D NumPy array `[raw_dx, raw_dy]` representing the
                        hypothetical raw relative distances to the current beacon
                        if the action were taken.
        """
        # Calculate the hypothetical new Marine position if the action were taken
        hypothetical_marine_x, hypothetical_marine_y = calc_target_position(
            self.marine_pos[0],
            self.marine_pos[1],
            ACTION_DIRECTION[action],
            self.distance_to_move,
            self.screen_size,
        )

        # Calculate raw (non-discretized) distances to the current beacon
        raw_x_distance: float = self.beacon_pos[0] - hypothetical_marine_x
        raw_y_distance: float = self.beacon_pos[1] - hypothetical_marine_y

        return np.array([raw_x_distance, raw_y_distance], dtype=np.float32)

    def save_replay(self, replay_dir: str) -> None:
        """
        Saves a replay of the current episode if the underlying PySC2 environment supports it.

        Args:
            replay_dir (str): The directory where the replay file should be saved.
        """
        if self._env is not None and hasattr(self._env, "save_replay"):
            try:
                self._env.save_replay(replay_dir)
                # print(f"Replay saved in directory: {replay_dir}")
            except Exception as e:
                print(f"Error saving replay: {e}")
        # else:
        # print("Replay saving not supported or environment not initialized.")

    def close(self) -> None:
        """
        Closes the environment and releases any resources.
        This should be called when the environment is no longer needed.
        """
        if self._env is not None:
            self._env.close()  # Close the underlying PySC2 environment
        super().close()  # Call close on the base gym.Env class if it has specific cleanup

    # --- Properties for Agent Compatibility (Consistent with AbstractAgent expectations) ---
    @property
    def state_shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the observation space (discretized state).

        Returns:
            Tuple[int, ...]: The shape of the observation space, e.g., (2,) for [dx, dy].
        """
        return self.observation_space.shape

    @property
    def action_shape(self) -> range:  # Or Tuple[int, ...] if representing shape
        """
        Returns information about the action space. For discrete spaces,
        this typically returns a range representing the number of actions.

        This property is provided for compatibility with agent expectations.
        For a `gym.spaces.Discrete(n)`, `n` is the number of actions.
        Returning `range(self.action_space.n)` provides an iterable of action indices.

        Returns:
            range: A range object from 0 to `n-1`, where `n` is the number of discrete actions.
        """
        if isinstance(self.action_space, gym.spaces.Discrete):
            return range(self.action_space.n)
        else:
            # Fallback or raise error if action space is not Discrete as expected by some agents
            # For this environment, action_space is always Discrete.
            raise TypeError(
                "Expected Discrete action space for 'action_shape' property in this context."
            )
