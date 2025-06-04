"""
This module defines the DefeatRoachesEnv class, a Gym-compatible environment
wrapper for the StarCraft II "DefeatRoaches" mini-game.

In this mini-game, the agent controls a group of Marines and the objective is
to defeat a group of enemy Roaches. The environment provides visual features
from the game screen as observations and accepts screen-based attack commands
as actions. This environment is suitable for agents that can process spatial,
image-like input, such as a DQN with a Convolutional Neural Network (CNN).
"""

import gym
import numpy as np
from typing import Any, Dict, Tuple, Optional  # Added Optional

# Import specific classes and modules from gym.spaces for type hinting and defining spaces
from gym.spaces import Discrete, Box

# Import necessary components from the PySC2 library
from pysc2.env import sc2_env  # The core StarCraft II environment
from pysc2.lib import (
    actions,
    features,
)  # For defining actions and accessing screen features

# Define a type alias for the pysc2 TimeStep object for clarity in type hints
PySC2TimeStep = (
    Any  # sc2_env.TimeStep is not directly importable for typing in all contexts
)


class DefeatRoachesEnv(
    gym.Env[np.ndarray, int]
):  # Generic types for observation and action
    """
    A gym.Env wrapper for the PySC2 "DefeatRoaches" mini-game.

    The agent controls a group of Marines with the goal of defeating enemy Roaches.
    Observations are derived from screen features, typically the 'player_relative'
    layer, reshaped to include a channel dimension (H, W, C). Actions are discrete,
    representing an attack command on a specific pixel of the game screen.

    Attributes:
        screen_size (int): The resolution of the screen observation (e.g., 64 for 64x64 pixels).
        step_mul (int): The number of game steps to take per agent action.
        is_visualize (bool): If True, a GUI of the game will be rendered by PySC2.
        reselect_army_each_nstep (int): Frequency (in agent steps) to re-issue the
                                        'select_army' command. This can help if units
                                        get deselected. If 0, army is selected only at reset.
        action_space (gym.spaces.Discrete): The action space, a discrete value for each
                                            pixel on the screen.
        observation_space (gym.spaces.Box): The observation space, representing the
                                            player_relative screen feature map.
        attack_action_id (int): The PySC2 ID for the 'Attack_screen' action.
        select_army_action_id (int): The PySC2 ID for the 'select_army' action.
        _env (sc2_env.SC2Env): The underlying PySC2 environment instance.
        _pysc2_timestep (Optional[PySC2TimeStep]): The last timestep received from PySC2.
        _step_counter_env (int): Counter for the number of steps taken within an episode.
    """

    def __init__(
        self,
        screen_size: int = 64,  # Screen resolution (height and width)
        step_mul: int = 8,  # Number of game steps per agent action
        is_visualize: bool = False,  # Whether to render the game GUI
        reselect_army_each_nstep: int = 5,  # How often to re-select the army
    ):
        """
        Initializes the DefeatRoachesEnv.

        Args:
            screen_size (int, optional): The resolution for the screen features
                                         (e.g., 64 for a 64x64 observation).
                                         Defaults to 64.
            step_mul (int, optional): The number of game steps that are simulated
                                      for each action taken by the agent.
                                      Defaults to 8.
            is_visualize (bool, optional): If True, PySC2 will attempt to render
                                           the game. Defaults to False.
            reselect_army_each_nstep (int, optional): The agent will re-issue a
                                                      'select_army' command every
                                                      `reselect_army_each_nstep` steps.
                                                      Set to 0 to disable periodic re-selection
                                                      (army selected only at reset).
                                                      Defaults to 5.
        """
        super(DefeatRoachesEnv, self).__init__()  # Initialize the base gym.Env class

        self.screen_size: int = screen_size
        self.step_mul_value: int = (
            step_mul  # Renamed to avoid conflict if gym.Env had step_mul
        )
        self.visualize_game: bool = is_visualize
        self.reselect_army_each_nstep: int = reselect_army_each_nstep
        self._step_counter_env: int = (
            0  # Initialize step counter for the current episode
        )

        # --- Initialize the underlying PySC2 environment ---
        self._env: sc2_env.SC2Env = sc2_env.SC2Env(
            map_name="DefeatRoaches",  # Name of the mini-game map
            players=[
                sc2_env.Agent(sc2_env.Race.random)
            ],  # Agent plays as random race (usually Terran Marines)
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(
                    screen=self.screen_size,
                    minimap=self.screen_size,  # Set screen/minimap resolution
                ),
                use_feature_units=True,  # Enable detailed unit information (though not heavily used by this env's state)
            ),
            step_mul=self.step_mul_value,  # Set step multiplier
            game_steps_per_episode=0,  # Let the game decide episode end (no fixed step limit from env)
            visualize=self.visualize_game,  # Enable/disable PySC2 visualization
        )

        # --- Define Action Space ---
        # The action space is discrete, with one action for each pixel on the screen.
        # An action `i` corresponds to attacking the (y, x) coordinates derived from `i`.
        # Total actions = screen_size * screen_size.
        self.action_space: Discrete = Discrete(self.screen_size * self.screen_size)

        # --- Define Observation Space ---
        # The observation is typically the 'player_relative' feature screen layer.
        # It's a 2D map where pixel values indicate unit allegiance (background, self, ally, neutral, enemy).
        # Shape is (Height, Width, Channels). Here, one channel for player_relative.
        # Values range from 0 up to the scale of player_relative feature.
        self.observation_space: Box = Box(
            low=0,
            high=features.SCREEN_FEATURES.player_relative.scale,  # Max value for player_relative feature
            shape=(
                self.screen_size,
                self.screen_size,
                1,
            ),  # Format: (Height, Width, Channels)
            dtype=np.int32,  # Data type of the observation
        )

        # Store PySC2 action IDs for convenience
        self.attack_action_id: int = actions.FUNCTIONS.Attack_screen.id
        self.select_army_action_id: int = actions.FUNCTIONS.select_army.id

        # Placeholder for the last timestep received from PySC2
        self._pysc2_timestep: Optional[PySC2TimeStep] = None

    def reset(self) -> np.ndarray:
        """
        Resets the environment to the start of a new episode.

        This involves resetting the underlying PySC2 environment, selecting
        the agent's army, and returning the initial state observation.

        Returns:
            np.ndarray: The initial state observation of the environment,
                        matching `self.observation_space`.
        """
        self._step_counter_env = 0  # Reset episode step counter
        # Reset the PySC2 environment and get the first timestep
        self._pysc2_timestep = self._env.reset()[
            0
        ]  # [0] because reset() returns a list of timesteps (for multi-agent)

        # Ensure the agent's army is selected at the beginning of the episode
        self._select_army()

        # Return the processed initial state
        return self._get_state()

    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes one agent action in the environment.

        This involves:
        1. Optionally re-selecting the army.
        2. Translating the discrete `action_index` to PySC2 screen coordinates.
        3. Issuing an 'Attack_screen' command to PySC2.
        4. Processing the resulting timestep from PySC2 to get the new state,
           reward, done flag, and additional info.

        Args:
            action_index (int): A discrete action index, where
                                `0 <= action_index < screen_size * screen_size`.
                                This index is mapped to (y, x) screen coordinates.

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: A tuple containing:
                - next_state (np.ndarray): The observation after the action.
                - reward (float): The reward received for the action.
                - done (bool): True if the episode has ended, False otherwise.
                - info (Dict[str, Any]): A dictionary with auxiliary diagnostic
                  information (e.g., cumulative score).
        """
        # Periodically re-select the army to ensure units respond to commands
        # This can be helpful if units sometimes get deselected during gameplay.
        if (
            self.reselect_army_each_nstep
            > 0  # Check if periodic re-selection is enabled
            and self._step_counter_env
            > 0  # Don't re-select at the very first step (already done in reset)
            and self._step_counter_env % self.reselect_army_each_nstep == 0
        ):
            self._select_army()

        # Execute the PySC2 step with the chosen action
        self._pysc2_timestep = self._pysc2_step_with_attack(action_index)
        self._step_counter_env += 1  # Increment the episode step counter

        # Extract MDP components from the PySC2 timestep
        state: np.ndarray = self._get_state()
        reward: float = float(self._pysc2_timestep.reward)  # Ensure reward is a float
        done: bool = self._pysc2_timestep.last()  # True if episode ended

        # Include cumulative score in the info dictionary
        # score_cumulative is a PySC2 observation field
        info: Dict[str, Any] = {
            "score": self._pysc2_timestep.observation["score_cumulative"][0]
        }

        return state, reward, done, info

    def _pysc2_step_with_attack(self, action_index: int) -> PySC2TimeStep:
        """
        Helper function to convert a discrete action index to PySC2 attack command
        and execute it. Handles potential errors if the attack command is not available.

        Args:
            action_index (int): The discrete action index.

        Returns:
            PySC2TimeStep: The timestep returned by PySC2 after the action.
        """
        # Convert the 1D action_index into 2D screen coordinates (y, x)
        # np.unravel_index gives (row_idx, col_idx), which corresponds to (y, x) for screen
        coords_yx: Tuple[int, int] = np.unravel_index(
            action_index, (self.screen_size, self.screen_size)
        )
        # PySC2 actions often expect (x, y) order for screen coordinates
        coords_xy: Tuple[int, int] = (coords_yx[1], coords_yx[0])  # Swap to (x,y)

        # Construct the PySC2 action: Attack_screen at coords_xy
        # The [0] for "queued" argument means "now" (not queued).
        pysc2_action = [actions.FunctionCall(self.attack_action_id, [[0], coords_xy])]

        timestep: PySC2TimeStep
        try:
            # Attempt to execute the attack action
            timestep = self._env.step(pysc2_action)[0]
        except ValueError as e:
            # A ValueError can occur if the action (e.g., Attack_screen) is not available,
            # often because no units are selected or the ability is on cooldown.
            # print(f"Warning: Attack action failed (possibly no units selected or action unavailable). Error: {e}")
            # As a fallback, re-select the army and effectively "skip" this attack attempt.
            # The agent will try to act again in the next environment step.
            self._select_army()
            # Return the timestep resulting from the select_army action,
            # or the last known valid one if select_army also had issues (unlikely here).
            # For simplicity, we step with a select_army action to get a new timestep.
            # This ensures _pysc2_timestep is always updated.
            timestep = self._env.step(
                [
                    actions.FunctionCall(self.select_army_action_id, [[0]])
                ]  # [[0]] for "select all"
            )[0]
        return timestep

    def _select_army(self) -> None:
        """
        Issues the 'select_army' command to PySC2.

        This action selects all controllable units for the agent. It's important
        because many actions (like 'Attack_screen') require units to be selected.
        """
        # This step action also returns a timestep, but we primarily care about its
        # side effect of selecting units. The timestep itself might be used implicitly
        # if _get_state is called immediately after, but often it's just to set game state.
        # The [[0]] argument for select_army typically means "select all" or "add to selection".
        self._env.step([actions.FunctionCall(self.select_army_action_id, [[0]])])
        # print("Issued select_army command.") # Optional: for debugging

    def _get_state(self) -> np.ndarray:
        """
        Extracts and processes the state observation from the current PySC2 timestep.

        The state is derived from the 'player_relative' screen feature, which indicates
        the allegiance of units/structures at each pixel. It's reshaped to include a
        channel dimension (H, W, 1) to be compatible with typical CNN inputs.

        Returns:
            np.ndarray: The processed state observation (H, W, C). Returns a zero array
                        of the correct shape if the timestep is not yet available (e.g., before first reset).
        """
        if self._pysc2_timestep is None:
            # This case should only happen if _get_state is called before the first reset.
            # Return a zero array matching the observation space shape.
            return np.zeros(
                self.observation_space.shape, dtype=self.observation_space.dtype
            )

        # Extract the 'player_relative' feature layer from the full observation
        feature_screen: np.ndarray = self._pysc2_timestep.observation["feature_screen"]
        player_relative_map: np.ndarray = feature_screen[
            features.SCREEN_FEATURES.player_relative.index
        ]  # This is a 2D array (H, W)

        # Create a copy and ensure the correct data type (as defined in observation_space)
        state_array: np.ndarray = np.array(
            player_relative_map, copy=True, dtype=self.observation_space.dtype
        )

        # Reshape to add the channel dimension: (H, W) -> (H, W, 1)
        # This makes it compatible with CNNs expecting (Batch, Height, Width, Channels)
        # or (Batch, Channels, Height, Width) after permutation.
        return state_array[..., np.newaxis]

    def close(self) -> None:
        """
        Closes the environment and releases any resources.
        This should be called when the environment is no longer needed.
        """
        if self._env is not None:
            self._env.close()  # Close the underlying PySC2 environment
        super().close()  # Call close on the base gym.Env class if it has any specific cleanup

    @property
    def unwrapped(self) -> "DefeatRoachesEnv":  # type: ignore[override]
        """
        Returns the base environment, bypassing any wrappers.
        In this case, it returns itself as it's not a wrapper in the Gym sense.

        Returns:
            DefeatRoachesEnv: The instance itself.
        """
        return self

    def render(self, mode: str = "human") -> Optional[np.ndarray]:  # type: ignore[override]
        """
        Renders the environment.

        - If `mode` is "human" and visualization was enabled during `__init__`,
          PySC2 handles rendering, so this method does nothing.
        - If `mode` is "rgb_array", it attempts to return the RGB screen observation
          from PySC2 if available.

        Args:
            mode (str, optional): The mode to render with. Supported modes:
                                  "human", "rgb_array". Defaults to "human".

        Returns:
            Optional[np.ndarray]: If mode is "rgb_array", returns an RGB array
                                  of the screen. Otherwise, returns None or
                                  relies on PySC2's external rendering.
        """
        if mode == "human" and self.visualize_game:
            # PySC2 handles its own rendering window if visualize=True was set in __init__
            return None  # Or, could try to ensure window is active/visible if PySC2 API allows
        elif mode == "rgb_array":
            # Try to return the RGB screen if available in the observation
            if (
                self._pysc2_timestep is not None
                and "rgb_screen" in self._pysc2_timestep.observation
            ):
                return self._pysc2_timestep.observation["rgb_screen"]
            else:
                # Fallback: return a black image if rgb_screen is not available
                return np.zeros((self.screen_size, self.screen_size, 3), dtype=np.uint8)
        else:
            # For other modes, delegate to the base class's render method
            return super().render(mode=mode)
