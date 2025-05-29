import gym
import numpy as np


from typing import Any, Dict, Tuple  # Added Any here
from gym.spaces import Discrete, Box
from pysc2.env import sc2_env
from pysc2.lib import actions, features


class DefeatRoachesEnv(gym.Env[np.ndarray[Any, np.dtype[np.int32]], int]):

    def __init__(
        self,
        screen_size: int = 64,
        step_mul: int = 8,
        is_visualize: bool = False,
        reselect_army_each_nstep: int = 5,
    ):
        super(DefeatRoachesEnv, self).__init__()

        self.screen_size = screen_size
        self.reselect_army_each_nstep = reselect_army_each_nstep
        self._step_counter_env = 0

        self._env = sc2_env.SC2Env(
            map_name="DefeatRoaches",
            players=[sc2_env.Agent(sc2_env.Race.random)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(
                    screen=self.screen_size, minimap=self.screen_size
                ),
                use_feature_units=True,
            ),
            step_mul=step_mul,
            game_steps_per_episode=0,
            visualize=is_visualize,
        )

        self.action_space = Discrete(self.screen_size * self.screen_size)

        self.observation_space = Box(
            low=0,
            high=features.SCREEN_FEATURES.player_relative.scale,
            shape=[self.screen_size, self.screen_size, 1],  # H, W, C
            dtype=np.int32,
        )

        self.attack_action_id = actions.FUNCTIONS.Attack_screen.id
        self.select_army_action_id = actions.FUNCTIONS.select_army.id

        self._pysc2_timestep = None

    def reset(self) -> np.ndarray:
        self._step_counter_env = 0
        self._pysc2_timestep = self._env.reset()[0]
        self._select_army()
        return self._get_state()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:  # Use Dict for info
        if (
            self.reselect_army_each_nstep > 0
            and self._step_counter_env > 0
            and self._step_counter_env % self.reselect_army_each_nstep == 0
        ):
            self._select_army()

        self._pysc2_timestep = self._pysc2_step(action)
        self._step_counter_env += 1

        state = self._get_state()
        reward = float(self._pysc2_timestep.reward)
        done = self._pysc2_timestep.last()

        info = {"score": self._pysc2_timestep.observation["score_cumulative"][0]}

        return state, reward, done, info

    def _pysc2_step(self, action_index: int) -> Any:  # Returns a pysc2 TimeStep object
        coords_yx = np.unravel_index(action_index, (self.screen_size, self.screen_size))
        coords_xy = coords_yx[::-1]

        pysc2_action = [actions.FunctionCall(self.attack_action_id, [[0], coords_xy])]

        try:
            timestep = self._env.step(pysc2_action)[0]
        except ValueError:
            # If attack failed (e.g. no units selected), select army and return current timestep
            # The agent will try to act again in the next environment step.
            self._select_army()
            # Return the timestep resulting from the select_army action, or the last known valid one.
            # For simplicity, we can just step with select_army.
            timestep = self._env.step(
                [actions.FunctionCall(self.select_army_action_id, [[0]])]
            )[0]
        return timestep

    def _select_army(self) -> None:
        """Selects all units of the agent's army."""
        # This step action also returns a timestep, but we don't need to process it here
        # as its primary purpose is to change the game state (unit selection).
        self._env.step([actions.FunctionCall(self.select_army_action_id, [[0]])])

    def _get_state(self) -> np.ndarray:
        if self._pysc2_timestep is None:
            return np.zeros(
                self.observation_space.shape, dtype=self.observation_space.dtype
            )

        feature_screen = self._pysc2_timestep.observation["feature_screen"]
        player_relative_map = feature_screen[
            features.SCREEN_FEATURES.player_relative.index
        ]

        state_array = np.array(
            player_relative_map, copy=True, dtype=self.observation_space.dtype
        )  # Ensure dtype

        return state_array[..., np.newaxis]  # Add channel dim: (H, W) -> (H, W, 1)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
        super().close()

    @property
    def unwrapped(self) -> "DefeatRoachesEnv":
        return self

    def render(self, mode="human") -> Any:
        if mode == "human" and hasattr(self._env, "visualize") and self._env.visualize:
            return
        elif mode == "rgb_array":
            if (
                self._pysc2_timestep
                and "rgb_screen" in self._pysc2_timestep.observation
            ):
                return self._pysc2_timestep.observation["rgb_screen"]
            return np.zeros((self.screen_size, self.screen_size, 3), dtype=np.uint8)
        super().render(mode=mode)
