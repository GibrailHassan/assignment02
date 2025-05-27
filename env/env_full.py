import math
import numpy as np
import gym
from pysc2.env import sc2_env
from pysc2.lib import actions, features

from env.utils import calc_target_position, calc_direction_and_distance_from_action, preprocess_channels

# pysc2 constants, do not touch
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_RELATIVE_SCALE = features.SCREEN_FEATURES.player_relative.scale
_PLAYER_FRIENDLY = 1 # marine
_FUNCTIONS = actions.FUNCTIONS

ACTION_DIRECTION = {
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
    def __init__(self, distance_range=8, screen_size=64, step_mul=8, is_visualize=False):
        self.screen_size = screen_size
        self.distance_range = distance_range
        self.distance_delta = math.floor(screen_size / distance_range)
        
        # pysc2 env
        self._env = sc2_env.SC2Env(
                map_name="MoveToBeacon",
                players=[sc2_env.Agent(sc2_env.Race.terran)],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=screen_size,
                                                           minimap=screen_size),
                    use_feature_units=True
                ),
                step_mul=step_mul,
                game_steps_per_episode=0,
                visualize=is_visualize
        )
        
        # action is represented as a tuple (direction, distance).
        # However DQNs and some PGs cannot work with MultiDiscrete spaces, so we need to flatten them to a Discrete space
        self.action_space = gym.spaces.Discrete(self.distance_range * len(ACTION_DIRECTION))
        self.observation_space = gym.spaces.Box(low=0, high=_PLAYER_RELATIVE_SCALE, shape=(2, screen_size, screen_size), dtype=np.int32)

    def step(self, action):
        self.last_action = action
        # calculate the target marine position depending on the selected action
        direction, distance = calc_direction_and_distance_from_action(action, self.distance_range, self.distance_delta)
        new_x, new_y = calc_target_position(self.marine_pos[0], self.marine_pos[1], ACTION_DIRECTION[direction], distance, self.screen_size)
        self.marine_pos = [new_x, new_y]
        
        # perform pysc2 action
        if _FUNCTIONS.Move_screen.id in self._obs.observation.available_actions:
            self._obs = self._env.step([_FUNCTIONS.Move_screen("now", self.marine_pos)])[0]
        else:
            self._obs = self._env.step([_FUNCTIONS.select_army("select")])[0]
        
        # MDP
        self.state = self._get_state()
        self.reward = self._obs.reward
        done = self._obs.last()
        return self.state, self.reward, done, {}

    def reset(self):
        # select the army and get the initial observation
        self._obs = self._env.step([_FUNCTIONS.select_army("select")])[0]
        
        # get the initial coordinates of a marine from the observation
        self._set_marine_position()
        
        # default values for MDP parameters
        self.last_action = None
        self.reward = 0
        self.state = self._get_state()
        return self.state

    def _get_state(self, channels=2):
        # get the state to a view (num_channels, screen_size, screen_size)
        state_size = self._obs.observation.feature_screen.shape
        state = np.ndarray(shape=(channels, state_size[1], state_size[2]))
        if channels == 17:
            state = preprocess_channels(self._obs)
        elif channels == 2:
            state[0] = self._obs.observation.feature_screen.player_relative
            state[1] = self._obs.observation.feature_screen.unit_density
        elif channels == 1:
            state[0] = self._obs.observation.feature_screen.unit_density
        return state

    def _set_marine_position(self):
        player_relative = self._obs.observation["feature_screen"][_PLAYER_RELATIVE]
        marine_y, marine_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
        self.marine_pos = np.mean(list(zip(marine_x, marine_y)), axis=0).round()

    def save_replay(self, replay_dir):
        if self._env is not None:
            self._env.save_replay(replay_dir)

    def close(self):
        if self._env is not None:
            self._env.close()
        super().close()

    @property
    def state_shape(self):
        # np array with a shape (2, screen_size, screen_size)
        return self.observation_space.shape

    @property
    def action_shape(self):
        # space.Discrete returns a shape (), so return an enhanced shape
        return (self.action_space.n,)