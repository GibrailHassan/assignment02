import numpy as np
import gym
from pysc2.env import sc2_env
from pysc2.lib import actions, features

from env.utils import discretize_distance, calc_target_position

# pysc2 constants, do not touch
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1 # marine
_PLAYER_NEUTRAL = 3  # beacon
_FUNCTIONS = actions.FUNCTIONS

# fix the actions for 8 directions
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

class MoveToBeaconDiscreteEnv(gym.Env):
    def __init__(self, distance_discrete_range=10, distance=64, screen_size=64, step_mul=8, is_visualize=False):
        self.distance_discrete_range = distance_discrete_range
        self.distance = distance
        self.screen_size = screen_size
        
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
        
        # two variables need to be redefined from the parent gym.Env class, however, we won't use them (so far)
        self.action_space = gym.spaces.Discrete(len(ACTION_DIRECTION))
        self.observation_space = gym.spaces.Box(low=-self.distance_discrete_range, high=self.distance_discrete_range, shape=(2,), dtype=np.int32)

    def step(self, action):
        self.last_action = action
        # calculate the target marine position depending on the selected direction
        new_x, new_y = calc_target_position(self.marine_pos[0], self.marine_pos[1], ACTION_DIRECTION[action], self.distance, self.screen_size)
        self.marine_pos = [new_x, new_y]
        
        # perform pysc2 action
        if _FUNCTIONS.Move_screen.id in self._obs.observation.available_actions:
            self._obs = self._env.step([_FUNCTIONS.Move_screen("now", self.marine_pos)])[0]
        else:
            self._obs = self._env.step([_FUNCTIONS.select_army("select")])[0]
        
        # get state
        self.state = self._get_state()
        
        # get reward
        self.reward = self._obs.reward
        if self.reward == 1:
            # a win: need to update a position of a new beacon
            self._set_beacon_position()
        
        # done?
        done = self._obs.last()
        return self.state, self.reward, done, {}

    def reset(self):
        # select the army and get the initial observation
        self._obs = self._env.step([_FUNCTIONS.select_army("select")])[0]
        
        # get the initial coordinates from the observation
        self._set_marine_position()
        self._set_beacon_position()
        
        # default values for MDP parameters
        self.last_action = None
        self.reward = 0
        self.state = self._get_state()
        return self.state
    
    def _set_marine_position(self):
        player_relative = self._obs.observation["feature_screen"][_PLAYER_RELATIVE]
        marine_y, marine_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
        self.marine_pos = np.mean(list(zip(marine_x, marine_y)), axis=0).round()

    def _set_beacon_position(self):
        player_relative = self._obs.observation["feature_screen"][_PLAYER_RELATIVE]
        beacon_y, beacon_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
        self.beacon_pos = np.mean(list(zip(beacon_x, beacon_y)), axis=0).round()
        
    def _get_state(self):
        # get x and y linear distances
        x_distance = self.beacon_pos[0] - self.marine_pos[0]
        y_distance = self.beacon_pos[1] - self.marine_pos[1]
        
        # discretize distances
        return np.array([discretize_distance(x_distance, self.screen_size, self.distance_discrete_range), 
                        discretize_distance(y_distance, self.screen_size, self.distance_discrete_range)], 
                        dtype=np.int32)
    
    def roll_to_next_state(self, action):
        # for basic agents only: without performing of action, just rolling to the next state w/o discretization to see what happens
        new_x, new_y = calc_target_position(self.marine_pos[0], self.marine_pos[1], ACTION_DIRECTION[action], self.distance, self.screen_size)
        x_distance = self.beacon_pos[0] - new_x
        y_distance = self.beacon_pos[1] - new_y
        return np.array([x_distance, y_distance], dtype=np.int32)
    
    def save_replay(self, replay_dir):
        if self._env is not None:
            self._env.save_replay(replay_dir)

    def close(self):
        if self._env is not None:
            self._env.close()
        super().close()

    @property
    def state_shape(self):
        # np array with a shape (2, ), i.e. [x, y]
        return self.observation_space.shape

    @property
    def action_shape(self):
        # space.Discrete returns a shape (), so return a range here
        return range(self.action_space.n)