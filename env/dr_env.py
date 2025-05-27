import datetime
import gym
import numpy as np
import os
import sys
from gym.spaces import Discrete, Box
from pysc2.env import sc2_env
from pysc2.lib import actions, features
from torch.utils.tensorboard import SummaryWriter


class DefeatRoachesEnv(gym.Env):
    def __init__(
        self,
        screen_size=64,
        step_mul=8,
        is_visualize=False,
        reselect_army_each_nstep=5,
        tensorboard_dir=None,
        tensorboard_prefix="rainbow",
    ):
        super(DefeatRoachesEnv, self).__init__()

        self.screen_size = screen_size
        self.reselect_army_each_nstep = reselect_army_each_nstep

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

        self.action_space = Discrete(
            self.screen_size**2
        )  # https://github.com/pekaalto/sc2atari/blob/master/sc2/sc2toatari.py#L39
        self.observation_space = Box(
            low=0,
            high=features.SCREEN_FEATURES.player_relative.scale,
            shape=[self.screen_size, self.screen_size, 1],
        )

        self.agg_n_episodes = 50
        self.rolling_episode_score = np.zeros(self.agg_n_episodes, dtype=np.float32)
        self.step_counter = 0
        self.episode_counter = 0
        self.total_score = 0

        self.tensorboard_dir = tensorboard_dir
        if self.tensorboard_dir is not None:
            self.tb_path = os.path.join(
                os.path.abspath(tensorboard_dir),
                f'{tensorboard_prefix}_{datetime.datetime.now().strftime("%y%m%d_%H%M")}',
            )
            self.writer = SummaryWriter(self.tb_path)

    def reset(self):
        self.attack_action_id = [
            k for k in actions.FUNCTIONS if k.name == "Attack_screen"
        ][0].id
        self.steps = 0
        self._pysc2_timestep = self._env.reset()[0]
        return self._get_state()

    def step(self, action):
        # reselect each N steps
        if (
            self.reselect_army_each_nstep > 0
            and self.steps % self.reselect_army_each_nstep == 0
        ):
            self._select_army()

        self._pysc2_timestep = self._pysc2_step(action)
        self.steps += 1

        # mdp
        state = self._get_state()
        reward = self._pysc2_timestep.reward
        done = self._pysc2_timestep.last()
        if done:
            self._summarize_episode()
        return state, reward, done, {}

    def _pysc2_step(self, action):
        coords = np.unravel_index(action, (self.screen_size,) * 2)
        # in pysc2 the coordinates are reversed
        action = [actions.FunctionCall(self.attack_action_id, [[0], coords[::-1]])]
        try:
            timestep = self._env.step(action)[0]
        except ValueError:
            # if attack is not available, select army and try again on the next step
            self._select_army()
            timestep = self._env.step(action)[0]
        return timestep

    def _select_army(self):
        self._env.step([actions.FunctionCall(actions.FUNCTIONS.select_army.id, [[0]])])

    def _get_state(self):
        return self._pysc2_timestep.observation["feature_screen"][
            features.SCREEN_FEATURES.player_relative.index
        ][..., np.newaxis]

    def _summarize_episode(self):
        episode_score = self._pysc2_timestep.observation["score_cumulative"][0]
        self.rolling_episode_score[self.episode_counter % self.agg_n_episodes] = (
            episode_score
        )
        self.episode_counter += 1
        self.total_score += episode_score
        r = self.rolling_episode_score[: min(self.episode_counter, self.agg_n_episodes)]
        print(
            f"episode {self.episode_counter}, score: {episode_score}, - avg {round(r.mean(), 3)}, min {round(r.min(), 3)}, max {round(r.max(),3)}"
        )
        if self.tensorboard_dir is not None:
            self.writer.add_scalar(
                "episodic_score", episode_score, self.episode_counter
            )
            self.writer.add_scalar(
                "total_score", self.total_score, self.episode_counter
            )
        sys.stdout.flush()

    def close(self):
        if self._env is not None:
            self._env.close()
        super().close()

    @property
    def state_shape(self):
        return self.observation_space.shape

    @property
    def action_shape(self):
        return (self.action_space.n,)
