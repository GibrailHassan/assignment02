import sys
from absl import flags

from agents.dqnAgent import DQNAgent
from env.env_discrete import MoveToBeaconDiscreteEnv
from runner.runner import Runner
import torch

# pysc2 routine, do not touch
FLAGS = flags.FLAGS
FLAGS(sys.argv)

env = MoveToBeaconDiscreteEnv(
    distance_discrete_range=10,
    distance=1,
    screen_size=32,
    step_mul=8,
    is_visualize=False,
)

agent = DQNAgent(state_shape=env.state_shape, action_shape=env.action_shape)

runner = Runner(agent=agent, env=env, is_training=True)

# key point: with basic agent it should be a special run_basic as it checks all actions to decide
runner.run_dqn(episodes=700, c=250)
