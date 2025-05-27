import sys
from absl import flags

from agents.qlAgent import QLearningAgent
from env.env_discrete import MoveToBeaconDiscreteEnv
from runner.runner import Runner
import torch

# pysc2 routine, do not touch
FLAGS = flags.FLAGS
FLAGS(sys.argv)

# here define your run configs

env = MoveToBeaconDiscreteEnv(
    distance_discrete_range=10,
    distance=1,
    screen_size=32,
    step_mul=8,
    is_visualize=False,
)

model_path = "models/250508_0748_QLearningAgent"

agent = QLearningAgent.load_model(model_path)
agent.epsilon = agent.epsilon_min


runner = Runner(agent=agent, env=env, is_training=False)

runner.run_ql(100)
