import sys
from absl import flags

from agents import DQNAgent
from env import DefeatRoachesEnv
from runner import Runner

# pysc2 routine, do not touch
FLAGS = flags.FLAGS
FLAGS(sys.argv)

env = DefeatRoachesEnv(
    screen_size=32,
    step_mul=8,
    is_visualize=True
)
print(f"State shape {env.state_shape}, action shape{env.action_shape}")

agent = DQNAgent(
    state_shape=env.state_shape,
    action_shape=env.action_shape,
    batch_size=128, 
    learning_rate=0.0025, 
    discount_factor=0.99, 
    epsilon=1.0, 
    epsilon_decay=0.9999, 
    epsilon_min=0.1,
    net_arch=[256, 256], 
    target_update_freq=50, 
    memory_capacity=10000,
    learn_after_steps=5000, 
)

runner = Runner(
    agent=agent,
    env=env,
    tensorboard_log_dir="",
    save_model_each_episode_num=0
)

runner.run(500)