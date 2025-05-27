import numpy as np
import os
from agents.abstractAgent import AbstractAgent
from env.env_discrete import ACTION_DIRECTION


class RandomAgent(AbstractAgent):
    """
    Random agent that keeps its history
    """

    def __init__(self, state_shape, action_shape):
        super().__init__(state_shape, action_shape)

        print(state_shape)
        self.state_range = 20
        self.history = ""

    def get_action(self, state):
        self.history += f"From State {state}"
        action = np.random.randint(8)
        self.history += f"Move {ACTION_DIRECTION[action]}\n"
        return action

    def update(self, state, action, reward, next_state, done):
        pass

    def save_model(self, path, filename="history.txt"):
        with open(os.path.join(path, filename), "w") as f:
            f.write(self.history)

    def load_model(self, path, filename):
        raise NotImplementedError("Error: Random Agent is stateless!")
