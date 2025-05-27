import numpy as np
import pandas as pd
import torch
import random
import pickle
import os

from agents.abstractAgent import AbstractAgent


class QLearningAgent(AbstractAgent):
    """
    Q Learning RL agent that produces actions based on the highest Q value, computes and saves sa-pairs in a Q table
    """

    def __init__(
        self,
        state_shape,
        action_shape,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.1,
    ):

        super().__init__(state_shape, action_shape)

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # TODO: create the Q table e.g. as pandas.DataFrame or 2D numpy array
        # Q = [x,y,actions]
        print(state_shape)
        self.Q_table = torch.zeros((64, 64, len(action_shape)))
        print(self.Q_table)
        print(self.Q_table.size())

    def random_action(self):
        return np.random.randint(self.Q_table.shape[2])

    def q_of_state(self, state: np.ndarray) -> torch.Tensor:
        return self.Q_table[state[0], state[1]]

    def get_action(self, state):
        # Explore: choose random action based on epsilon probability
        if random.random() < self.epsilon:
            return self.random_action()

        # Exploit: choose best action based on Q values
        q_values = self.q_of_state(state)
        best_action = q_values.argmax().item()

        # If all Q values are 0, choose a random action
        if q_values[best_action] == 0:
            return self.random_action()

        return best_action

    def update(self, state, action, reward, next_state, done):
        epsilon = None
        # update q-table
        self.Q_table[state[0], state[1], action] += self.learning_rate * (
            reward
            + self.discount_factor
            * torch.max(self.Q_table[next_state[0], next_state[1]])
            - self.Q_table[state[0], state[1], action]
        )
        # update epsilon
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            epsilon = self.epsilon
            print(self.epsilon)

        return epsilon

    def save_model(self, path, filename="qlagent"):
        os.makedirs(path, exist_ok=True)
        # Save directly as PyTorch tensor
        torch.save(self.q_table, os.path.join(path, f"{filename}.pt"))

    @classmethod
    def load_model(cls, path, filename="qlagent"):
        pt_path = os.path.join(path, f"{filename}.pt")

        # Create an instance of the class
        instance = cls(state_shape=(0,), action_shape=())  # Temporary values

        # Load the PyTorch tensor
        instance.q_table = torch.tensor(torch.load(pt_path, weights_only=False))
        return instance
