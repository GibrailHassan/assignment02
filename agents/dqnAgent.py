import numpy as np
import os
import torch
import random
from collections import deque, namedtuple
from torch import nn

from agents.abstractAgent import AbstractAgent
from agents.NN_model import Model, init_weights

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class DQNAgent(AbstractAgent):
    """
    A DQN model that uses neural networks to approximate the value function for Q-Learning.
    """

    def __init__(
        self,
        state_shape,
        action_shape,
        batch_size=32,
        learning_rate=0.0005,
        discount_factor=0.97,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.0335,
        net_arch=[64, 64],
        target_update_freq=10,
        memory_capacity=10000,
        learn_after_steps=6000,
    ):
        super().__init__(state_shape, action_shape)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # TODO: Stuff is missing here
        self.state_shape = state_shape
        self.action_shape = action_shape

        # setup online neural network
        # print(f"len of action shape: {len(self.state_shape)}")
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.online_nn = Model(output=len(self.action_shape)).to(device=self.device)
        self.online_nn.apply(init_weights)
        self.opt = torch.optim.AdamW(
            self.online_nn.parameters(), lr=self.learning_rate, weight_decay=0.01
        )

        # setup target neural network
        self.target_nn = Model(output=len(self.action_shape)).to(device=self.device)
        self.target_nn.load_state_dict(self.online_nn.state_dict())

        # setup memory
        self.memory = ReplayMemory(memory_capacity, self.batch_size)

    def random_action(self):
        return np.random.randint(len(self.action_shape))

    def get_action(self, state):
        # explore: choose random action based on epsilon probability
        if random.random() < self.epsilon:
            return self.random_action()

        # exploit: select max from online nn output
        input = torch.as_tensor([float(s) for s in state])
        prediction = self.online_nn.fw(input)
        return np.argmax(prediction.tolist())

    def update(self, state, action, reward, next_state, done):
        epsilon = None
        # exp replay
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) >= self.batch_size:
            sample = self.memory.sample()
            batch = Transition(*zip(*sample))
            states = torch.tensor(batch.state, dtype=torch.float32)  # / 255.0
            actions = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1)
            rewards = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1)
            next_states = torch.tensor(batch.next_state, dtype=torch.float32)  # / 255.0
            dones = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1)

            # print(f"state batch: {states} | action batch: {actions} | reward batch: {rewards} | next state: {next_states}")

            # next action
            # target_input = torch.as_tensor(sample)
            with torch.no_grad():
                target_prediction, _ = self.target_nn.fw(next_states).max(1)
                y = rewards.flatten() + self.discount_factor * target_prediction * (
                    1 - dones.flatten()
                )

            # online nn
            # online_input = torch.as_tensor(sample)
            online_prediction = self.online_nn.fw(states).gather(1, actions).squeeze()

            # backpropagation
            ce_loss = nn.MSELoss()
            # print(f"y: {y}, online pred: {online_prediction}")
            loss = ce_loss(y, online_prediction)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # update epsilon
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            epsilon = self.epsilon
            print(self.epsilon)

        return epsilon

    def update_epsilon(self, step):
        if step == 500:
            self.epsilon = 0.2
            print(f"epsilon: {self.epsilon}")
        elif step == 800:
            self.epsilon = 0.0
            print(f"epsilon: {self.epsilon}")
        return self.epsilon

    def reset_target_nn(self):
        # print("load online into target")
        self.target_nn.load_state_dict(self.online_nn.state_dict())

    def save_model(self, path, filename="dqn.pt"):
        # TODO: Save the learnable parameters of your model and the hyperparameters
        # torch.save should be helpful for this :)
        pass

    @classmethod
    def load_model(
        cls, path, filename="dqn.pt", reset_timesteps=False, load_memory=True
    ):
        # TODO: Load the learnable parameters of your model and the hyperparameters
        # torch.load should be helpful for this :)
        # Afterwards you have to instantiated a DQN agent with those parameters
        pass


class ReplayMemory:
    """
    Replay memory for off-policy agents: a simple buffer and a PER version
    """

    def __init__(
        self,
        capacity,
        batch_size,
        is_prioritized=False,
        alpha=0.6,
        beta=0.4,
        beta_annealing=0.9999,
    ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = deque(maxlen=self.capacity)

    def add(self, state, action, reward, next_state, done):
        # add experience to memory
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self):
        # returns a random batch from the experiences
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)
