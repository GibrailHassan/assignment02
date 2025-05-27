"""
This module defines the DQNAgent class, which implements the Deep Q-Network
algorithm. It uses a neural network to approximate the Q-value function and
employs techniques like experience replay and a target network to stabilize
learning.
"""

import numpy as np
import os
import torch
import random
from collections import deque, namedtuple
from torch import nn

from agents.abstractAgent import AbstractAgent
from agents.NN_model import Model, init_weights

# A named tuple to represent a single transition in the environment.
# (state, action, reward, next_state, done_flag)
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class DQNAgent(AbstractAgent):
    """
    A Deep Q-Network (DQN) agent.

    This agent learns an optimal policy by estimating the Q-values (action-value function)
    using a neural network. It utilizes an experience replay buffer to store and sample
    past transitions, and a separate target network to stabilize the learning process.
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
        target_update_freq=10, # Not directly used in this version's reset_target_nn, controlled by runner
        memory_capacity=10000,
        learn_after_steps=6000, # Not directly used, learning starts when memory > batch_size
    ):
        """
        Initializes the DQNAgent.

        Args:
            state_shape (tuple): The shape of the state space.
            action_shape (tuple or int): The shape or size of the action space.
            batch_size (int): The size of the mini-batch sampled from the replay memory.
            learning_rate (float): The learning rate for the optimizer.
            discount_factor (float): The discount factor (gamma) for future rewards.
            epsilon (float): The initial exploration rate for the epsilon-greedy policy.
            epsilon_decay (float): The rate at which epsilon decays after each episode.
            epsilon_min (float): The minimum value epsilon can reach.
            net_arch (list): A list defining the architecture of the neural network's
                             hidden layers (e.g., [64, 64] for two hidden layers
                             with 64 units each). The DQNAgent currently uses a fixed
                             architecture defined in NN_model.py, so this arg is not directly used.
            target_update_freq (int): Frequency (in steps or episodes, depending on runner)
                                      at which the target network is updated.
            memory_capacity (int): The maximum capacity of the replay memory.
            learn_after_steps (int): Number of steps to collect before starting to learn.
                                     Learning actually starts when memory size >= batch_size.
        """
        super().__init__(state_shape, action_shape)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # self.net_arch = net_arch # Not used as NN_model.py has a fixed architecture for now
        # self.target_update_freq = target_update_freq # Handled by the runner
        # self.learn_after_steps = learn_after_steps # Learning starts when memory is sufficient

        self.state_shape = state_shape
        self.action_shape = action_shape

        # Determine the device to use for PyTorch (GPU if available, otherwise CPU).
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # --- Online Neural Network (the one being actively trained) ---
        # The output size of the network is the number of possible actions.
        self.online_nn = Model(output=len(self.action_shape)).to(device=self.device)
        self.online_nn.apply(init_weights) # Initialize network weights.
        # Optimizer for the online network (AdamW is a common choice).
        self.opt = torch.optim.AdamW(
            self.online_nn.parameters(), lr=self.learning_rate, weight_decay=0.01
        )

        # --- Target Neural Network (provides stable targets for Q-value updates) ---
        # Also has the same architecture as the online network.
        self.target_nn = Model(output=len(self.action_shape)).to(device=self.device)
        # Initialize target network with the same weights as the online network.
        self.target_nn.load_state_dict(self.online_nn.state_dict())
        self.target_nn.eval() # Set target network to evaluation mode (no gradients needed).

        # --- Replay Memory ---
        # Stores experiences (transitions) for later learning.
        self.memory = ReplayMemory(memory_capacity, self.batch_size)

    def random_action(self):
        """Selects a random action from the action space."""
        return np.random.randint(len(self.action_shape))

    def get_action(self, state):
        """
        Selects an action using an epsilon-greedy policy.
        With probability epsilon, a random action is chosen (exploration).
        Otherwise, the action with the highest Q-value (estimated by the online network)
        is chosen (exploitation).

        Args:
            state: The current state observation.

        Returns:
            int: The selected action.
        """
        # Exploration: choose a random action with probability epsilon.
        if random.random() < self.epsilon:
            return self.random_action()

        # Exploitation: select the action with the highest Q-value from the online network.
        # Convert state to a PyTorch tensor and send to the appropriate device.
        # Ensure state is float for the network.
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0) # Add batch dimension
        with torch.no_grad(): # No need to calculate gradients for action selection.
            q_values = self.online_nn.fw(state_tensor)
        return torch.argmax(q_values).item() # Get the action with the max Q-value.

    def update(self, state, action, reward, next_state, done):
        """
        Updates the online network using a batch of experiences from the replay memory.
        This involves calculating the target Q-values using the target network and
        minimizing the Mean Squared Error (MSE) loss between the predicted Q-values
        (from the online network) and the target Q-values.

        Args:
            state: The state from which the action was taken.
            action: The action taken.
            reward (float): The reward received.
            next_state: The state transitioned to.
            done (bool): Whether the episode has terminated.

        Returns:
            float or None: The current epsilon value if the episode is done, otherwise None.
        """
        epsilon = None
        # Add the current transition to the replay memory.
        self.memory.add(state, action, reward, next_state, done)

        # Start learning only if there are enough samples in the memory.
        if len(self.memory) >= self.batch_size:
            # Sample a mini-batch of transitions from the replay memory.
            transitions = self.memory.sample()
            # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for details).
            # This converts batch-array of Transitions to Transition of batch-arrays.
            batch = Transition(*zip(*transitions))

            # Convert batch data to PyTorch tensors and move to the device.
            states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
            actions = torch.tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1) # Unsqueeze to make it a column vector.
            rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
            next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
            dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1) # Boolean to float for multiplication.

            # --- Calculate Target Q-values ---
            # Q_target = r + gamma * max_a'(Q_target_net(s', a')) for non-terminal s'
            # Q_target = r for terminal s'
            with torch.no_grad():
                # Get the maximum Q-value for the next states from the target network.
                # .max(1) returns a tuple (values, indices), we only need values.
                next_q_values_target = self.target_nn.fw(next_states).max(1, keepdim=True)[0]
                # Calculate the target Q-value. If done is True, the future reward is 0.
                y = rewards + self.discount_factor * next_q_values_target * (1 - dones)

            # --- Calculate Predicted Q-values (from Online Network) ---
            # Get the Q-values for the actions taken in the batch from the online network.
            # .gather(1, actions) selects the Q-value corresponding to the action taken.
            online_prediction = self.online_nn.fw(states).gather(1, actions)

            # --- Compute Loss and Perform Optimization ---
            loss_fn = nn.MSELoss() # Mean Squared Error loss.
            loss = loss_fn(online_prediction, y)

            self.opt.zero_grad() # Clear previous gradients.
            loss.backward()      # Compute gradients.
            # torch.nn.utils.clip_grad_value_(self.online_nn.parameters(), 100) # Optional: gradient clipping
            self.opt.step()       # Update network weights.

        # --- Epsilon Decay ---
        # If the episode is done, decay epsilon.
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            epsilon = self.epsilon # Return the new epsilon value.
            # self._log_to_console(f"Epsilon updated to: {self.epsilon:.4f}") # Optional logging

        return epsilon # Return current epsilon (or None if not updated).

    def update_epsilon(self, step):
        """
        DEPRECATED or for specific scheduled epsilon updates.
        The primary epsilon update happens in the `update` method upon episode completion.
        This method seems to be for a fixed schedule of epsilon changes.

        Args:
            step (int): The current step or episode number (context dependent).

        Returns:
            float: The current epsilon value.
        """
        # This method provides a fixed schedule for epsilon updates,
        # which might override or complement the decay in the `update` method.
        # Consider if this is still needed or if the per-episode decay is sufficient.
        if step == 500: # Example: set epsilon to 0.2 at step 500.
            self.epsilon = 0.2
            print(f"Epsilon manually set to: {self.epsilon} at step {step}")
        elif step == 800: # Example: set epsilon to 0.0 at step 800.
            self.epsilon = 0.0
            print(f"Epsilon manually set to: {self.epsilon} at step {step}")
        return self.epsilon

    def reset_target_nn(self):
        """
        Copies the weights from the online network to the target network.
        This is typically done periodically to stabilize learning.
        """
        # print("Updating target network with online network weights.") # Optional logging
        self.target_nn.load_state_dict(self.online_nn.state_dict())

    def save_model(self, path, filename="dqn_agent.pt"):
        """
        Saves the learnable parameters (state_dict) of the online neural network
        and relevant hyperparameters.

        Args:
            path (str): The directory path to save the model.
            filename (str): The name for the saved model file. Defaults to "dqn_agent.pt".
        """
        os.makedirs(path, exist_ok=True) # Ensure the directory exists.
        model_filepath = os.path.join(path, filename)
        torch.save({
            'online_nn_state_dict': self.online_nn.state_dict(),
            'target_nn_state_dict': self.target_nn.state_dict(), # Good practice to save target too
            'optimizer_state_dict': self.opt.state_dict(),
            'epsilon': self.epsilon,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'state_shape': self.state_shape,
            'action_shape': self.action_shape,
            # Add any other hyperparameters you want to save
        }, model_filepath)
        print(f"DQNAgent model saved to {model_filepath}")

    @classmethod
    def load_model(
        cls, path, filename="dqn_agent.pt", eval_mode=True
    ):
        """
        Loads a saved DQNAgent model, including network weights and hyperparameters.

        Args:
            path (str): The directory path from where to load the model.
            filename (str): The name of the model file. Defaults to "dqn_agent.pt".
            eval_mode (bool): If True, sets the loaded agent's networks to evaluation mode
                              and potentially sets epsilon to its minimum for exploitation.
                              Defaults to True.

        Returns:
            DQNAgent: An instance of DQNAgent with loaded parameters.
        """
        model_filepath = os.path.join(path, filename)
        if not os.path.exists(model_filepath):
            raise FileNotFoundError(f"Model file not found at {model_filepath}")

        checkpoint = torch.load(model_filepath)

        # Create a new agent instance with the saved hyperparameters.
        # Ensure all necessary hyperparameters are present in the checkpoint.
        agent = cls(
            state_shape=checkpoint['state_shape'],
            action_shape=checkpoint['action_shape'],
            batch_size=checkpoint.get('batch_size', 32), # Provide defaults if not in older checkpoints
            learning_rate=checkpoint.get('learning_rate', 0.0005),
            discount_factor=checkpoint.get('discount_factor', 0.97),
            epsilon=checkpoint.get('epsilon', 1.0), # Load saved epsilon
            epsilon_decay=checkpoint.get('epsilon_decay', 0.995),
            epsilon_min=checkpoint.get('epsilon_min', 0.0335),
            # net_arch and other params if they were saved and used in __init__
        )

        agent.online_nn.load_state_dict(checkpoint['online_nn_state_dict'])
        if 'target_nn_state_dict' in checkpoint:
            agent.target_nn.load_state_dict(checkpoint['target_nn_state_dict'])
        else: # For older models that might not have saved target_nn separately
            agent.target_nn.load_state_dict(checkpoint['online_nn_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            agent.opt.load_state_dict(checkpoint['optimizer_state_dict'])

        if eval_mode:
            agent.online_nn.eval()
            agent.target_nn.eval()
            agent.epsilon = agent.epsilon_min # For pure exploitation during evaluation

        print(f"DQNAgent model loaded from {model_filepath}")
        return agent


class ReplayMemory:
    """
    A simple circular buffer for storing and sampling experiences (transitions)
    for off-policy reinforcement learning agents like DQN.
    """

    def __init__(
        self,
        capacity,
        batch_size,
        # Parameters for Prioritized Experience Replay (PER) - currently not implemented
        # is_prioritized=False,
        # alpha=0.6,
        # beta=0.4,
        # beta_annealing=0.9999,
    ):
        """
        Initializes the ReplayMemory.

        Args:
            capacity (int): The maximum number of transitions the memory can hold.
            batch_size (int): The number of transitions to sample in a batch.
            is_prioritized (bool): Whether to use Prioritized Experience Replay (Not implemented).
            alpha (float): PER hyperparameter (Not implemented).
            beta (float): PER hyperparameter (Not implemented).
            beta_annealing (float): PER hyperparameter (Not implemented).
        """
        self.capacity = capacity
        self.batch_size = batch_size
        # Using collections.deque for efficient appends and pops from both ends.
        self.memory = deque(maxlen=self.capacity)
        # self.is_prioritized = is_prioritized # PER flag
        # if self.is_prioritized:
        #     # Initialize structures for PER (e.g., sum tree)
        #     raise NotImplementedError("Prioritized Experience Replay is not yet implemented.")

    def add(self, state, action, reward, next_state, done):
        """
        Adds a new transition to the memory.

        Args:
            state: The state.
            action: The action taken.
            reward (float): The reward received.
            next_state: The next state.
            done (bool): Whether the episode terminated.
        """
        # Add the experience (transition) to the memory.
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self):
        """
        Samples a random mini-batch of transitions from the memory.

        Returns:
            list[Transition]: A list of randomly sampled transitions.
        """
        # Returns a random batch of experiences from the memory.
        # Ensure batch_size does not exceed the current number of items in memory.
        actual_batch_size = min(self.batch_size, len(self.memory))
        return random.sample(self.memory, actual_batch_size)

    def __len__(self):
        """
        Returns the current number of transitions stored in the memory.

        Returns:
            int: The current size of the memory.
        """
        return len(self.memory)
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
