import abc


class AbstractAgent(metaclass=abc.ABCMeta):
    """
    Abstract class defining the structure and behavior of an RL agent.
    """

    def __init__(self, state_shape, action_shape):
        self.state_shape = state_shape
        self.action_shape = action_shape

    @abc.abstractmethod
    def get_action(self, state):
        """
        Select an action based on the current state.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, state, action, reward, next_state, next_action, done):
        """
        Update the parameters based on the observed reward and transition to the next state.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, state, action, reward, next_state, done):
        """
        Update the parameters based on the observed reward and transition to the next state.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, path, filename):
        """
        Serializes and saves the model as a file
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def load_model(cls, path, filename):
        """
        Loads the model from a file
        """
        raise NotImplementedError
