import os
import pickle

import torch

from agents.qlAgent import QLearningAgent

    
def load_model(path, filename="qlagent"):
    # Load the Q table using pickle
    with open(os.path.join(path, f"{filename}.pkl"), "rb") as f:
        q_table = pickle.load(f)
    
    # Create an instance of the class and set the loaded Q table
    instance = QLearningAgent(state_shape=(2,), action_shape=range(8))
    instance.q_table = q_table
    return instance


# agent = load_model("models/250508_0748_QLearningAgent")
# agent.save_model("models/250508_0748_QLearningAgent")

instance = QLearningAgent.load_model("models/250508_0748_QLearningAgent")

print("hey")