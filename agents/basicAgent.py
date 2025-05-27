from agents import AbstractAgent

class BasicAgent(AbstractAgent):
    """
    Basic dummy agent that selects the action based on the straightforward angle calculation, no RL
    """
    
    def __init__(self, state_shape, action_shape):
        super().__init__(state_shape, action_shape)

    def get_action(self, env):
        # go over all 8 actions to select the best one based by min absolute distance bw marine and beacon
        min_dist = 1e6
        best_action = -1
        print(env.action_shape)
        for action in env.action_shape:
            next_dist = env.roll_to_next_state(action)
            dist = ((next_dist[0] ** 2) + (next_dist[1] ** 2)) ** 0.5
            if dist < min_dist:
                min_dist = dist
                best_action = action
        print(best_action)
        return best_action

    def update(self, state, action, reward, next_state, done):
        pass
    
    def save_model(self, path, filename):
        pass
    
    @classmethod
    def load_model(cls, path, filename):
        pass
