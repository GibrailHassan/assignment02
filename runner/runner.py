import datetime
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class Runner:
    def __init__(
        self,
        agent,
        env,
        is_training=True,
        save_model_each_episode_num=5,
        tensorboard_log_dir="./logs",
        model_save_dir="./models",
    ):
        self.agent = agent
        self.env = env

        self.is_training = is_training
        self.total_score = 0
        self.episodic_score = 0
        self.episode = 1
        self.sliding_window = [-1] * 10
        self.curr_epsilon = 1.0

        self.is_saving_model_during_training = (
            is_training and save_model_each_episode_num > 0 and model_save_dir
        )
        if self.is_saving_model_during_training:
            self.models_path = os.path.join(
                os.path.abspath(model_save_dir),
                f'{datetime.datetime.now().strftime("%y%m%d_%H%M")}'
                + "_"
                + type(agent).__name__,
            )
        self.save_model_each_episode_num = save_model_each_episode_num

        self.tensorboard_log_dir = tensorboard_log_dir
        if tensorboard_log_dir:
            self.tb_path = os.path.join(
                os.path.abspath(tensorboard_log_dir),
                f'{datetime.datetime.now().strftime("%y%m%d_%H%M")}'
                + ("_train_" if self.is_training else "_eval_")
                + type(agent).__name__,
            )
            self.writer = SummaryWriter(self.tb_path)

    def summarize(self):
        # save the model and tensorboard log
        if self.tensorboard_log_dir:
            self.writer.add_scalar("episodic_score", self.episodic_score, self.episode)
            self.writer.add_scalar("total_score", self.total_score, self.episode)
            # mean reward over 10 episodes
            if self.sliding_window[0] != -1:
                self.writer.add_scalar(
                    "mean_score", np.mean(self.sliding_window), self.episode
                )
                print(
                    f"Finished episode {self.episode}, mean score {np.mean(self.sliding_window)}, list: {self.sliding_window}"
                )
                self.sliding_window = [-1] * 10
            # epsilon
            self.writer.add_scalar("epsilon", self.curr_epsilon, self.episode)
        if (
            self.is_saving_model_during_training
            and self.episode % self.save_model_each_episode_num == 0
        ):
            self.agent.save_model(self.models_path)
        # print score
        self.total_score += self.episodic_score
        print(
            f"Finished episode {self.episode}, episodic score {self.episodic_score}, total score {self.total_score} "
        )
        self.episode += 1
        self.sliding_window.pop(0)
        self.sliding_window.append(self.episodic_score)
        self.episodic_score = 0

    def run(self, episodes):
        for _ in range(episodes):
            # get initial state
            state = self.env.reset()
            next_state = None
            done = False
            while not done:
                # get action from the agent
                action = self.agent.get_action(state)
                # get envirionmental response for the performed action
                next_state, reward, done, _ = self.env.step(action)
                # update the agent with the new experience
                if self.is_training:
                    self.agent.update(state, action, reward, next_state, done)
                # update variables
                state = next_state
                self.episodic_score += reward
            self.summarize()
        self.env.close()

    def run_ql(self, episodes):
        for _ in range(episodes):
            # get initial state
            state = self.env.reset()
            next_state = None
            done = False
            while not done:
                # get action from the agent
                action = self.agent.get_action(state)
                # get envirionmental response for the performed action
                next_state, reward, done, _ = self.env.step(action)
                # update the agent with the new experience
                if self.is_training:
                    epsilon = self.agent.update(state, action, reward, next_state, done)
                    self.curr_epsilon = (
                        epsilon if epsilon != None else self.curr_epsilon
                    )
                # update variables
                state = next_state
                self.episodic_score += reward
            self.summarize()
        self.env.close()

    def run_dqn(self, episodes, c):
        step = 0
        for e in range(episodes):
            # get initial state
            state = self.env.reset()
            next_state = None
            done = False
            while not done:
                # get action from the agent
                action = self.agent.get_action(state)
                # get envirionmental response for the performed action
                next_state, reward, done, _ = self.env.step(action)
                # update the agent with the new experience
                if self.is_training:
                    epsilon = self.agent.update(state, action, reward, next_state, done)
                    self.curr_epsilon = (
                        epsilon if epsilon != None else self.curr_epsilon
                    )
                # update variables
                state = next_state
                self.episodic_score += reward
                if step == c:
                    step = 0
                    self.agent.reset_target_nn()
                else:
                    step += 1
            self.summarize()
        self.env.close()

    def run_sarsa(self, episodes):
        for _ in range(episodes):
            # get initial state
            state = self.env.reset()
            # get action from the agent
            action = self.agent.get_action(state)
            next_state = None
            done = False
            while not done:
                next_state, reward, done, _ = self.env.step(action)
                # get action from the agent
                next_action = self.agent.get_action(next_state)
                # update the agent with the new experience
                if self.is_training:
                    epsilon = self.agent.update(
                        state, action, reward, next_state, next_action, done
                    )
                    self.curr_epsilon = (
                        epsilon if epsilon != None else self.curr_epsilon
                    )
                # update variables
                state = next_state
                action = next_action
                self.episodic_score += reward
            self.summarize()
        self.env.close()

    def run_basic(self, episodes):
        # this is run method ONLY for the basic agent
        for _ in range(episodes):
            self.env.reset()
            done = False
            while not done:
                # pay attention: the basic agent consumes the whole environment to calculate the best action and ignores the states
                action = self.agent.get_action(self.env)
                _, reward, done, _ = self.env.step(action)
                self.episodic_score += reward
            self.summarize()
        self.env.close()
