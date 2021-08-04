import gym
import gym_maze
from sys import exit
from math import log10
from time import sleep
from random import random
from numpy import prod, zeros, ones, amax, argmax, ndarray, save, load


class QLearningAgent:
    def __init__(self, init_lr, init_epsilon, min_lr, min_epsilon, gamma, action_space, observation_space, show_time,
                 render_maze, num_episodes, max_time_step):
        self.lr = init_lr
        self.min_lr = min_lr
        self.epsilon = init_epsilon
        self.min_epsilon = min_epsilon
        self.action_space = action_space
        self.num_actions = action_space.n
        self.gamma = gamma
        self.show_time = show_time
        self.render_maze = render_maze
        self.num_episodes = num_episodes
        self.max_time_step = max_time_step
        self.state_bounds = list(zip(observation_space.low, observation_space.high))
        self.maze_size = tuple((observation_space.high + ones(observation_space.shape)).astype(int))
        # hyper-parameter for learning rate and explore rate
        self.decay_factor = prod(self.maze_size, dtype=float) / 8.0
        # initialize q_table with zero matrix
        self.q_table = zeros(self.maze_size + (self.num_actions,), dtype=float)

    def update_explore_rate(self, i_episode: int) -> float:
        return max(self.min_epsilon, min(0.8, 1.0 - log10(i_episode / self.decay_factor)))

    def update_learning_rate(self, i_episode: int) -> float:
        return max(self.min_lr, min(0.8, 1.0 - log10(i_episode / self.decay_factor)))

    def choose_action(self, state: tuple) -> int:
        """ to explore environment we apply greedy-epsilon algorithm to select a action. """
        # case1: select a random action.
        if random() < self.epsilon:
            action = self.action_space.sample()
        # case2: select an action with the highest q value.
        else:
            action = int(argmax(self.q_table[state]))  # cast numpy.int64 to int
        return action

    def state_to_bucket(self, state: ndarray) -> tuple:
        bucket_indice = []
        for i in range(len(state)):
            if state[i] <= self.state_bounds[i][0]:
                bucket_index = 0
            elif state[i] >= self.state_bounds[i][1]:
                bucket_index = self.maze_size[i] - 1
            else:
                # mapping the state bounds to the bucket array
                bound_width = self.state_bounds[i][1] - self.state_bounds[i][0]
                offset = (self.maze_size[i] - 1) * self.state_bounds[i][0] / bound_width
                scaling = (self.maze_size[i] - 1) / bound_width
                bucket_index = int(round(scaling * state[i] - offset))
            bucket_indice.append(bucket_index)
        return tuple(bucket_indice)

    def play(self, episodes: int, update_lr_and_epsilon: bool = True):
        # render tha maze
        env.render()
        for episode in range(1, episodes + 1):
            total_reward = 0
            obv_t = env.reset()                    # reset the environment.
            state_t = self.state_to_bucket(obv_t)  # initialize state.
            for time_step in range(1, self.max_time_step + 1):
                # select an action.
                action_t = self.choose_action(state_t)

                # execute the action and get feedback by env.
                obv_next_t, reward_t, done, _ = env.step(action_t)

                # observe the result.
                state_next_t = self.state_to_bucket(obv_next_t)
                total_reward += reward_t

                # update the Q-table.
                real_q = self.q_table[state_t + (action_t,)]
                evaluated_q = reward_t + self.gamma * amax(self.q_table[state_next_t])
                self.q_table[state_t + (action_t,)] += self.lr * (evaluated_q - real_q)

                # update state for the next iteration.
                state_t = state_next_t

                # render tha maze.
                if self.render_maze:
                    env.render()
                    sleep(self.show_time)

                if env.is_game_over():
                    exit()

                if done:
                    msg = f'Episode {episode} finished after {time_step} time-steps with total reward = {total_reward:f}.'
                    print(msg)
                    break

                elif time_step >= self.max_time_step:
                    msg = f'Episode {episode} timed out at {time_step} time-steps with total reward = {total_reward:f}.'
                    print(msg)

            if update_lr_and_epsilon:
                self.lr = self.update_learning_rate(i_episode=episode)
                self.epsilon = self.update_explore_rate(i_episode=episode)

    def train_and_save_model(self):
        self.play(episodes=self.num_episodes, update_lr_and_epsilon=True)
        save('./saved_model/saved_q_table', self.q_table)

    def load_saved_model_and_play(self):
        self.lr = self.min_lr
        self.epsilon = self.min_epsilon
        self.q_table = load('./saved_model/saved_q_table.npy')
        self.play(episodes=10, update_lr_and_epsilon=False)


if __name__ == "__main__":

    env = gym.make("maze-sample-10x10-v0")  # maze-random-10x10-v0

    # hyper-parameters for our agent.
    params = {
        # # environment params.
        'observation_space': env.observation_space,
        'action_space': env.action_space,

        # # learning related params.
        'gamma': 0.9,          # discount factor for agent's td-error.
        'init_lr': 0.8,        # initialize learning rate.
        'init_epsilon': 0.8,   # initialize explore rate.
        'min_lr': 0.2,         # the minima of learning rate during entire learning process.
        'min_epsilon': 0.001,  # the minima of explore rate during entire learning process.

        # # main function related params.
        'show_time': 0.08,     # the showing time of circle, modify this value can shorten the program execution time.
        'render_maze': True,
        'num_episodes': 180,   # the number of episode.
        'max_time_step': 10000
    }
    q = QLearningAgent(**params)
    q.train_and_save_model()

    # # If use random maze environment, load_saved_model_and_play() will fail,
    # # because each reload of the maze environment will be different from the last time.
    q.load_saved_model_and_play()
