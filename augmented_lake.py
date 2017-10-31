import numpy as np
import gym.envs.toy_text.frozen_lake as original

LEFT = original.LEFT
DOWN = original.DOWN
RIGHT = original.RIGHT
UP = original.UP

ACTION_STRINGS = {LEFT: "LEFT", DOWN: "DOWN", RIGHT: "RIGHT", UP: "UP"}
ACTION_DELTAS = {LEFT: (-1, 0), DOWN: (0, 1), RIGHT: (1, 0), UP: (0, -1)}

class AugmentedFrozenLake(original.FrozenLakeEnv):
    def print_map(self, prefix=''):
        for line in self.desc:
            print(prefix + ''.join([item.decode("utf-8") for item in line]))

    def arrange_on_grid(self, vector):
        return np.reshape(vector, newshape=(self.ncol, self.nrow))

    def position_from_state(self, state):
        x = state % self.ncol
        y = state // self.ncol
        return x, y

    def enforce_board_dimensions(self, x, y):
        x = max(min(x, self.ncol - 1), 0)
        y = max(min(y, self.nrow - 1), 0)
        return x, y

    def state_from_position(self, x, y):
        return x + (y * self.ncol)

    def naive_outcome(self, state, action):
        """ Where would I go if I were walking on solid ground? """
        x, y = self.position_from_state(state)
        delta_x, delta_y = ACTION_DELTAS[action]
        x_prime, y_prime = self.enforce_board_dimensions(x + delta_x, y + delta_y)
        return self.state_from_position(x_prime, y_prime)

    def make_naive_transitions(self):
        transitions = np.zeros(shape=(self.nS, self.nA, self.nS))
        for s in range(self.nS):
            for a in range(self.nA):
                transitions[s, a, self.naive_outcome(s, a)] = 1
        return transitions

    def make_rewards_tensor(self):
        rewards = np.zeros(shape=(self.nS, self.nA, self.nS))
        for state in range(self.nS):
            for action in range(self.nA):
                for _, next_state, reward, _ in self.P[state][action]:
                    rewards[state, action, next_state] = reward
        return rewards

    def make_real_transitions(self):
        transitions = np.zeros(shape=(self.nS, self.nA, self.nS))
        for state in range(self.nS):
            for action in range(self.nA):
                for probability, next_state, _, _ in self.P[state][action]:
                    transitions[state, action, next_state] = probability
        return transitions
