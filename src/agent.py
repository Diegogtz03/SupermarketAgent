import numpy as np

class QLearningAgent:
    def __init__(self, n_states, n_actions, epsilon=0.5, alpha=0.5, gamma=0.9):
        self.q_table = np.zeros((n_states, n_actions))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def select_action(self, state_idx):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.q_table.shape[1])
        return np.argmax(self.q_table[state_idx])

    def update(self, state_idx, action, reward, next_state_idx, done):
        best_next = np.max(self.q_table[next_state_idx])
        target = reward if done else reward + self.gamma * best_next
        self.q_table[state_idx, action] += self.alpha * (target - self.q_table[state_idx, action]) 