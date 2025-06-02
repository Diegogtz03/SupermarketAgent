import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, epsilon=0.8, epsilon_min=0.05, epsilon_decay=0.995, gamma=0.99, lr=1e-3, memory_size=10000, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size

        self.memory = deque(maxlen=memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state, valid_actions=None):
        if valid_actions is not None:
            valid_indices = [i for i, v in enumerate(valid_actions) if v]
            
            # Guided exploration: increase chance of promo actions when available
            promo_indices = [i for i in valid_indices if i >= len(valid_actions) - len([v for v in valid_actions if not v]) and i < len(valid_actions) - 1]
            
            if np.random.rand() < self.epsilon:
                # 70% chance to pick promo action if available, 30% any valid action
                if promo_indices and np.random.rand() < 0.7:
                    return np.random.choice(promo_indices)
                else:
                    return np.random.choice(valid_indices)
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor).cpu().numpy().flatten()
            # Mask invalid actions
            q_values[~np.array(valid_actions)] = -np.inf
            return int(np.argmax(q_values))
        else:
            # Fallback: standard epsilon-greedy
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.action_dim)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return q_values.argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.model(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1, keepdim=True)[0]
            target = rewards + self.gamma * next_q_values * (1 - dones)
        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _promo_available(self, promo):
        # This method should be implemented based on your specific promo logic
        # For now, we'll use a placeholder return
        return True

    def step(self):
        
        return

    def _calculate_reward(self):
        total = 0
        for pid in self.basket:
            price = float(self.products.loc[self.products['product_id'] == pid, 'price'].iloc[0])
            for promo in self.applied_promos:
                if promo['get'] == pid:
                    price *= promo['discount']
            total += price
        use_value_bonus = sum(
            1 for pid in self.basket if int(self.products.loc[self.products['product_id'] == pid, 'use_value'].iloc[0]) > 20
        )
        promo_bonus = 10 * len(self.applied_promos)  # <-- Strong bonus for each promo used
        return total + use_value_bonus + promo_bonus

    def print_promo_availability(self):
        for i, promo in enumerate(self.promotions):
            if self._promo_available(promo):
                print(f"Promo {i} available! Take action {self.n_products + i} to apply.") 