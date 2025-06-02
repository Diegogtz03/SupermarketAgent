import pandas as pd
import numpy as np
import random
import os

class ShoppingEnv:
    def __init__(self, rules_csv="models/association_rules.csv", baskets_pkl="data/val.pkl", max_steps=10):
        self.rules = pd.read_csv(rules_csv)
        self.baskets = pd.read_pickle(baskets_pkl)["product_id"].tolist()
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.current_basket = random.choice(self.baskets).copy()
        # Start with half the basket (rounded down)
        if self.current_basket:
            n = max(1, len(self.current_basket) // 2)
            self.agent_basket = [int(pid) for pid in random.sample(self.current_basket, n)]
        else:
            self.agent_basket = []
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        # Observation: current agent basket (list of product ids)
        return self.agent_basket.copy()

    def get_available_bundles(self, top_n=10):
        print(f"Agent basket: {self.agent_basket}")
        # Return top N bundles (consequents) for current agent basket
        bundles = []
        for _, row in self.rules.iterrows():
            antecedents = eval(row["antecedents"])
            if set(antecedents).issubset(set(self.agent_basket)):
                consequents = eval(row["consequents"])
                bundles.append((consequents, row["confidence"]))
        bundles = sorted(bundles, key=lambda x: -x[1])[:top_n]
        return bundles

    def step(self, action):
        # Action: a bundle (list of product ids) to offer
        self.steps += 1
        reward = 0
        done = False
        
        for pid in action:
            if pid in self.current_basket and pid not in self.agent_basket:
                self.agent_basket.append(pid)
                reward += 1
        # Episode ends if all products in current basket are collected or max_steps reached
        if set(self.current_basket) == set(self.agent_basket) or self.steps >= self.max_steps:
            done = True
        return self._get_obs(), reward, done, {}

def run_simulation(n_episodes=100):
    env = ShoppingEnv()
    total_rewards = []
    total_steps = []
    for ep in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        steps = 0
        for _ in range(env.max_steps):
            bundles = env.get_available_bundles()
            if not bundles:
                break
            action = bundles[0][0]  # Take the highest-confidence bundle
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1
            if done:
                break
        total_rewards.append(episode_reward)
        total_steps.append(steps)
    print(f"Simulated {n_episodes} episodes.")
    print(f"Average reward per episode: {np.mean(total_rewards):.2f}")
    print(f"Average steps per episode: {np.mean(total_steps):.2f}")
    print(f"Completion rate (all products recommended): {np.mean([r == len(set(env.current_basket)) for r in total_rewards]):.2%}")
    return total_rewards, total_steps

# Example usage:
if __name__ == "__main__":
    env = ShoppingEnv()
    obs = env.reset()
    print("Initial basket:", obs)

    # Load product metadata
    products = pd.read_csv("data/products.csv")  # adjust path as needed
    id_to_name = dict(zip(products["product_id"], products["product_name"]))

    # Convert basket IDs to names
    basket_names = [id_to_name.get(int(pid), pid) for pid in env.agent_basket]
    print("Initial basket (names):", basket_names)

    for _ in range(10):
        bundles = env.get_available_bundles()
        if not bundles:
            print("No more bundles to offer.")
            break
        action = bundles[0][0]  # Take the highest-confidence bundle
        print(f"Offering bundle: {action}")
        obs, reward, done, _ = env.step(action)
        print(f"Reward: {reward}, New basket: {obs}")
        if done:
            print("Episode finished.")
            break

    run_simulation(100)
