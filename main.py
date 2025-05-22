import pandas as pd
import os
import pickle
from gensim.models import Word2Vec
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split
import numpy as np
from src.data_prep import get_products, get_promotions, load_instacart_data, get_basket_matrix, find_frequent_pairs, assign_use_value, create_promotions
from src.supermarket_env import SupermarketEnv
from src.agent import QLearningAgent
from src.state_indexer import StateIndexer

# Load data from Kaggle and modified products.csv with "life span"
def load_data():
  origin = "yasserh/instacart-online-grocery-basket-analysis-dataset"

  orders = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    origin,
    "orders.csv",
  )

  products = pd.read_csv("data/products.csv")

  departments = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    origin,
    "departments.csv",
  )

  aisles = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    origin,
    "aisles.csv",
  )

  order_products_prior = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    origin,
    "order_products__prior.csv",
  )

  order_products_train = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    origin,
    "order_products__train.csv",
  )

  return orders, products, departments, aisles, order_products_prior, order_products_train

# Join dataframes and clean data for further analysis
def preprocess_data(orders, products, departments, aisles, order_products_prior, order_products_train):
  merged_orders = orders.merge(order_products_prior, on='order_id', how='left')
  merged_orders = merged_orders.merge(products, on='product_id', how='left')
  merged_orders = merged_orders.merge(departments, on='department_id', how='left')
  merged_orders = merged_orders.merge(aisles, on='aisle_id', how='left')

  # Drop rows with missing values
  merged_orders = merged_orders[~merged_orders['eval_set'].isna()]

  seqs = (merged_orders.sort_values(["user_id", "order_number", "add_to_cart_order"])
          .groupby(["user_id", "order_id"])
          .agg({"product_id": list,
                "use_value": list,
                "order_hour_of_day":"first"}))

  with open("data/sequences.pkl", "wb") as f:
    pickle.dump(seqs, f)

  # --- user‑level split so orders from the same shopper never leak ---
  users = seqs.index.get_level_values(0).unique()
  u_train, u_tmp = train_test_split(users, test_size=0.30, random_state=42)
  u_val, u_test  = train_test_split(u_tmp,  test_size=0.50, random_state=42)
  
  seqs.loc[u_train].to_pickle(os.path.join("models", "train.pkl"))
  seqs.loc[u_val]  .to_pickle(os.path.join("models", "val.pkl"))
  seqs.loc[u_test] .to_pickle(os.path.join("models", "test.pkl"))

  print(seqs.head())

  return merged_orders

def main():
    # 1. Load and prepare data
    orders, products, departments, aisles, order_products_prior, order_products_train = load_data()

    # --- Efficient filtering for memory ---
    TOP_N_PRODUCTS = 10
    MAX_ORDERS = 1000

    # Find top N most popular products
    top_products = (order_products_prior['product_id']
                    .value_counts()
                    .head(TOP_N_PRODUCTS)
                    .index.tolist())
    products = products[products['product_id'].isin(top_products)].reset_index(drop=True)

    # Filter order_products_prior to only these products
    order_products_prior = order_products_prior[order_products_prior['product_id'].isin(top_products)]

    # Find orders that contain only these products
    valid_orders = (order_products_prior.groupby('order_id')['product_id']
                    .apply(lambda x: all(pid in top_products for pid in x)))
    valid_order_ids = valid_orders[valid_orders].index.tolist()
    order_products_prior = order_products_prior[order_products_prior['order_id'].isin(valid_order_ids)]

    # Limit to max_orders
    order_ids = order_products_prior['order_id'].unique()[:MAX_ORDERS]
    order_products_prior = order_products_prior[order_products_prior['order_id'].isin(order_ids)]

    # --- Basket matrix for frequent pairs ---
    basket = (order_products_prior
              .groupby(['order_id', 'product_id'])['product_id']
              .count().unstack().fillna(0))
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    basket = basket[[pid for pid in products['product_id'] if pid in basket.columns]]

    # --- Continue as before ---
    products = assign_use_value(products)
    frequent_pairs = find_frequent_pairs(basket, min_support=0.01)
    promotions = create_promotions(frequent_pairs, products, discount=0.5, top_n=5)  # Use top 5 for speed

    # 2. Build environment and agent
    n_products = len(products)
    n_promos = len(promotions)
    n_actions = n_products + n_promos + 1  # add each product, accept each promo, checkout
    n_states = 2 ** (n_products + n_promos)  # basket one-hot + promo_active

    env = SupermarketEnv(products, promotions)
    agent = QLearningAgent(n_states, n_actions)
    indexer = StateIndexer(n_products, n_promos)

    n_episodes = 500  # Lower for speed; increase for better learning
    rewards = []

    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0
        for t in range(20):  # max steps per episode
            state_idx = indexer.state_to_idx(state)
            action = agent.select_action(state_idx)
            next_state, reward, done, _ = env.step(action)
            next_state_idx = indexer.state_to_idx(next_state)
            agent.update(state_idx, action, reward, next_state_idx, done)
            state = next_state
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
        if (ep + 1) % 20 == 0:
            print(f"Episode {ep+1}, Reward: {total_reward}")
            env.render()

    print("Average reward over last 20 episodes:", np.mean(rewards[-20:]))

if __name__ == "__main__":
    main()