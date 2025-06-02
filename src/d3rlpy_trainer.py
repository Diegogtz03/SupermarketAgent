import d3rlpy
import pickle
import numpy as np
import pandas as pd
import os

def train_with_d3rlpy(dataset, env, n_steps=20000):
    """Train using d3rlpy offline RL algorithms"""
    
    device = 'cuda:0' if d3rlpy.cuda.is_available() else 'cpu'
    
    # Conservative Q-Learning (CQL)
    cql = d3rlpy.algos.CQLConfig(
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        batch_size=256,
        target_update_interval=8000,
    ).create(device=device)
    
    cql.fit(
        dataset,
        n_steps=n_steps,
        evaluators={
            "environment": d3rlpy.metrics.EnvironmentEvaluator(env, n_trials=10),
        },
        experiment_name="supermarket_cql"
    )
    
    cql.save("models/cql_supermarket.d3")
    return cql

def evaluate_d3rlpy_agent(model_path: str, env, n_episodes=50):
    """Evaluate the trained d3rlpy agent"""
    
    # Load model
    if "cql" in model_path:
        agent = d3rlpy.algos.CQL.from_file(model_path)
    elif "iql" in model_path:
        agent = d3rlpy.algos.IQL.from_file(model_path)
    else:
        raise ValueError(f"Unknown model type in path: {model_path}")
    
    # Run evaluation episodes
    total_rewards = []
    promo_usage = []
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        promos_used = 0
        
        for step in range(30):  # Max 30 steps per episode
            action = agent.predict([state])[0]
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            
            # Count promo actions (assuming promo actions are after product actions)
            n_products = len(env.products)
            if action >= n_products and action < n_products + len(env.promotions):
                promos_used += 1
            
            state = next_state
            if done:
                break
        
        total_rewards.append(episode_reward)
        promo_usage.append(promos_used)
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Reward={episode_reward:.2f}, Promos={promos_used}")
    
    results = {
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'avg_promos': np.mean(promo_usage),
        'promo_rate': np.mean([p > 0 for p in promo_usage])
    }
    
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Average Promos Used: {results['avg_promos']:.2f}")
    print(f"Promo Usage Rate: {results['promo_rate']:.2%}")
    
    return results

if __name__ == "__main__":
    # Load your converted dataset
    with open("data/d3rlpy_dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    
    # Load environment components
    products = pd.read_csv("data/products.csv")
    promotions = [
        {'buy': 123, 'get': 456, 'discount': 0.5},
        {'buy': 789, 'get': 101, 'discount': 0.3},
    ]
    
    env = SupermarketEnv(products, promotions)
    
    # Train using d3rlpy offline RL algorithms
    cql = train_with_d3rlpy(dataset, env)
    
    # Evaluate the trained d3rlpy agent
    evaluate_d3rlpy_agent("models/cql_supermarket.d3", env)
    
    # Also try IQL (Implicit Q-Learning) - often works well for offline data
    iql = d3rlpy.algos.IQLConfig(
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        batch_size=256,
    ).create(device='cuda:0' if d3rlpy.cuda.is_available() else 'cpu')
    
    iql.fit(
        dataset,
        n_steps=50000,
        evaluators={
            "environment": d3rlpy.metrics.EnvironmentEvaluator(env, n_trials=10),
        },
        experiment_name="supermarket_iql"
    )
    
    iql.save("models/iql_supermarket.d3")
    print("IQL model saved!")
    
    # Evaluate both models
    evaluate_d3rlpy_agent("models/cql_supermarket.d3", env)
    evaluate_d3rlpy_agent("models/iql_supermarket.d3", env) 