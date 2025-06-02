import pandas as pd
import numpy as np
import pickle
from typing import List, Dict, Tuple
import os

def convert_baskets_to_episodes(products_path: str, baskets_path: str, promotions: List[Dict], 
                               products_df=None, max_episodes=1000, max_products=100) -> List[Dict]:
    """Convert basket sequences to RL episodes with promotion opportunities - optimized for memory"""
    
    print(f"Loading data from {baskets_path}...")
    
    # Use the already filtered products DataFrame if provided
    if products_df is not None:
        products = products_df.copy()
        print(f"Using provided products DataFrame with {len(products)} products")
    else:
        if not os.path.exists(products_path):
            print(f"Warning: {products_path} not found, using fallback")
            products = pd.DataFrame({'product_id': range(max_products), 'price': np.random.uniform(1, 20, max_products)})
        else:
            products = pd.read_csv(products_path)
            products = products.head(max_products).reset_index(drop=True)
    
    if not os.path.exists(baskets_path):
        print(f"Warning: {baskets_path} not found, creating dummy data")
        dummy_baskets = pd.DataFrame({
            'product_id': [[1, 2, 3], [4, 5], [1, 4, 6, 7], [2, 8, 9]]
        })
        dummy_baskets.to_pickle(baskets_path)
        baskets = dummy_baskets
    else:
        baskets = pd.read_pickle(baskets_path)
        baskets = baskets.reset_index()
        baskets = baskets.head(max_episodes)
    
    print(f"Processing {len(baskets)} baskets with {len(products)} products...")
    print(f"Product ID range in products: {products['product_id'].min()} to {products['product_id'].max()}")
    
    # Check what product IDs are actually in the baskets
    all_basket_products = set()
    for _, row in baskets.head(10).iterrows():  # Sample first 10 baskets
        basket_items = row['product_id']
        if isinstance(basket_items, list):
            all_basket_products.update([int(item) for item in basket_items])
    
    print(f"Sample product IDs in baskets: {sorted(list(all_basket_products))[:10]}")
    
    episodes = []
    valid_product_ids = set(products['product_id'].values)
    print(f"Valid product IDs: {sorted(list(valid_product_ids))[:10]}...")
    
    baskets_processed = 0
    baskets_with_valid_products = 0
    
    for i, (idx, row) in enumerate(baskets.iterrows()):
        baskets_processed += 1
        
        if i % 50 == 0:
            print(f"Processed {i}/{len(baskets)} baskets...")
            
        basket_items = row['product_id']
        
        if not isinstance(basket_items, list):
            continue
        
        # Convert to integers and filter to valid products
        basket_items = [int(float(item)) for item in basket_items if not pd.isna(item)]
        original_count = len(basket_items)
        basket_items = [item for item in basket_items if item in valid_product_ids]
        
        if len(basket_items) > 0:
            baskets_with_valid_products += 1
        
        if i < 5:  # Debug first few baskets
            print(f"Basket {i}: {original_count} total items, {len(basket_items)} valid items")
        
        if len(basket_items) < 2:
            continue
            
        # Limit basket size to prevent memory issues
        basket_items = basket_items[:10]
            
        episode = create_episode_from_basket(basket_items, promotions, products)
        
        if episode and len(episode['observations']) > 1:
            episodes.append(episode)
            
        # Stop if we have enough episodes
        if len(episodes) >= max_episodes:
            break
    
    print(f"Processed {baskets_processed} baskets")
    print(f"Baskets with valid products: {baskets_with_valid_products}")
    print(f"Created {len(episodes)} episodes")
    return episodes

def create_episode_from_basket(basket_items: List, promotions: List[Dict], products: pd.DataFrame) -> Dict:
    """Create a single episode from a basket"""
    episode = {
        'observations': [],
        'actions': [], 
        'rewards': [],
        'terminals': []
    }
    
    current_basket = []
    
    for step, item in enumerate(basket_items):
        # Current state (simplified)
        state = create_simple_state(current_basket, promotions, len(products))
        episode['observations'].append(state)
        
        # Find action index - map product_id to index in products DataFrame
        try:
            product_rows = products[products['product_id'] == item]
            if len(product_rows) == 0:
                continue
                
            action = product_rows.index[0]
            episode['actions'].append(action)
        except Exception as e:
            print(f"Error finding product {item}: {e}")
            continue
        
        # Calculate reward
        reward = calculate_simple_reward(item, current_basket, promotions, products)
        episode['rewards'].append(reward)
        
        current_basket.append(item)
        episode['terminals'].append(step == len(basket_items) - 1)
    
    return episode

def create_simple_state(basket: List, promotions: List[Dict], n_products: int) -> np.ndarray:
    """Create a simplified state representation that matches the environment"""
    
    # FIXED: Create state that matches environment exactly
    # Environment state = [product_one_hot, promo_availability]
    
    # Product one-hot vector (full size to match environment)
    product_features = np.zeros(n_products, dtype=np.float32)
    for i, item in enumerate(basket):
        if i < n_products:  # Safety check
            product_features[i] = 1.0
    
    # Promotion availability vector (match environment exactly)
    promo_features = np.zeros(len(promotions), dtype=np.float32)
    for i, promo in enumerate(promotions):
        if i < len(promo_features):
            available = promo['buy'] in basket and promo['get'] not in basket
            promo_features[i] = float(available)
    
    # Combine to match environment state format
    state = np.concatenate([product_features, promo_features])
    
    print(f"Created state with {len(state)} dimensions ({n_products} products + {len(promotions)} promos)")
    
    return state

def calculate_simple_reward(item: int, basket: List, promotions: List[Dict], products: pd.DataFrame) -> float:
    """Simplified reward calculation"""
    base_reward = 1.0
    
    # Promo bonus (simplified)
    promo_bonus = 0.0
    for promo in promotions[:5]:
        if promo['buy'] == item:
            promo_bonus += 2.0
        if promo['get'] == item and promo['buy'] in basket:
            promo_bonus += 5.0
    
    return base_reward + promo_bonus

def create_d3rlpy_dataset(episodes: List[Dict]):
    """Convert episodes to d3rlpy dataset format with memory optimization"""
    print("Converting episodes to d3rlpy format...")
    
    if not episodes:
        print("No episodes to convert!")
        return None
    
    observations = []
    actions = []
    rewards = []
    terminals = []
    
    for i, episode in enumerate(episodes):
        if i % 100 == 0:
            print(f"Processing episode {i}/{len(episodes)}...")
        
        # Validate episode data
        if not all(key in episode for key in ['observations', 'actions', 'rewards', 'terminals']):
            print(f"Skipping episode {i}: missing required keys")
            continue
            
        if len(episode['observations']) != len(episode['actions']):
            print(f"Skipping episode {i}: mismatched lengths")
            continue
            
        observations.extend(episode['observations'])
        actions.extend(episode['actions'])
        rewards.extend(episode['rewards'])
        terminals.extend(episode['terminals'])
    
    if not observations:
        print("No valid transitions found!")
        return None
    
    # Convert to numpy arrays
    print("Converting to numpy arrays...")
    observations = np.array(observations, dtype=np.float32)
    actions = np.array(actions, dtype=np.int32)
    rewards = np.array(rewards, dtype=np.float32)
    terminals = np.array(terminals, dtype=bool)
    
    print(f"Dataset shape: {observations.shape[0]} transitions, {observations.shape[1]} features")
    print(f"Action range: {actions.min()} to {actions.max()}")
    print(f"Reward range: {rewards.min():.2f} to {rewards.max():.2f}")
    
    # Create d3rlpy dataset
    import d3rlpy
    dataset = d3rlpy.dataset.MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals
    )
    
    # Return both dataset and transition count
    return dataset, observations.shape[0]

if __name__ == "__main__":
    # Example promotions (adjust based on your data)
    promotions = [
        {'buy': 123, 'get': 456, 'discount': 0.5},  # Buy 123, get 456 at 50% off
        {'buy': 789, 'get': 101, 'discount': 0.3},  # Buy 789, get 101 at 30% off
    ]
    
    episodes = convert_baskets_to_episodes("data/products.csv", "data/train.pkl", promotions)
    dataset, transition_count = create_d3rlpy_dataset(episodes)
    
    # Save dataset
    with open("data/d3rlpy_dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)
    
    print(f"Created dataset with {transition_count} transitions") 