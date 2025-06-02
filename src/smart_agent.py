import numpy as np

class SmartPromotionAgent:
    def __init__(self, products, promotions):
        self.products = products
        self.promotions = promotions
        
    def select_action(self, state, valid_actions):
        """Smart agent that prioritizes promotions"""
        n_products = len(self.products)
        
        # Parse current basket from state
        basket_items = []
        for i, has_item in enumerate(state[:n_products]):
            if has_item > 0:
                product_id = self.products.iloc[i]['product_id']
                basket_items.append(product_id)
        
        # Check for available promotions first
        for i, promo in enumerate(self.promotions):
            promo_action = n_products + i
            if promo_action < len(valid_actions) and valid_actions[promo_action]:
                print(f"🎯 Taking promotion: Buy {promo['buy']} → Get {promo['get']} at {int(promo['discount']*100)}% off")
                return promo_action
        
        # Look for items that enable promotions
        for i, promo in enumerate(self.promotions):
            buy_item = promo['buy']
            if buy_item not in basket_items:
                # Find the action for this product
                try:
                    product_idx = self.products[self.products['product_id'] == buy_item].index[0]
                    if product_idx < len(valid_actions) and valid_actions[product_idx]:
                        print(f"📦 Adding {buy_item} to enable promotion")
                        return product_idx
                except:
                    continue
        
        # Fallback: random valid action
        valid_indices = [i for i, v in enumerate(valid_actions) if v]
        return np.random.choice(valid_indices) if valid_indices else 0

def test_smart_agent(env, products, promotions):
    """Test the smart promotion agent"""
    agent = SmartPromotionAgent(products, promotions)
    
    total_rewards = []
    promo_usage = []
    
    for episode in range(20):
        state = env.reset()
        episode_reward = 0
        promos_used = 0
        
        print(f"\n--- Episode {episode + 1} ---")
        
        for step in range(15):
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)
            
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            # Count promos
            if action >= len(products) and action < len(products) + len(promotions):
                promos_used += 1
            
            state = next_state
            if done:
                break
        
        total_rewards.append(episode_reward)
        promo_usage.append(promos_used)
        
        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Promos={promos_used}")
        if episode == 0:  # Show first episode in detail
            env.render()
    
    print(f"\n📊 Smart Agent Results:")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    print(f"Average Promos Used: {np.mean(promo_usage):.2f}")
    print(f"Promo Usage Rate: {np.mean([p > 0 for p in promo_usage]):.1%}")
    
    return total_rewards

if __name__ == "__main__":
    # This would go in your main.py
    from src.supermarket_env import SupermarketEnv
    # ... load your data ...
    # env = SupermarketEnv(products, promotions)
    # test_smart_agent(env, products, promotions) 