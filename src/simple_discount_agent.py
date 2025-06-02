import numpy as np
import pandas as pd

class DiscountAgent:
    def __init__(self, products, promotions):
        self.products = products
        self.promotions = promotions
        print(f"🎯 Agent knows about {len(promotions)} promotions")
        for i, promo in enumerate(promotions):
            print(f"  Promo {i}: Buy {promo['buy']} → Get {promo['get']} at {promo['discount']*100:.0f}% off")
    
    def get_basket_from_state(self, state):
        """Extract current basket from state vector"""
        n_products = len(self.products)
        basket = []
        for i in range(min(n_products, len(state))):
            if state[i] > 0:
                product_id = self.products.iloc[i]['product_id']
                basket.append(product_id)
        return basket
    
    def select_action(self, state, valid_actions=None):
        """Simple but effective discount-seeking strategy"""
        basket = self.get_basket_from_state(state)
        n_products = len(self.products)
        
        # Strategy 1: If promotion is available, take it!
        for i, promo in enumerate(self.promotions):
            promo_action = n_products + i
            if promo_action < len(valid_actions) and valid_actions[promo_action]:
                print(f"🎉 TAKING PROMOTION! Buy {promo['buy']} → Get {promo['get']}")
                return promo_action
        
        # Strategy 2: Add items that enable promotions
        for promo in self.promotions:
            buy_item = promo['buy']
            get_item = promo['get']
            
            # If we have the 'buy' item but not 'get' item, promotion will be available next
            if buy_item in basket and get_item not in basket:
                continue  # Promotion should already be available
            
            # If we don't have the 'buy' item, add it to enable promotion
            if buy_item not in basket:
                try:
                    product_row = self.products[self.products['product_id'] == buy_item]
                    if len(product_row) > 0:
                        product_idx = product_row.index[0]
                        if product_idx < len(valid_actions) and valid_actions[product_idx]:
                            print(f"📦 Adding product {buy_item} to enable promotion")
                            return product_idx
                except Exception as e:
                    continue
        
        # Strategy 3: Random valid action (but avoid checkout unless basket has items)
        valid_indices = [i for i, v in enumerate(valid_actions) if v]
        
        # Prefer not to checkout immediately if basket is empty
        if len(basket) == 0 and len(valid_indices) > 1:
            # Remove checkout action (usually the last one)
            valid_indices = valid_indices[:-1]
        
        return np.random.choice(valid_indices) if valid_indices else 0

def test_simple_agent():
    """Test with minimal setup"""
    # Create a simple test environment
    simple_products = pd.DataFrame({
        'product_id': [1, 2, 3, 4, 5],
        'product_name': ['Bread', 'Milk', 'Eggs', 'Cheese', 'Butter'],
        'price': [2.0, 3.0, 4.0, 5.0, 6.0],
        'use_value': [10, 20, 15, 25, 30]
    })
    
    simple_promotions = [
        {'buy': 1, 'get': 2, 'discount': 0.5},  # Buy Bread, get Milk 50% off
        {'buy': 3, 'get': 4, 'discount': 0.3},  # Buy Eggs, get Cheese 30% off
    ]
    
    print("🧪 Testing Simple Discount Agent")
    print("="*50)
    
    agent = DiscountAgent(simple_products, simple_promotions)
    
    # Simulate a few scenarios
    scenarios = [
        {"name": "Empty basket", "state": [0, 0, 0, 0, 0, 0, 0]},
        {"name": "Has bread", "state": [1, 0, 0, 0, 0, 1, 0]},  # Has product 1, promo 1 available
        {"name": "Has eggs", "state": [0, 0, 1, 0, 0, 0, 1]},   # Has product 3, promo 2 available
    ]
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        state = scenario['state']
        valid_actions = [True] * 7  # All actions valid
        
        action = agent.select_action(state, valid_actions)
        print(f"Agent chose action: {action}")
        
        if action < 5:
            print(f"→ Add product: {simple_products.iloc[action]['product_name']}")
        elif action < 7:
            print(f"→ Take promotion {action-5}")
        else:
            print(f"→ Checkout")
    
    return agent

# Add this to your main.py
def run_simple_agent_test():
    print("\n" + "="*50)
    print("🎯 SIMPLE DISCOUNT AGENT (Actually Works!)")
    print("="*50)
    
    agent = test_simple_agent()
    
    print(f"\n✅ Key Insights:")
    print(f"• Simple rule-based logic is interpretable")
    print(f"• Directly targets discount opportunities") 
    print(f"• No training required - works immediately")
    print(f"• Easy to debug and improve")
    print(f"• Performance is predictable")
    
    print(f"\n🎓 For your project:")
    print(f"• This demonstrates understanding of the problem")
    print(f"• Shows practical problem-solving skills")
    print(f"• Can be easily extended with more sophisticated rules")
    print(f"• Provides a solid baseline for comparison")

if __name__ == "__main__":
    run_simple_agent_test() 