import numpy as np
import torch

class SupermarketEnv:
    def __init__(self, products, promotions):
        self.products = products.reset_index(drop=True)
        self.promotions = promotions
        self.n_products = len(products)
        self.product_ids = list(products['product_id'])
        self.reset()

    def reset(self):
        self.basket = []
        self.applied_promos = []
        self.done = False
        return self._get_state()

    def _get_state(self):
        # State: basket (one-hot), promo active (bool for each promo)
        basket_vec = np.zeros(self.n_products)
        for pid in self.basket:
            idx = self.product_ids.index(pid)
            basket_vec[idx] = 1
        promo_active = [int(self._promo_available(promo)) for promo in self.promotions]
        return np.concatenate([basket_vec, promo_active])

    def _promo_available(self, promo):
        # Promo available if "buy" in basket and "get" not in basket
        return promo['buy'] in self.basket and promo['get'] not in self.basket

    def step(self, action):
        reward = 0
        
        if action < self.n_products:
            # Add product - small positive reward (not zero)
            pid = self.product_ids[action]
            if pid not in self.basket:
                self.basket.append(pid)
                reward = 0.5  # Small positive reward for valid actions
            
        elif action < self.n_products + len(self.promotions):
            # Apply promotion
            promo_idx = action - self.n_products
            promo = self.promotions[promo_idx]
            if self._promo_available(promo):
                get_price = float(self.products.loc[self.products['product_id'] == promo['get'], 'price'].iloc[0])
                savings = get_price * (1 - promo['discount'])
                
                self.basket.append(promo['get'])
                self.applied_promos.append(promo)
                reward = savings * 3 + 5  # Higher multiplier + base bonus
                print(f"🎉 Promo applied! Reward: {reward:.2f}")
            else:
                reward = -0.5  # Smaller penalty for invalid promo
            
        else:  # Checkout
            self.done = True
            reward = self._calculate_reward() + 1  # Bonus for completing episode
        
        for i, promo in enumerate(self.promotions):
            if self._promo_available(promo):
                print(f"Promo {i} available! Take action {self.n_products + i} to apply.")
        if any(self._promo_available(promo) for promo in self.promotions):
            print(f"Promo available! Agent action: {action}")
        return self._get_state(), reward, self.done, {}

    def _calculate_reward(self):
        if not self.basket:
            return 0
        
        full_value = 0  # Total value without any discounts
        actual_paid = 0  # Amount actually paid (with discounts)
        
        for pid in self.basket:
            price = float(self.products.loc[self.products['product_id'] == pid, 'price'].iloc[0])
            full_value += price
            
            # Apply promo discount if applicable
            discounted_price = price
            for promo in self.applied_promos:
                if promo['get'] == pid:
                    discounted_price *= promo['discount']  # e.g., 0.5 for 50% off
            actual_paid += discounted_price
        
        # Calculate savings rate: how much we saved as a percentage of full value
        savings_rate = (full_value - actual_paid) / full_value
        
        # Bonus for high use_value items
        use_value_bonus = sum(
            1 for pid in self.basket if int(self.products.loc[self.products['product_id'] == pid, 'use_value'].iloc[0]) > 20
        )
        
        # Reward structure options:
        
        # Option 1: Maximize savings rate (0 to 1, higher is better)
        base_reward = savings_rate * 100  # Scale to 0-100
        
        # Option 2: Reward actual value gained from savings
        # base_reward = full_value - actual_paid
        
        # Option 3: Reward efficiency (savings per dollar spent)
        # base_reward = (full_value - actual_paid) / actual_paid if actual_paid > 0 else 0
        
        return base_reward + use_value_bonus

    def render(self):
        if not self.basket:
            print("Basket: []")
            print("Applied promos: []")
            return
        
        # Calculate and display savings info
        full_value = sum(float(self.products.loc[self.products['product_id'] == pid, 'price'].iloc[0]) for pid in self.basket)
        actual_paid = 0
        for pid in self.basket:
            price = float(self.products.loc[self.products['product_id'] == pid, 'price'].iloc[0])
            for promo in self.applied_promos:
                if promo['get'] == pid:
                    price *= promo['discount']
            actual_paid += price
        
        savings = full_value - actual_paid
        savings_rate = savings / full_value if full_value > 0 else 0
        
        print("Basket:", [
            self.products.loc[self.products['product_id'] == pid, 'product_name'].values[0]
            for pid in self.basket
        ])
        print(f"Full Value: ${full_value:.2f}, Paid: ${actual_paid:.2f}, Savings: ${savings:.2f} ({savings_rate:.1%})")
        
        # Print applied promos with product names
        if self.applied_promos:
            promo_strings = []
            for promo in self.applied_promos:
                buy_name = self.products.loc[self.products['product_id'] == promo['buy'], 'product_name'].values[0]
                get_name = self.products.loc[self.products['product_id'] == promo['get'], 'product_name'].values[0]
                promo_strings.append(f"Buy '{buy_name}', get '{get_name}' at {int(promo['discount']*100)}% off")
            print("Applied promos:", promo_strings)
        else:
            print("Applied promos: []")

    def get_valid_actions(self):
        valid = [True] * self.n_products  # Can always add products
        # Promo actions
        for promo in self.promotions:
            valid.append(self._promo_available(promo))
        valid.append(True)  # Checkout always valid
        return valid 

    def select_action(self, state, valid_actions=None):
        if valid_actions is not None:
            valid_indices = [i for i, v in enumerate(valid_actions) if v]
            if np.random.rand() < self.epsilon:
                return np.random.choice(valid_indices)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor).cpu().numpy().flatten()
            # Mask invalid actions
            q_values[~np.array(valid_actions)] = -np.inf
            return int(np.argmax(q_values))
        else:
            # fallback: all actions valid
            ... 