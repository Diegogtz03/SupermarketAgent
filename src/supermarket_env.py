import numpy as np

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
        # action: 0..n_products-1 = add product, n_products..n_products+len(promos)-1 = accept promo, last = checkout
        reward = 0
        info = {}
        if action < self.n_products:
            pid = self.product_ids[action]
            if pid not in self.basket:
                self.basket.append(pid)
        elif action < self.n_products + len(self.promotions):
            promo_idx = action - self.n_products
            promo = self.promotions[promo_idx]
            if self._promo_available(promo):
                self.basket.append(promo['get'])
                self.applied_promos.append(promo)
        else:  # Checkout
            self.done = True
            reward = self._calculate_reward()
        return self._get_state(), reward, self.done, info

    def _calculate_reward(self):
        # Reward: total basket value + bonus for high use_value items + promo bonus
        total = 0
        for pid in self.basket:
            price = float(self.products.loc[self.products['product_id'] == pid, 'price'])
            # Apply promo discount if applicable
            for promo in self.applied_promos:
                if promo['get'] == pid:
                    price *= promo['discount']
            total += price
        # Bonus: $1 for each item with use_value > 20
        use_value_bonus = sum(
            1 for pid in self.basket if int(self.products.loc[self.products['product_id'] == pid, 'use_value']) > 20
        )
        promo_bonus = 2 * len(self.applied_promos)
        return total + use_value_bonus + promo_bonus

    def render(self):
        # Print basket product names
        print("Basket:", [
            self.products.loc[self.products['product_id'] == pid, 'product_name'].values[0]
            for pid in self.basket
        ])
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