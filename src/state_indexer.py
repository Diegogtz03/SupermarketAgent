import numpy as np

class StateIndexer:
    def __init__(self, n_products, n_promos):
        self.n_products = n_products
        self.n_promos = n_promos

    def state_to_idx(self, state):
        # state: basket one-hot + promo_active (binary vector)
        # Convert to tuple for hashing
        return int("".join(str(int(x)) for x in state), 2) 