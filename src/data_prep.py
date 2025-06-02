import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

def load_instacart_data():
    orders = pd.read_csv('data/orders.csv')
    order_products = pd.read_csv('data/order_products__prior.csv')
    products = pd.read_csv('data/products.csv')
    return orders, order_products, products

def get_basket_matrix(order_products, products):
    # Create a basket matrix: rows=orders, columns=products, values=1 if bought
    basket = (order_products
              .groupby(['order_id', 'product_id'])['product_id']
              .count().unstack().fillna(0))
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    # Only keep products in our products list
    basket = basket[[pid for pid in products['product_id'] if pid in basket.columns]]
    return basket

def find_frequent_pairs(basket, min_support=0.01):
    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True, max_len=2)
    # Only keep pairs
    frequent_pairs = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) == 2)]
    # Convert frozensets to tuples
    pairs = [tuple(sorted(list(x))) for x in frequent_pairs['itemsets']]
    return pairs

def assign_use_value(products):
    # Assign a random use_value (or use category, or shelf-life if you have it)
    np.random.seed(42)
    products['use_value'] = np.random.randint(1, 30, size=len(products))
    return products

def create_promotions(frequent_pairs, products, discount=0.5, top_n=10):
    promos = []
    for pair in frequent_pairs[:top_n]:
        p1, p2 = pair
        uv1 = products.loc[products['product_id'] == p1, 'use_value'].values[0]
        uv2 = products.loc[products['product_id'] == p2, 'use_value'].values[0]
        # Always make the higher use_value product the "get" item
        if uv1 > uv2:
            promos.append({'buy': p2, 'get': p1, 'discount': discount})
        else:
            promos.append({'buy': p1, 'get': p2, 'discount': discount})
    return promos

def get_products():
    # Dummy product list
    products = pd.DataFrame({
        'product_id': [0, 1, 2, 3, 4],
        'product_name': ['Milk', 'Bread', 'Eggs', 'Apple', 'Cereal'],
        'price': [2.5, 1.5, 3.0, 0.5, 4.0]
    })
    # Assign random shelf-life (days)
    np.random.seed(42)
    products['shelf_life'] = np.random.randint(3, 30, size=len(products))
    return products

def get_frequent_pairs():
    # Dummy frequent pairs (product_id tuples)
    return [(0, 1), (1, 2), (3, 4)]  # e.g., Milk & Bread, Bread & Eggs, Apple & Cereal

def get_promotions():
    # Each promo: buy X, get Y at 50% off
    return [{'buy': 0, 'get': 4, 'discount': 0.5}]  # Buy Milk, get Cereal 50% off 