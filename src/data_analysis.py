import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import numpy as np

def find_frequent_pairs(order_products_df):
    # order_products_df: columns ['order_id', 'product_id']
    basket = (order_products_df
              .groupby(['order_id', 'product_id'])['product_id']
              .count().unstack().fillna(0))
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    return rules

def assign_shelf_life(products_df):
    # products_df: columns ['product_id', 'product_name', ...]
    # For now, assign random shelf-life
    products_df['shelf_life'] = np.random.randint(1, 30, size=len(products_df))
    return products_df 