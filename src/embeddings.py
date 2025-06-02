# src/embeddings.py
import os
import pickle
import pandas as pd
from gensim.models import Word2Vec
# Add imports for market basket analysis
try:
    from mlxtend.frequent_patterns import apriori, association_rules
except ImportError:
    apriori = None
    association_rules = None

def load_sequences(pickle_path):
    """Returns a list of lists of str(product_id) for Word2Vec."""
    seqs = pd.read_pickle(pickle_path)
    # seqs.index = MultiIndex(user_id, order_id); each row has list already
    return [ [str(pid) for pid in row]           # Word2Vec wants str tokens
             for row in seqs["product_id"].values ]

def train_item2vec(
        train_pkl="data/sequences.pkl",
        embed_dim=128,
        window=5,
        min_count=3,
        epochs=10,
        out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)

    sentences = load_sequences(train_pkl)
    print(f"[item2vec] #orders={len(sentences)}, "
          f"median basket size={pd.Series(map(len, sentences)).median()}")

    model = Word2Vec(
        sentences=sentences,
        vector_size=embed_dim,
        window=window,
        min_count=min_count,
        sg=1,                 # skip-gram; better for sparse products
        workers=4,
        epochs=epochs
    )

    # Save full gensim model (≈ 200 MB) and a lightweight vocab→vec dict
    model.save(os.path.join(out_dir, "item2vec.model"))

    embed = { w : model.wv[w] for w in model.wv.index_to_key }
    with open(os.path.join(out_dir, "item2vec.pkl"), "wb") as f:
        pickle.dump(embed, f)

    print(f"[item2vec] Saved embeddings: "
          f"{len(embed)} products → {out_dir+'/item2vec.pkl'}")

def run_market_basket_analysis(pickle_path, min_support=0.01, min_confidence=0.3, out_dir="models", max_baskets=10000, max_products=1000):
    """Run Apriori to find frequent itemsets and association rules."""
    os.makedirs(out_dir, exist_ok=True)
    seqs = pd.read_pickle(pickle_path)
    baskets = seqs["product_id"].tolist()[:max_baskets]
    # Flatten and count most common products
    from collections import Counter
    product_counts = Counter(pid for basket in baskets for pid in basket)
    most_common = set([pid for pid, _ in product_counts.most_common(max_products)])
    # Filter baskets to only include most common products
    baskets = [[pid for pid in basket if pid in most_common] for basket in baskets]
    all_products = sorted(most_common)
    onehot = pd.DataFrame(0, index=range(len(baskets)), columns=all_products)
    for i, basket in enumerate(baskets):
        onehot.loc[i, basket] = 1

    if apriori is not None:
        freq_itemsets = apriori(onehot, min_support=min_support, use_colnames=True)
        rules = association_rules(freq_itemsets, metric="confidence", min_threshold=min_confidence)
        freq_itemsets.to_csv(os.path.join(out_dir, "frequent_itemsets.csv"), index=False)
        rules.to_csv(os.path.join(out_dir, "association_rules.csv"), index=False)
        print(f"[apriori] Saved frequent itemsets and rules to {out_dir}")
    else:
        print("[apriori] mlxtend not installed. Please install mlxtend for market basket analysis.")

if __name__ == "__main__":
    train_item2vec()
    # Example: run market basket analysis on train data
    run_market_basket_analysis("data/train.pkl")