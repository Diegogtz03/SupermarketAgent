import pandas as pd
import numpy as np
import scipy
import sklearn
import matplotlib
import kagglehub
from kagglehub import KaggleDatasetAdapter

orders = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "yasserh/instacart-online-grocery-basket-analysis-dataset",
  "orders.csv",
)

products = pd.read_csv("data/products.csv")

departments = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "yasserh/instacart-online-grocery-basket-analysis-dataset",
  "departments.csv",
)

aisles = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "yasserh/instacart-online-grocery-basket-analysis-dataset",
  "aisles.csv",
)

order_products_prior = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "yasserh/instacart-online-grocery-basket-analysis-dataset",
  "order_products__prior.csv",
)

order_products_train = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "yasserh/instacart-online-grocery-basket-analysis-dataset",
  "order_products__train.csv",
)