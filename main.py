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

# Fusionar order_products_prior con products para obtener detalles de cada producto en los pedidos
# Esta es la tabla principal que usaremos para analizar el contenido de las canastas y la vida útil de los productos
order_details = pd.merge(order_products_prior, products, on='product_id', how='left')

print("\n--- Análisis de Frecuencia de Productos (con Vida Útil) ---")

# Crear diccionarios para mapear product_id a nombres y otros detalles para facilitar la lectura
product_id_to_name = products.set_index('product_id')['product_name'].to_dict()
product_id_to_department = products.set_index('product_id')['department'].to_dict()
product_id_to_aisle = products.set_index('product_id')['aisle'].to_dict()

print("\nCalculando la frecuencia de cada item (producto, vida_útil)...")
# Agrupar por product_id y use_value y contar el tamaño de cada grupo.
# Esto nos da la frecuencia con la que cada combinación (product_id, use_value) aparece en el total de productos comprados.
individual_item_frequencies = order_details.groupby(['product_id', 'use_value']).size().sort_values(ascending=False)

N_top_items = 20
print(f"\nTop {N_top_items} productos más frecuentes (considerando su vida útil):")
for (pid, uv), count in individual_item_frequencies.head(N_top_items).items():
    product_name = product_id_to_name.get(pid, f"ID Desconocido: {pid}")
    department = product_id_to_department.get(pid, "N/A")
    aisle = product_id_to_aisle.get(pid, "N/A")
    print(f"- {product_name} (Vida Útil: {uv}) | Dept: {department}, Pasillo: {aisle} | Frecuencia: {count}")

# Ahora, para encontrar *combinaciones* populares (múltiples productos comprados juntos),
# necesitamos preparar las "canastas". Cada canasta es una lista de los items comprados en un pedido.
# Un "item" aquí será la tupla (product_id, use_value) para capturar ambos aspectos.

print("\nPreparando canastas para el análisis de conjuntos de items frecuentes...")
# Agrupar por order_id. Para cada pedido, crear una lista de tuplas (product_id, use_value).
# Esto representa cada canasta con sus items (incluyendo vida útil).
# Esta operación puede ser intensiva en memoria y tiempo para datasets grandes.
baskets_for_mining = order_details.groupby('order_id').apply(
    lambda x: [tuple(item) for item in x[['product_id', 'use_value']].values]
).tolist()
# El .values convierte las columnas seleccionadas del DataFrame del grupo a un array NumPy.
# Luego iteramos sobre las filas de ese array para crear las tuplas.
# .tolist() convierte la Serie resultante (donde el índice es order_id y el valor es una lista de items) en una lista de listas.

print(f"\nSe crearon {len(baskets_for_mining)} canastas para el análisis.")
print("Ejemplo de las primeras 2 canastas (items como (product_id, use_value)):")
if len(baskets_for_mining) > 0:
    # Mostrar hasta 10 items de la primera canasta para brevedad
    print(f"Canasta 1 ({len(baskets_for_mining[0])} items): {baskets_for_mining[0][:10]}")
if len(baskets_for_mining) > 1:
    # Mostrar hasta 10 items de la segunda canasta para brevedad
    print(f"Canasta 2 ({len(baskets_for_mining[1])} items): {baskets_for_mining[1][:10]}")

print("\n\nCon las 'baskets_for_mining' (una lista de listas, donde cada sublista es una canasta de items (product_id, use_value)),")
print("se pueden usar algoritmos de minería de reglas de asociación (como Apriori o FP-Growth, usualmente de la librería mlxtend)")
print("para encontrar 'conjuntos de items frecuentes', es decir, combinaciones de productos (con su vida útil) que se compran juntos a menudo.")
print("Estos algoritmos identifican qué grupos de productos superan un umbral mínimo de aparición (soporte).")
print("Posteriormente, también se pueden generar 'reglas de asociación' (ej. si compran A y B, es probable que también compren C).")

print("\n\n--- Preparación de Datos para Embeddings de Productos (Word2Vec) ---")

# Para Word2Vec, necesitamos secuencias de productos por pedido.
# Los productos deben estar en el orden en que se agregaron al carrito.
# Usaremos product_id como los "tokens" o "palabras".

print("\nCreando secuencias de productos por pedido para Word2Vec...")
# Asegurarnos de que order_details está ordenado por order_id y luego por add_to_cart_order
# Esto es importante para que las secuencias de productos dentro de cada pedido sean correctas.
ordered_products_in_orders = order_details.sort_values(['order_id', 'add_to_cart_order'])

# Agrupar por order_id y recopilar los product_id (convertidos a string) en una lista para cada pedido.
# Estas listas de product_id serán nuestras "frases" para Word2Vec.
product_sequences = ordered_products_in_orders.groupby('order_id')['product_id'].apply(lambda x: [str(pid) for pid in x]).tolist()

print(f"Se crearon {len(product_sequences)} secuencias de productos (una por pedido).")
print("Ejemplo de las primeras 2 secuencias (product_ids como strings):")
if len(product_sequences) > 0:
    # Mostrar hasta 10 productos de la primera secuencia para brevedad
    print(f"Secuencia 1 ({len(product_sequences[0])} productos): {product_sequences[0][:10]}")
if len(product_sequences) > 1:
    # Mostrar hasta 10 productos de la segunda secuencia para brevedad
    print(f"Secuencia 2 ({len(product_sequences[1])} productos): {product_sequences[1][:10]}")

print("\n\n--- Entrenamiento del Modelo Word2Vec (requiere la librería gensim) ---")
print("El siguiente bloque de código es un ejemplo de cómo entrenarías un modelo Word2Vec.")
print("Necesitarás tener 'gensim' instalado en tu entorno (ej. pip install gensim).")
print("Descomenta y ejecuta este bloque en tu máquina local.")

# --- INICIO: Bloque para ejecutar localmente con gensim ---
from gensim.models import Word2Vec
import logging

# Configurar logging para ver el progreso del entrenamiento de Word2Vec
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if product_sequences: # Solo entrenar si tenemos secuencias
    print("\nEntrenando el modelo Word2Vec...")
    # Parámetros comunes para Word2Vec:
    # - sentences: las secuencias de productos que preparamos.
    # - vector_size: la dimensionalidad de los vectores de producto (embeddings).
    # - window: el tamaño máximo de la ventana de contexto (cuántos productos antes y después se consideran).
    # - min_count: ignorar todos los productos con una frecuencia total inferior a esta.
    # - workers: número de hilos a usar para el entrenamiento (paralelización).
    # - sg: 0 para CBOW (predice palabra actual basado en contexto), 1 para Skip-gram (predice contexto basado en palabra actual). Skip-gram suele ser mejor para datasets grandes.
    
    # Ejemplo de parámetros (puedes ajustarlos):
    vector_dim = 100  # Dimensionalidad de los embeddings
    window_size = 5   # Ventana de contexto
    min_product_count = 5 # Mínima frecuencia de un producto para ser considerado
    num_workers = 4   # Hilos de CPU

    word2vec_model = Word2Vec(
        sentences=product_sequences,
        vector_size=vector_dim,
        window=window_size,
        min_count=min_product_count,
        workers=num_workers,
        sg=1 # Usar Skip-gram
    )

    print("\nEntrenamiento del modelo Word2Vec completado.")

    # Guardar el modelo para uso futuro
    model_filename = "product_word2vec.model"
    word2vec_model.save(model_filename)
    print(f"Modelo Word2Vec guardado como: {model_filename}")

    # Vocabulario del modelo
    # print(f"\nTamaño del vocabulario del modelo: {len(word2vec_model.wv.index_to_key)}")

    # Ejemplo: Obtener el vector de un producto (si está en el vocabulario)
    # Primero, necesitamos un product_id de ejemplo que sepamos que está en los datos y que probablemente superó min_count.
    # Usemos el product_id de "Banana" (24852) que es muy frecuente.
    example_product_id = "24852" # Banana
    if example_product_id in word2vec_model.wv:
        print(f"\nVector para el producto {product_id_to_name.get(int(example_product_id), example_product_id)} (ID: {example_product_id}):")
        # print(word2vec_model.wv[example_product_id])
    else:
        print(f"\nEl producto ID {example_product_id} no se encuentra en el vocabulario del modelo (quizás filtrado por min_count).")

    # Ejemplo: Encontrar los productos más similares a "Banana"
    print(f"\nProductos más similares a {product_id_to_name.get(int(example_product_id), example_product_id)} (ID: {example_product_id}):")
    try:
        similar_products = word2vec_model.wv.most_similar(example_product_id, topn=10)
        for prod_id_str, similarity in similar_products:
            prod_name = product_id_to_name.get(int(prod_id_str), f"ID: {prod_id_str}")
            print(f"- {prod_name} (Similaridad: {similarity:.4f})")
    except KeyError:
        print(f"No se pudo encontrar similitudes para el producto ID {example_product_id} (no está en el vocabulario).")

else:
    print("\nNo hay secuencias de productos para entrenar el modelo Word2Vec.")

# --- FIN: Bloque para ejecutar localmente con gensim ---