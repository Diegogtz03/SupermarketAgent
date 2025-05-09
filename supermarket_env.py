import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

class SupermarketEnv(gym.Env):
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4}

    def __init__(self, render_mode=None):
        super().__init__()

        # --- Cargar Datos y Modelo ---
        try:
            self.products_df = pd.read_csv("data/products.csv")
        except FileNotFoundError:
            print("Error: data/products.csv no encontrado. Asegúrate de que el archivo está en la ubicación correcta.")
            raise
        
        try:
            self.word2vec_model = Word2Vec.load("product_word2vec.model")
            self.embedding_dim = self.word2vec_model.vector_size
        except FileNotFoundError:
            print("Error: product_word2vec.model no encontrado. Asegúrate de que el modelo fue entrenado y guardado.")
            raise

        # Mapeos útiles
        self.product_id_to_info = self.products_df.set_index('product_id').to_dict('index')
        self.valid_product_ids = list(self.products_df['product_id'])
        self.num_products = len(self.valid_product_ids)

        # --- Definir Espacios ---
        # Espacio de Observación:
        # Placeholder - Necesitamos definir esto más concretamente.
        # Ejemplo: [avg_embedding_basket (dim), last_item_use_value (1), num_items_in_basket (1), was_last_item_fresh (1)]
        # Por ahora, un Box simple. Ajustaremos las dimensiones más adelante.
        # Supongamos que el embedding_dim es 100.
        # avg_embedding (100) + last_use_value (1) + num_items (1) + last_fresh (1) + last_long_discountable (1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.embedding_dim + 4,), dtype=np.float32
        )

        # Espacio de Acción:
        # Acción 0: finalizar compra
        # Acción 1 a N: añadir product_id (indexado)
        # Usaremos un mapeo de índice de acción a product_id
        self.action_to_product_id = {i + 1: pid for i, pid in enumerate(self.valid_product_ids)}
        self.product_id_to_action = {pid: i + 1 for i, pid in enumerate(self.valid_product_ids)}
        self.action_space = spaces.Discrete(self.num_products + 1) # +1 para la acción "finalizar compra"
        
        self.FINISH_SHOPPING_ACTION = 0

        # --- Estado del Entorno ---
        self.current_basket = [] # Lista de product_ids
        self.current_basket_details = [] # Lista de diccionarios con info del producto
        self.last_added_item_info = None # Para la lógica de descuentos
        self.current_step = 0
        self.max_steps_per_episode = 50 # Límite de items en la canasta o pasos

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        print(f"Entorno SupermarketEnv inicializado.")
        print(f"Dimension de embedding: {self.embedding_dim}")
        print(f"Número de productos: {self.num_products}")
        print(f"Tamaño del espacio de acción: {self.action_space.n}")


    def _get_obs(self):
        # Calcular el embedding promedio de la canasta
        avg_embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        if self.current_basket:
            embeddings = []
            for pid in self.current_basket:
                try:
                    embeddings.append(self.word2vec_model.wv[str(pid)])
                except KeyError:
                    # Si un producto no tiene embedding (ej. filtrado por min_count), usar vector de ceros
                    embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
            if embeddings:
                avg_embedding = np.mean(embeddings, axis=0)
        
        num_items_in_basket = float(len(self.current_basket))
        
        last_item_use_value = 0.0 # Valor por defecto si la canasta está vacía o no hay info
        was_last_item_fresh = 0.0 # 0 para no, 1 para sí
        was_last_item_long_life_and_discountable = 0.0 # 0 para no, 1 para sí

        if self.last_added_item_info:
            last_item_use_value = float(self.last_added_item_info.get('use_value', 0))
            if last_item_use_value in [1, 2]: # Fresco
                was_last_item_fresh = 1.0
            if last_item_use_value in [4, 5]: # Larga vida util
                 # Esto es para indicar si el *último* item añadido fue de larga vida útil,
                 # no necesariamente si se le *aplicó* un descuento. La lógica de descuento es sobre el *siguiente*.
                was_last_item_long_life_and_discountable = 1.0


        obs = np.concatenate([
            avg_embedding,
            np.array([last_item_use_value], dtype=np.float32),
            np.array([num_items_in_basket], dtype=np.float32),
            np.array([was_last_item_fresh], dtype=np.float32),
            np.array([was_last_item_long_life_and_discountable], dtype=np.float32)
        ])
        return obs

    def _get_info(self):
        # Devolver información adicional, útil para debugging o análisis
        return {
            "basket_size": len(self.current_basket),
            "current_basket_pids": self.current_basket.copy(),
            "last_added_item_uv": self.last_added_item_info['use_value'] if self.last_added_item_info else None,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Necesario para gestionar el generador de números aleatorios de Gym

        self.current_basket = []
        self.current_basket_details = []
        self.last_added_item_info = None
        self.current_step = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
            
        return observation, info

    def step(self, action):
        self.current_step += 1
        terminated = False
        truncated = False
        reward = 0.0
        
        applied_discount_for_long_life = False

        if action == self.FINISH_SHOPPING_ACTION:
            terminated = True
            # Recompensa podría basarse en la canasta final, si cumple la hipótesis, etc.
            # Por ahora, una pequeña recompensa por finalizar.
            reward = 0.1 
            if len(self.current_basket) == 0: # Penalizar si termina sin nada
                reward = -1.0

        elif self.action_space.contains(action): # Es una acción de añadir producto
            selected_product_id = self.action_to_product_id.get(action)
            
            if selected_product_id: # Asegurarse de que la acción mapea a un ID válido
                product_info = self.product_id_to_info.get(selected_product_id)
                
                if product_info:
                    self.current_basket.append(selected_product_id)
                    self.current_basket_details.append(product_info)
                    
                    current_product_use_value = product_info.get('use_value', 0)
                    reward = 0.05 # Pequeña recompensa por añadir cualquier producto

                    # Lógica de descuento según hipótesis
                    if self.last_added_item_info:
                        last_uv = self.last_added_item_info.get('use_value', 0)
                        # Si el último fue fresco (1 o 2) y el actual es de larga vida útil (4 o 5)
                        if last_uv in [1, 2] and current_product_use_value in [4, 5]:
                            reward += 0.5 # Recompensa adicional por el descuento hipotético
                            applied_discount_for_long_life = True
                            # print(f"Debug: Descuento aplicado! Último UV: {last_uv}, Actual UV: {current_product_use_value}")

                    self.last_added_item_info = product_info # Actualizar para el próximo paso
                else:
                    # Acción inválida (product_id no encontrado), aunque no debería pasar con el mapeo
                    reward = -0.5 
            else:
                # Acción inválida (no es FINISH y no mapea a producto)
                reward = -1.0
                terminated = True # Terminar si la acción es completamente inválida

        else: # Acción no válida
            reward = -1.0
            terminated = True # Terminar si la acción no está en el espacio de acciones

        # Condición de truncamiento (si el episodio dura demasiado)
        if self.current_step >= self.max_steps_per_episode:
            truncated = True
            # print("Debug: Episodio truncado por max_steps.")

        observation = self._get_obs()
        info = self._get_info()
        info['applied_discount'] = applied_discount_for_long_life


        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'ansi':
            return self._render_frame_ansi()
        #elif self.render_mode == "human":
            # self._render_frame() se llama en step y reset

    def _render_frame_ansi(self):
        output = f"Paso: {self.current_step}\n"
        output += f"Canasta ({len(self.current_basket)} items):\n"
        for i, item in enumerate(self.current_basket_details):
            name = item.get('product_name', 'Desconocido')
            uv = item.get('use_value', 'N/A')
            output += f"  {i+1}. {name} (UV: {uv})\n"
        
        if self.last_added_item_info:
            last_name = self.last_added_item_info.get('product_name', 'N/A')
            last_uv = self.last_added_item_info.get('use_value', 'N/A')
            output += f"Último añadido: {last_name} (UV: {last_uv})\n"
        
        obs = self._get_obs()
        # output += f"Observación actual (primeros 5 elementos embedding + otros): {np.round(obs[:5], 2).tolist()} ... {np.round(obs[self.embedding_dim:], 2).tolist()}\n"

        return output

    def _render_frame(self):
        # Para render_mode = 'human', podríamos usar matplotlib o pygame en el futuro.
        # Por ahora, solo imprimiremos a la consola.
        print(self._render_frame_ansi())


    def close(self):
        # Limpiar recursos si es necesario (ej. cerrar ventanas de Pygame)
        pass

if __name__ == '__main__':
    # --- Prueba básica del entorno ---
    print("\n--- Probando el entorno SupermarketEnv ---")
    
    # tener gensim y pandas instalados, y los archivos de datos/modelo.
    try:
        env = SupermarketEnv(render_mode='human')
        
        # Verificar espacios
        print(f"Espacio de Observación: {env.observation_space}")
        print(f"Forma de Observación: {env.observation_space.sample().shape}")
        print(f"Espacio de Acción: {env.action_space}")
        print(f"Acción de ejemplo: {env.action_space.sample()}")

        obs, info = env.reset()
        print("\nEstado inicial (primeros 5 elementos embedding + otros):")
        print(f"{np.round(obs[:5], 2).tolist()} ... {np.round(obs[env.embedding_dim:], 2).tolist()}")
        print(f"Info inicial: {info}")

        terminated = False
        truncated = False
        total_reward = 0
        
        for step_num in range(10): # Probar 10 pasos aleatorios
            action = env.action_space.sample() # Tomar una acción aleatoria
            
            action_name = "Finalizar Compra"
            if action != env.FINISH_SHOPPING_ACTION:
                pid = env.action_to_product_id.get(action)
                if pid:
                    action_name = env.product_id_to_info.get(pid, {}).get('product_name', f"ID: {pid}")

            print(f"\n--- Paso {step_num + 1} --- tomando acción: {action} ({action_name}) ---")
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"Recompensa: {reward:.2f}")
            print(f"Observación (primeros 5 elementos embedding + otros): {np.round(obs[:5], 2).tolist()} ... {np.round(obs[env.embedding_dim:], 2).tolist()}")
            print(f"Info: {info}")
            print(f"Terminado: {terminated}, Truncado: {truncated}")

            if terminated or truncated:
                print(f"\nEpisodio finalizado después de {step_num + 1} pasos. Recompensa total: {total_reward:.2f}")
                obs, info = env.reset()
                total_reward = 0
                print("Entorno reseteado.")
                # break # Descomentar si quieres parar después del primer episodio terminado

        env.close()
        
    except FileNotFoundError as e:
        print(f"Error al inicializar el entorno de prueba: {e}")
        print("Asegúrate de que 'data/products.csv' y 'product_word2vec.model' existen.")
        print("Puedes necesitar ejecutar 'main.py' primero para generar el modelo Word2Vec.")
    except ImportError as e:
        print(f"Error de importación: {e}. Asegúrate de tener Gymnasium, Pandas y Gensim instalados.")

