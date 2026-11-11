
import numpy as np

from IA.model import NeuralNetwork
from IAE.memory import ExperienceMemory
from IAE.exploration import WeightExplorer



class EvolutionaryAI:
    def __init__(
        self,
        memory_capacity=500,
        noise_scale=0.02,
        candidates=10,
        seed=42
    ):
        # Red neuronal convencional
        self.network = NeuralNetwork(seed=seed)

        # Memoria de experiencias
        self.memory = ExperienceMemory(
            capacity=memory_capacity,
            seed=seed
        )

        # Exploración de pesos
        self.explorer = WeightExplorer(
            noise_scale=noise_scale,
            candidates=candidates,
            seed=seed
        )

    def load(self, path):
        self.network.load(path)

    def predict(self, X):
        return self.network.predict(X)

    def remember(self, X, y):
        self.memory.add(X, y)

    def adapt(
        self,
        X_new,
        y_new,
        X_val,
        y_val,
        replay_size=400,
        epochs=1000,
        lr=0.1
    ):
        # Recuperamos experiencias anteriores
        if len(self.memory) > 0:
            X_old, y_old = self.memory.sample(
                n=replay_size
            )

            X_train = np.vstack([X_new, X_old])
            y_train = np.vstack([y_new, y_old])
        else:
            X_train = X_new
            y_train = y_new

        # Aprendizaje convencional
        print("=== ENTRENAMIENTO CON MEMORIA ===")

        self.network.train(
            X_train,
            y_train,
            epochs=epochs,
            lr=lr
        )

        # Exploración adicional
        print("\n=== EXPLORACIÓN DE PESOS ===")

        result = self.explorer.explore(
            self.network,
            X_val,
            y_val
        )

        print(
            f"Error inicial: {result['initial_loss']:.6f}"
        )
        print(
            f"Mejor error: {result['best_loss']:.6f}"
        )
        print(
            f"¿Mejoró?: {result['improved']}"
        )

        # Guardamos experiencias nuevas para el futuro
        self.remember(X_new, y_new)

        return result
