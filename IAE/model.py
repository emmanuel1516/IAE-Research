
import numpy as np

from IA.model import NeuralNetwork
from IAE.memory import ExperienceMemory
from IAE.exploration import MemoryGuidedExplorer
from IAE.memory_monitor import MemoryMonitor
from IAE.memory_controller import MemoryController


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

        # Memoria de experiencias con reservorio
        self.memory = ExperienceMemory(
            capacity=memory_capacity,
            seed=seed
        )

        # Exploración de pesos
        self.explorer = MemoryGuidedExplorer(
            noise_scale=noise_scale,
            candidates=candidates,
            memory_weight=0.5,
            seed=seed
        )

        # Detector de posible olvido
        self.monitor = MemoryMonitor(tolerance=0.005)
        
        self.controller = MemoryController()

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
        has_memory = len(self.memory) > 0

        before = None
        forgetting = None
        exploration = None

        # Recuperar recuerdos y medir su rendimiento
        # antes de aprender los nuevos ejemplos.
        if has_memory:
            X_old, y_old = self.memory.sample(
                n=replay_size
            )

            before = self.monitor.evaluate(
                self.network,
                X_old,
                y_old
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

        # Comprobar si aumentó el error antiguo
        if has_memory:
            after = self.monitor.evaluate(
                self.network,
                X_old,
                y_old
            )

            forgetting = self.monitor.detect_forgetting(
                before,
                after
            )

            print("\n=== MONITOR DE MEMORIA ===")
            print(
                f"Loss anterior: "
                f"{before['loss']:.6f}"
            )
            print(
                f"Loss actual: "
                f"{after['loss']:.6f}"
            )
            print(
                f"Cambio de accuracy: "
                f"{forgetting['accuracy_change']:+.2f} puntos"
            )
            print(
                f"¿Posible olvido?: "
                f"{forgetting['forgetting_detected']}"
            )

            # Explorar solamente si detectamos deterioro
            if forgetting["forgetting_detected"]:
                print("\n=== EXPLORACIÓN ACTIVADA ===")
                
                memory_weight = self.controller.calculate_weight(
                    forgetting["loss_increase"]
                )

                self.explorer.memory_weight = memory_weight

                print(f"Peso de memoria seleccionado: {memory_weight:.2f}")

                # X_val e y_val deben contener validación
                # de la fase nueva, separada del entrenamiento.
                # Usamos recuerdos antiguos para orientar
                # la selección de mutaciones.
                exploration = self.explorer.explore_guided(
                    self.network,
                    X_old,
                    y_old,
                    X_val,
                    y_val
                )

                print(
                    f"Error antiguo: "
                    f"{exploration['initial']['old_loss']:.6f}"
                    f" -> {exploration['best']['old_loss']:.6f}"
                )
                print(
                    f"Error nuevo: "
                    f"{exploration['initial']['new_loss']:.6f}"
                    f" -> {exploration['best']['new_loss']:.6f}"
                )
                print(
                    f"Error combinado: "
                    f"{exploration['initial']['total_loss']:.6f}"
                    f" -> {exploration['best']['total_loss']:.6f}"
                )

            else:
                print("\nExploración omitida: no se detectó olvido.")

        else:
            print(
                "\nExploración omitida: "
                "todavía no hay recuerdos anteriores."
            )

        # Incorporar experiencias nuevas a la memoria
        self.remember(X_new, y_new)

        return {
            "forgetting": forgetting,
            "exploration": exploration,
            "memory_size": len(self.memory)
        }
