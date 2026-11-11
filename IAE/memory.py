
import numpy as np


class ExperienceMemory:
    def __init__(self, capacity=500, seed=42):
        if capacity <= 0:
            raise ValueError("La capacidad debe ser positiva")

        self.capacity = capacity
        self.X = []
        self.y = []

        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def add(self, X, y):
        for point, label in zip(X, y):
            self.seen += 1

            if len(self.X) < self.capacity:
                self.X.append(point.copy())
                self.y.append(label.copy())

            else:
                # Cada experiencia vista tiene la misma
                # probabilidad de quedar en el reservorio.
                index = self.rng.integers(0, self.seen)

                if index < self.capacity:
                    self.X[index] = point.copy()
                    self.y[index] = label.copy()

    def sample(self, n=100, seed=None):
        if len(self.X) == 0:
            raise ValueError("La memoria está vacía")

        if n <= 0:
            raise ValueError("n debe ser positivo")

        # Sin seed: muestras diferentes en cada llamada.
        # Con seed: selección reproducible.
        rng = (
            np.random.default_rng(seed)
            if seed is not None
            else self.rng
        )

        indices = rng.choice(
            len(self.X),
            size=min(n, len(self.X)),
            replace=False
        )

        return (
            np.array(self.X)[indices],
            np.array(self.y)[indices]
        )

    def __len__(self):
        return len(self.X)
