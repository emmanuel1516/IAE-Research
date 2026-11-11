
import numpy as np


class NeuralNetwork:
    def __init__(self, seed=42):
        rng = np.random.default_rng(seed)

        # Capa de entrada (2) -> capa oculta (16)
        self.W1 = rng.normal(0, 0.5, (2, 16))
        self.b1 = np.zeros((1, 16))

        # Capa oculta (16) -> salida (1)
        self.W2 = rng.normal(0, 0.5, (16, 1))
        self.b2 = np.zeros((1, 1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        # Primera capa: transformación no lineal
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)

        # Segunda capa: probabilidad de categoría 1
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def train(self, X, y, epochs=2000, lr=0.1):
        for epoch in range(epochs):
            prediction = self.predict(X)

            # Error de clasificación
            p = np.clip(prediction, 1e-8, 1 - 1e-8)
            loss = -np.mean(
                y * np.log(p) +
                (1 - y) * np.log(1 - p)
            )

            # Retropropagación
            m = len(X)
            dz2 = prediction - y
            dW2 = self.a1.T @ dz2 / m
            db2 = np.mean(dz2, axis=0, keepdims=True)

            dz1 = (dz2 @ self.W2.T) * (1 - self.a1 ** 2)
            dW1 = X.T @ dz1 / m
            db1 = np.mean(dz1, axis=0, keepdims=True)

            # Actualización de pesos
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2

            if epoch % 200 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

    def save(self, path):
        np.savez(
            path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2
        )
        
    
    def load(self, path):
        with np.load(path) as data:
            self.W1 = data["W1"].copy()
            self.b1 = data["b1"].copy()
            self.W2 = data["W2"].copy()
            self.b2 = data["b2"].copy()

