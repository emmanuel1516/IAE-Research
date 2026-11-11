
import numpy as np


class MemoryMonitor:
    def __init__(self, tolerance=0.005):
        if tolerance < 0:
            raise ValueError("La tolerancia no puede ser negativa")

        self.tolerance = tolerance

    def evaluate(self, model, X, y):
        probabilities = np.clip(
            model.predict(X),
            1e-8,
            1 - 1e-8
        )

        loss = -np.mean(
            y * np.log(probabilities)
            + (1 - y) * np.log(1 - probabilities)
        )

        accuracy = np.mean(
            (probabilities >= 0.5).astype(int) == y
        ) * 100

        return {
            "loss": float(loss),
            "accuracy": float(accuracy)
        }

    def detect_forgetting(self, before, after):
        loss_increase = (
            after["loss"] - before["loss"]
        )

        accuracy_change = (
            after["accuracy"] - before["accuracy"]
        )

        forgetting_detected = (
            loss_increase > self.tolerance
        )

        return {
            "forgetting_detected": forgetting_detected,
            "loss_increase": loss_increase,
            "accuracy_change": accuracy_change
        }
