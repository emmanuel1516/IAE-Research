
import numpy as np


class WeightExplorer:
    def __init__(self, noise_scale=0.02, candidates=10, seed=42):
        self.noise_scale = noise_scale
        self.candidates = candidates
        self.rng = np.random.default_rng(seed)

    def get_weights(self, model):
        return {
            "W1": model.W1.copy(),
            "b1": model.b1.copy(),
            "W2": model.W2.copy(),
            "b2": model.b2.copy()
        }

    def set_weights(self, model, weights):
        model.W1 = weights["W1"].copy()
        model.b1 = weights["b1"].copy()
        model.W2 = weights["W2"].copy()
        model.b2 = weights["b2"].copy()

    def mutate(self, weights):
        mutated = {}

        for name, matrix in weights.items():
            noise = self.rng.normal(
                loc=0.0,
                scale=self.noise_scale,
                size=matrix.shape
            )

            mutated[name] = matrix + noise

        return mutated

    def evaluate(self, model, X, y):
        predictions = model.predict(X)

        p = np.clip(predictions, 1e-8, 1 - 1e-8)

        loss = -np.mean(
            y * np.log(p) +
            (1 - y) * np.log(1 - p)
        )

        return float(loss)

    def explore(self, model, X_val, y_val):
        original = self.get_weights(model)

        best_weights = original
        best_loss = self.evaluate(model, X_val, y_val)
        initial_loss = best_loss

        for _ in range(self.candidates):
            candidate = self.mutate(original)

            self.set_weights(model, candidate)

            candidate_loss = self.evaluate(
                model, X_val, y_val
            )

            if candidate_loss < best_loss:
                best_loss = candidate_loss
                best_weights = candidate

        self.set_weights(model, best_weights)

        return {
            "initial_loss": initial_loss,
            "best_loss": best_loss,
            "improved": best_loss < initial_loss,
            "evaluations": self.candidates + 1
        }


class MemoryGuidedExplorer(WeightExplorer):
    def __init__(
        self,
        noise_scale=0.02,
        candidates=20,
        memory_weight=0.5,
        seed=42
    ):
        super().__init__(
            noise_scale=noise_scale,
            candidates=candidates,
            seed=seed
        )

        if not 0 <= memory_weight <= 1:
            raise ValueError(
                "memory_weight debe estar entre 0 y 1"
            )

        self.memory_weight = memory_weight

    def evaluate_balanced(
        self,
        model,
        X_old,
        y_old,
        X_new,
        y_new
    ):
        old_loss = self.evaluate(
            model, X_old, y_old
        )

        new_loss = self.evaluate(
            model, X_new, y_new
        )

        total_loss = (
            self.memory_weight * old_loss
            + (1 - self.memory_weight) * new_loss
        )

        return {
            "old_loss": old_loss,
            "new_loss": new_loss,
            "total_loss": total_loss
        }

    def explore_guided(
        self,
        model,
        X_old,
        y_old,
        X_new,
        y_new
    ):
        original = self.get_weights(model)

        best_weights = original

        initial = self.evaluate_balanced(
            model, X_old, y_old, X_new, y_new
        )

        best = initial.copy()

        for _ in range(self.candidates):
            candidate = self.mutate(original)

            self.set_weights(model, candidate)

            score = self.evaluate_balanced(
                model, X_old, y_old, X_new, y_new
            )

            if score["total_loss"] < best["total_loss"]:
                best = score.copy()
                best_weights = candidate

        self.set_weights(model, best_weights)

        return {
            "initial": initial,
            "best": best,
            "improved": (
                best["total_loss"] < initial["total_loss"]
            ),
            "evaluations": self.candidates + 1
        }
