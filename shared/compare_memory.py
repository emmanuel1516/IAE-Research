
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "IA"))
sys.path.insert(0, str(ROOT / "IAE"))
sys.path.insert(0, str(ROOT / "shared"))

from model import NeuralNetwork
from memory import ExperienceMemory
from dataset import generate_dataset


# 1. Datos nuevos: patrón B
X_new, y_new = generate_dataset(
    n_samples=800,
    seed=100,
    phase=1.5
)

# 2. Datos antiguos: patrón A
X_old, y_old = generate_dataset(
    n_samples=500,
    seed=101,
    phase=0.0
)

# 3. Pruebas independientes
X_test_A, y_test_A = generate_dataset(
    n_samples=1000,
    seed=200,
    phase=0.0
)

X_test_B, y_test_B = generate_dataset(
    n_samples=1000,
    seed=201,
    phase=1.5
)

# 4. Creamos modelos idénticos
ia = NeuralNetwork()
iae = NeuralNetwork()

weights_path = ROOT / "shared" / "base_weights.npz"

ia.load(weights_path)
iae.load(weights_path)


def accuracy(model, X, y):
    predictions = (model.predict(X) >= 0.5).astype(int)
    return np.mean(predictions == y) * 100


def evaluate(name, model):
    print(f"\n{name}")
    print(f"Patrón A: {accuracy(model, X_test_A, y_test_A):.2f}%")
    print(f"Patrón B: {accuracy(model, X_test_B, y_test_B):.2f}%")


print("=== ANTES DE ADAPTARSE ===")
evaluate("IA", ia)
evaluate("IAE", iae)


# 5. IA convencional: aprende solo B
print("\n=== ENTRENANDO IA ===")
ia.train(X_new, y_new, epochs=1000, lr=0.1)


# 6. IAE: aprende B y repasa A
memory = ExperienceMemory(capacity=500)
memory.add(X_old, y_old)

X_replay, y_replay = memory.sample(n=400)

X_combined = np.vstack([X_new, X_replay])
y_combined = np.vstack([y_new, y_replay])

print("\n=== ENTRENANDO IAE CON MEMORIA ===")
iae.train(X_combined, y_combined, epochs=1000, lr=0.1)


# 7. Resultados
print("\n=== RESULTADOS FINALES ===")
evaluate("IA", ia)
evaluate("IAE con memoria", iae)
