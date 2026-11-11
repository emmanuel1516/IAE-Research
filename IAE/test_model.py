
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))

from dataset import generate_dataset
from IAE.model import EvolutionaryAI


def accuracy(model, X, y):
    predictions = (model.predict(X) >= 0.5).astype(int)
    return np.mean(predictions == y) * 100


# 1. Creamos IAE y cargamos los pesos iniciales
iae = EvolutionaryAI(
    memory_capacity=500,
    noise_scale=0.02,
    candidates=20,
    seed=42
)

iae.load(ROOT / "shared" / "base_weights.npz")


# 2. Experiencias antiguas: patrón A
X_old, y_old = generate_dataset(
    n_samples=400,
    seed=101,
    phase=0.0
)

iae.remember(X_old, y_old)


# 3. Datos nuevos: patrón B
X_new, y_new = generate_dataset(
    n_samples=800,
    seed=100,
    phase=1.5
)


# 4. Datos exclusivos para validación
X_val, y_val = generate_dataset(
    n_samples=300,
    seed=500,
    phase=1.5
)


# 5. Datos exclusivos para prueba final
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


print("=== ANTES DE ADAPTARSE ===")
print(f"Patrón A: {accuracy(iae, X_test_A, y_test_A):.2f}%")
print(f"Patrón B: {accuracy(iae, X_test_B, y_test_B):.2f}%")


# 6. Entrenamiento y exploración
iae.adapt(
    X_new,
    y_new,
    X_val,
    y_val,
    replay_size=400,
    epochs=1000,
    lr=0.1
)


# 7. Evaluación final
print("\n=== DESPUÉS DE ADAPTARSE ===")

accuracy_A = accuracy(iae, X_test_A, y_test_A)
accuracy_B = accuracy(iae, X_test_B, y_test_B)

print(f"Patrón A: {accuracy_A:.2f}%")
print(f"Patrón B: {accuracy_B:.2f}%")
print(f"Promedio: {(accuracy_A + accuracy_B) / 2:.2f}%")


print("\n=== MEMORIA ===")
print(f"Experiencias guardadas: {len(iae.memory)}")
