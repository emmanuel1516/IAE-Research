
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "IA"))
sys.path.insert(0, str(ROOT / "shared"))

from model import NeuralNetwork
from dataset import generate_dataset
from exploration import WeightExplorer


# 1. Recuperamos nuestra red neuronal entrenada
model = NeuralNetwork(seed=42)
model.load(ROOT / "shared" / "base_weights.npz")

# 2. Generamos datos de validación del patrón B
X_val, y_val = generate_dataset(
    n_samples=300,
    seed=500,
    phase=1.5
)

# 3. Guardamos los pesos antes de explorar
original_weights = {
    name: getattr(model, name).copy()
    for name in ("W1", "b1", "W2", "b2")
}

# 4. Creamos el explorador
explorer = WeightExplorer(
    noise_scale=0.02,
    candidates=20,
    seed=42
)

# 5. Exploramos modificaciones
result = explorer.explore(model, X_val, y_val)

# 6. Comprobamos si los pesos cambiaron
weights_changed = any(
    not np.array_equal(
        original_weights[name],
        getattr(model, name)
    )
    for name in original_weights
)

print("\n=== RESULTADOS DE EXPLORACIÓN ===")
print(f"Error inicial: {result['initial_loss']:.6f}")
print(f"Mejor error: {result['best_loss']:.6f}")
print(f"¿Mejoró?: {result['improved']}")
print(f"¿Cambió los pesos?: {weights_changed}")
print(f"Evaluaciones realizadas: {result['evaluations']}")
