
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "IA"))

from model import NeuralNetwork

# Creamos dos modelos independientes
ia = NeuralNetwork(seed=1)
iae = NeuralNetwork(seed=999)

# Cargamos exactamente los mismos pesos
weights_path = ROOT / "shared" / "base_weights.npz"

ia.load(weights_path)
iae.load(weights_path)

# Comparamos los parámetros
same_weights = all(
    np.array_equal(getattr(ia, name), getattr(iae, name))
    for name in ("W1", "b1", "W2", "b2")
)

print("¿Los modelos tienen los mismos pesos?", same_weights)

# Comparamos predicciones
point = np.array([[1.2, 0.5]])

print("Predicción IA:", ia.predict(point))
print("Predicción IAE:", iae.predict(point))
