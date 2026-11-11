
import sys
from pathlib import Path

import numpy as np

# Ubicamos la raíz del proyecto
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "IA"))

from model import NeuralNetwork
from dataset import generate_dataset


# 1. Generamos datos reproducibles
X, y = generate_dataset(n_samples=1000, seed=42)

# 2. Mezclamos y separamos los ejemplos
rng = np.random.default_rng(123)
indices = rng.permutation(len(X))

train_idx = indices[:800]
test_idx = indices[800:]

X_train = X[train_idx]
y_train = y[train_idx]

X_test = X[test_idx]
y_test = y[test_idx]

# 3. Creamos la red neuronal
model = NeuralNetwork(seed=42)

# 4. Entrenamos con 800 ejemplos
print("ENTRENAMIENTO DEL MODELO BASE")
model.train(X_train, y_train, epochs=2000, lr=0.1)

# 5. Evaluamos con 200 ejemplos no utilizados
probabilities = model.predict(X_test)
predictions = (probabilities >= 0.5).astype(int)

accuracy = np.mean(predictions == y_test) * 100

print(f"\nPrecisión en prueba: {accuracy:.2f}%")

# 6. Guardamos los pesos iniciales compartidos
output = ROOT / "shared" / "base_weights.npz"
model.save(output)

print(f"Pesos guardados en: {output}")
