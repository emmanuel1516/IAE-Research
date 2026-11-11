
from pathlib import Path
from time import perf_counter

import numpy as np

from IA.model import NeuralNetwork
from IAE.exploration import WeightExplorer
from shared.dataset import generate_dataset


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "shared" / "base_weights.npz"

SEEDS = [10, 20, 30, 40, 50]


def metrics(model, X, y):
    probabilities = model.predict(X)
    predictions = (probabilities >= 0.5).astype(int)

    p = np.clip(probabilities, 1e-8, 1 - 1e-8)
    loss = -np.mean(
        y * np.log(p) + (1 - y) * np.log(1 - p)
    )

    accuracy = np.mean(predictions == y) * 100

    return float(loss), float(accuracy)


def run(seed):
    # Datos de entrenamiento
    X_B, y_B = generate_dataset(
        800, seed, phase=1.5
    )
    X_A, y_A = generate_dataset(
        400, seed + 1000, phase=0.0
    )

    X_train = np.vstack([X_B, X_A])
    y_train = np.vstack([y_B, y_A])

    # Validación: permite seleccionar pesos
    X_val_A, y_val_A = generate_dataset(
        300, seed + 2000, phase=0.0
    )
    X_val_B, y_val_B = generate_dataset(
        300, seed + 3000, phase=1.5
    )

    X_val = np.vstack([X_val_A, X_val_B])
    y_val = np.vstack([y_val_A, y_val_B])

    # Prueba final: no se utiliza para seleccionar
    X_test_A, y_test_A = generate_dataset(
        1000, seed + 4000, phase=0.0
    )
    X_test_B, y_test_B = generate_dataset(
        1000, seed + 5000, phase=1.5
    )

    X_test = np.vstack([X_test_A, X_test_B])
    y_test = np.vstack([y_test_A, y_test_B])

    # Modelo base con memoria
    model = NeuralNetwork()
    model.load(WEIGHTS)

    model.train(
        X_train, y_train,
        epochs=1000,
        lr=0.1
    )

    # Métricas antes de explorar
    val_before = metrics(model, X_val, y_val)
    test_before = metrics(model, X_test, y_test)

    explorer = WeightExplorer(
        noise_scale=0.02,
        candidates=20,
        seed=seed
    )

    start = perf_counter()

    result = explorer.explore(
        model, X_val, y_val
    )

    exploration_time = perf_counter() - start

    # Métricas después de explorar
    val_after = metrics(model, X_val, y_val)
    test_after = metrics(model, X_test, y_test)

    return {
        "val_loss_change": val_after[0] - val_before[0],
        "test_loss_change": test_after[0] - test_before[0],
        "test_accuracy_change": test_after[1] - test_before[1],
        "exploration_time": exploration_time,
        "evaluations": result["evaluations"]
    }


results = []

for seed in SEEDS:
    print(f"\n===== SEMILLA {seed} =====")

    result = run(seed)
    results.append(result)

    print(
        f"Cambio loss validación: "
        f"{result['val_loss_change']:+.6f}"
    )
    print(
        f"Cambio loss prueba: "
        f"{result['test_loss_change']:+.6f}"
    )
    print(
        f"Cambio accuracy prueba: "
        f"{result['test_accuracy_change']:+.2f} puntos"
    )
    print(
        f"Tiempo exploración: "
        f"{result['exploration_time']:.4f} segundos"
    )


print("\n===== PROMEDIOS =====")

for key in results[0]:
    values = [r[key] for r in results]
    print(f"{key}: {np.mean(values):.6f}")
