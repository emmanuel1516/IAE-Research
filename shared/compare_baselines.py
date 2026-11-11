
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "IA"))
sys.path.insert(0, str(ROOT / "shared"))

from model import NeuralNetwork
from dataset import generate_dataset


def accuracy(model, X, y):
    predictions = (model.predict(X) >= 0.5).astype(int)
    return np.mean(predictions == y) * 100


def evaluate(model, tests):
    return {
        name: accuracy(model, X, y)
        for name, (X, y) in tests.items()
    }


def run_experiment(seed):
    # Datos de adaptación
    X_B, y_B = generate_dataset(
        n_samples=800, seed=seed, phase=1.5
    )

    # Experiencias antiguas disponibles
    X_A, y_A = generate_dataset(
        n_samples=400, seed=seed + 1000, phase=0.0
    )

    # Pruebas finales independientes
    tests = {
        "A": generate_dataset(
            1000, seed=seed + 2000, phase=0.0
        ),
        "B": generate_dataset(
            1000, seed=seed + 3000, phase=1.5
        )
    }

    weights_path = ROOT / "shared" / "base_weights.npz"

    ia = NeuralNetwork()
    replay = NeuralNetwork()

    ia.load(weights_path)
    replay.load(weights_path)

    # IA: solamente patrón B
    # 1200 ejemplos por época
    X_ia = np.vstack([X_B, X_B[:400]])
    y_ia = np.vstack([y_B, y_B[:400]])

    # IA con memoria: 800 nuevos + 400 antiguos
    X_replay = np.vstack([X_B, X_A])
    y_replay = np.vstack([y_B, y_A])

    ia.train(X_ia, y_ia, epochs=1000, lr=0.1)
    replay.train(X_replay, y_replay, epochs=1000, lr=0.1)

    return evaluate(ia, tests), evaluate(replay, tests)


results_ia = []
results_replay = []

for seed in [10, 20, 30, 40, 50]:
    print(f"\n===== SEMILLA {seed} =====")

    ia_result, replay_result = run_experiment(seed)

    results_ia.append(ia_result)
    results_replay.append(replay_result)

    print("IA:", ia_result)
    print("IA con memoria:", replay_result)


print("\n===== PROMEDIOS FINALES =====")

for name, results in [
    ("IA", results_ia),
    ("IA con memoria", results_replay)
]:
    avg_A = np.mean([r["A"] for r in results])
    avg_B = np.mean([r["B"] for r in results])

    print(f"\n{name}")
    print(f"Patrón A: {avg_A:.2f}%")
    print(f"Patrón B: {avg_B:.2f}%")
    print(f"Promedio: {(avg_A + avg_B) / 2:.2f}%")
