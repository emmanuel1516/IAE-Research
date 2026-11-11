
from pathlib import Path

import numpy as np

from IA.model import NeuralNetwork
from IAE.model import EvolutionaryAI
from shared.dataset import generate_dataset


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "shared" / "base_weights.npz"


def accuracy(model, X, y):
    predictions = (model.predict(X) >= 0.5).astype(int)
    return float(np.mean(predictions == y) * 100)


def evaluate(model, tests):
    a = accuracy(model, *tests["A"])
    b = accuracy(model, *tests["B"])
    return a, b, (a + b) / 2


def experiment(seed):
    # Entrenamiento
    X_B, y_B = generate_dataset(800, seed, phase=1.5)
    X_A, y_A = generate_dataset(400, seed + 1000, phase=0.0)

    # Validación: usada para elegir candidatos
    X_val_A, y_val_A = generate_dataset(
        300, seed + 2000, phase=0.0
    )
    X_val_B, y_val_B = generate_dataset(
        300, seed + 3000, phase=1.5
    )

    X_val = np.vstack([X_val_A, X_val_B])
    y_val = np.vstack([y_val_A, y_val_B])

    # Prueba final: nunca utilizada para entrenar
    tests = {
        "A": generate_dataset(
            1000, seed + 4000, phase=0.0
        ),
        "B": generate_dataset(
            1000, seed + 5000, phase=1.5
        )
    }

    ia = NeuralNetwork()
    replay = NeuralNetwork()
    iae = EvolutionaryAI(
        memory_capacity=500,
        candidates=20,
        noise_scale=0.02,
        seed=seed
    )

    ia.load(WEIGHTS)
    replay.load(WEIGHTS)
    iae.load(WEIGHTS)

    # IA convencional: solo B
    X_only_B = np.vstack([X_B, X_B[:400]])
    y_only_B = np.vstack([y_B, y_B[:400]])

    # IA con memoria: A + B
    X_combined = np.vstack([X_B, X_A])
    y_combined = np.vstack([y_B, y_A])

    ia.train(X_only_B, y_only_B, epochs=1000, lr=0.1)
    replay.train(X_combined, y_combined, epochs=1000, lr=0.1)

    # IAE recibe exactamente los mismos ejemplos que replay
    iae.remember(X_A, y_A)

    iae.adapt(
        X_B, y_B,
        X_val, y_val,
        replay_size=400,
        epochs=1000,
        lr=0.1
    )

    return {
        "IA": evaluate(ia, tests),
        "Replay": evaluate(replay, tests),
        "IAE": evaluate(iae, tests)
    }


all_results = {
    "IA": [],
    "Replay": [],
    "IAE": []
}

for seed in [10, 20, 30, 40, 50]:
    print(f"\n===== EXPERIMENTO {seed} =====")

    results = experiment(seed)

    for name, values in results.items():
        all_results[name].append(values)
        print(
            f"{name}: A={values[0]:.2f}% "
            f"B={values[1]:.2f}% "
            f"Promedio={values[2]:.2f}%"
        )


print("\n===== RESULTADOS PROMEDIO =====")

for name, values in all_results.items():
    avg = np.mean(values, axis=0)

    print(
        f"{name}: A={avg[0]:.2f}% "
        f"B={avg[1]:.2f}% "
        f"Promedio={avg[2]:.2f}%"
    )
