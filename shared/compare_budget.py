
from pathlib import Path
from time import perf_counter

import numpy as np

from IA.model import NeuralNetwork
from IAE.exploration import WeightExplorer
from shared.dataset import generate_dataset


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "shared" / "base_weights.npz"
SEEDS = [10, 20, 30, 40, 50]


def loss_and_accuracy(model, X, y):
    p = np.clip(model.predict(X), 1e-8, 1 - 1e-8)

    loss = -np.mean(
        y * np.log(p) + (1 - y) * np.log(1 - p)
    )

    accuracy = np.mean(
        (p >= 0.5).astype(int) == y
    ) * 100

    return float(loss), float(accuracy)


def clone_model(weights):
    model = NeuralNetwork()
    model.load(weights)
    return model


def run(seed):
    X_B, y_B = generate_dataset(800, seed, phase=1.5)
    X_A, y_A = generate_dataset(
        400, seed + 1000, phase=0.0
    )

    X_train = np.vstack([X_B, X_A])
    y_train = np.vstack([y_B, y_A])

    X_val_A, y_val_A = generate_dataset(
        300, seed + 2000, phase=0.0
    )
    X_val_B, y_val_B = generate_dataset(
        300, seed + 3000, phase=1.5
    )

    X_val = np.vstack([X_val_A, X_val_B])
    y_val = np.vstack([y_val_A, y_val_B])

    X_test_A, y_test_A = generate_dataset(
        1000, seed + 4000, phase=0.0
    )
    X_test_B, y_test_B = generate_dataset(
        1000, seed + 5000, phase=1.5
    )

    X_test = np.vstack([X_test_A, X_test_B])
    y_test = np.vstack([y_test_A, y_test_B])

    # Entrenamiento común
    base = clone_model(WEIGHTS)
    base.train(X_train, y_train, epochs=1000, lr=0.1)

    # Guardamos exactamente el mismo punto de partida
    checkpoint = ROOT / "shared" / "_budget_temp.npz"
    base.save(checkpoint)

    replay = clone_model(checkpoint)
    gradient = clone_model(checkpoint)
    iae = clone_model(checkpoint)

    explorer = WeightExplorer(
        noise_scale=0.02,
        candidates=20,
        seed=seed
    )

    # Medimos cuánto tarda explorar
    start = perf_counter()
    explorer.explore(iae, X_val, y_val)
    exploration_time = perf_counter() - start

    # Medimos una época de gradientes.
    # Repetimos para reducir ruido de medición.
    timing_model = clone_model(checkpoint)

    start = perf_counter()
    timing_model.train(
        X_train, y_train,
        epochs=100,
        lr=0.1
    )
    epoch_time = (perf_counter() - start) / 100

    # Presupuesto aproximado equivalente en tiempo
    extra_epochs = max(
        1, round(exploration_time / epoch_time)
    )

    start = perf_counter()
    gradient.train(
        X_train, y_train,
        epochs=extra_epochs,
        lr=0.1
    )
    gradient_time = perf_counter() - start

    models = {
        "Replay": replay,
        "Gradientes": gradient,
        "IAE": iae
    }

    results = {}

    for name, model in models.items():
        test_loss, test_acc = loss_and_accuracy(
            model, X_test, y_test
        )
        results[name] = (test_loss, test_acc)

    return (
        results,
        extra_epochs,
        exploration_time,
        gradient_time
    )


all_results = {
    "Replay": [],
    "Gradientes": [],
    "IAE": []
}

for seed in SEEDS:
    print(f"\n===== SEMILLA {seed} =====")

    results, epochs, t_iae, t_grad = run(seed)

    print(f"Épocas adicionales: {epochs}")
    print(f"Tiempo exploración: {t_iae:.6f}s")
    print(f"Tiempo gradientes: {t_grad:.6f}s")

    for name, values in results.items():
        all_results[name].append(values)

        print(
            f"{name}: loss={values[0]:.6f} "
            f"accuracy={values[1]:.2f}%"
        )


print("\n===== RESULTADOS PROMEDIO =====")

for name, values in all_results.items():
    avg = np.mean(values, axis=0)

    print(
        f"{name}: loss={avg[0]:.6f} "
        f"accuracy={avg[1]:.2f}%"
    )

# Limpiamos el checkpoint temporal
checkpoint = ROOT / "shared" / "_budget_temp.npz"
if checkpoint.exists():
    checkpoint.unlink()
