
import numpy as np

from IA.model import NeuralNetwork
from IAE.memory import ExperienceMemory
from IAE.exploration import WeightExplorer
from shared.dataset import generate_dataset


PHASES = [0.0, 0.5, 1.0, 1.5]
SEEDS = [10, 20, 30, 40, 50]

EPOCHS = 500
LR = 0.1
MEMORY_CAPACITY = 500
REPLAY_SIZE = 400


def accuracy(model, X, y):
    predictions = (model.predict(X) >= 0.5).astype(int)
    return float(np.mean(predictions == y) * 100)


def loss(model, X, y):
    p = np.clip(model.predict(X), 1e-8, 1 - 1e-8)

    return float(-np.mean(
        y * np.log(p) + (1 - y) * np.log(1 - p)
    ))


def get_weights(model):
    return {
        name: getattr(model, name).copy()
        for name in ("W1", "b1", "W2", "b2")
    }


def set_weights(model, weights):
    for name, value in weights.items():
        setattr(model, name, value.copy())


def run(seed):
    replay = NeuralNetwork(seed=seed)
    iae = NeuralNetwork(seed=seed)

    # Mismos pesos iniciales
    set_weights(iae, get_weights(replay))

    # Memorias independientes, pero con la misma semilla
    replay_memory = ExperienceMemory(
        capacity=MEMORY_CAPACITY,
        seed=seed
    )
    iae_memory = ExperienceMemory(
        capacity=MEMORY_CAPACITY,
        seed=seed
    )

    explorer = WeightExplorer(
        noise_scale=0.02,
        candidates=20,
        seed=seed
    )

    # Pruebas independientes para cada fase
    tests = [
        generate_dataset(
            1000,
            seed + 10000 + i,
            phase=phase
        )
        for i, phase in enumerate(PHASES)
    ]

    # Matrices: filas = etapas de entrenamiento,
    # columnas = patrones evaluados.
    replay_scores = np.zeros((4, 4))
    iae_scores = np.zeros((4, 4))

    for i, phase in enumerate(PHASES):
        X_new, y_new = generate_dataset(
            800,
            seed + i * 100,
            phase=phase
        )

        # Obtenemos exactamente el mismo replay
        # para ambos modelos.
        if len(replay_memory) > 0:
            X_old, y_old = replay_memory.sample(
                REPLAY_SIZE,
                seed=seed + i
            )

            X_train = np.vstack([X_new, X_old])
            y_train = np.vstack([y_new, y_old])
        else:
            X_train = X_new
            y_train = y_new

        # Ambos entrenan con los mismos datos.
        replay.train(
            X_train, y_train,
            epochs=EPOCHS,
            lr=LR
        )
        iae.train(
            X_train, y_train,
            epochs=EPOCHS,
            lr=LR
        )

        # Validación independiente, equilibrada
        # entre las fases aprendidas hasta ahora.
        validation_sets = [
            generate_dataset(
                200,
                seed + 5000 + j,
                phase=PHASES[j]
            )
            for j in range(i + 1)
        ]

        X_val = np.vstack([
            data[0] for data in validation_sets
        ])
        y_val = np.vstack([
            data[1] for data in validation_sets
        ])

        explorer.explore(iae, X_val, y_val)

        # Guardamos nuevas experiencias en ambas
        # memorias, en el mismo orden.
        replay_memory.add(X_new, y_new)
        iae_memory.add(X_new, y_new)

        for j, (X_test, y_test) in enumerate(tests):
            replay_scores[i, j] = accuracy(
                replay, X_test, y_test
            )
            iae_scores[i, j] = accuracy(
                iae, X_test, y_test
            )

    return replay_scores, iae_scores


def print_matrix(title, matrix):
    print(f"\n{title}")
    print("Etapa       A       B       C       D")

    for i, row in enumerate(matrix):
        values = "  ".join(f"{v:6.2f}" for v in row)
        print(f"{'ABCD'[i]}       {values}")


all_replay = []
all_iae = []

for seed in SEEDS:
    print(f"\n===== SEMILLA {seed} =====")

    replay_scores, iae_scores = run(seed)

    all_replay.append(replay_scores)
    all_iae.append(iae_scores)


mean_replay = np.mean(all_replay, axis=0)
mean_iae = np.mean(all_iae, axis=0)

print_matrix("REPLAY - PRECISIÓN PROMEDIO", mean_replay)
print_matrix("IAE - PRECISIÓN PROMEDIO", mean_iae)

print_matrix(
    "DIFERENCIA IAE - REPLAY",
    mean_iae - mean_replay
)

print("\n===== RESULTADOS FINALES =====")

for name, matrix in [
    ("Replay", mean_replay),
    ("IAE", mean_iae)
]:
    final = matrix[-1]

    print(
        f"{name}: "
        f"A={final[0]:.2f}% "
        f"B={final[1]:.2f}% "
        f"C={final[2]:.2f}% "
        f"D={final[3]:.2f}% "
        f"Promedio={np.mean(final):.2f}%"
    )

    # Olvido en fases anteriores:
    # mejor precisión desde que se aprendió
    # cada fase hasta la etapa final.
    forgetting = []

    for j in range(3):
        best = np.max(matrix[j:, j])
        final_score = matrix[-1, j]
        forgetting.append(max(0, best - final_score))

    print(
        f"Olvido promedio A/B/C: "
        f"{np.mean(forgetting):.2f} puntos"
    )
