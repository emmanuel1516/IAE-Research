
import numpy as np

from IAE.memory import ExperienceMemory


memory = ExperienceMemory(
    capacity=500,
    seed=42
)

# Simulamos cuatro etapas de aprendizaje.
# La primera columna identifica la etapa.
for phase in range(4):
    X = np.column_stack([
        np.full(1000, phase),
        np.arange(1000)
    ])

    y = np.full((1000, 1), phase)

    memory.add(X, y)

    print(f"\nDespués de aprender fase {phase}:")
    print(f"Experiencias vistas: {memory.seen}")
    print(f"Recuerdos guardados: {len(memory)}")

    phases, counts = np.unique(
        np.array(memory.y).flatten(),
        return_counts=True
    )

    for p, count in zip(phases, counts):
        print(f"Fase {int(p)}: {count} recuerdos")


print("\n=== VERIFICACIONES ===")

assert len(memory) == 500
assert memory.seen == 4000

stored_phases = set(
    np.array(memory.y).flatten().astype(int)
)

assert stored_phases == {0, 1, 2, 3}

X_sample, y_sample = memory.sample(100)

assert len(X_sample) == 100
assert len(y_sample) == 100

print("Todas las verificaciones pasaron.")
