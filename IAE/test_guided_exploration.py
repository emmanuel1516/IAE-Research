
from pathlib import Path

from IA.model import NeuralNetwork
from IAE.exploration import MemoryGuidedExplorer
from shared.dataset import generate_dataset


ROOT = Path(__file__).resolve().parents[1]

model = NeuralNetwork(seed=42)
model.load(ROOT / "shared" / "base_weights.npz")

# Recuerdos antiguos
X_old, y_old = generate_dataset(
    400, seed=101, phase=0.0
)

# Datos nuevos
X_new, y_new = generate_dataset(
    400, seed=102, phase=1.5
)

explorer = MemoryGuidedExplorer(
    noise_scale=0.02,
    candidates=20,
    memory_weight=0.5,
    seed=42
)

result = explorer.explore_guided(
    model,
    X_old,
    y_old,
    X_new,
    y_new
)

print("=== ANTES DE EXPLORAR ===")
print(result["initial"])

print("\n=== DESPUÉS DE EXPLORAR ===")
print(result["best"])

print("\n=== RESULTADO ===")
print(f"¿Mejoró?: {result['improved']}")
print(f"Evaluaciones: {result['evaluations']}")

assert (
    result["best"]["total_loss"]
    <= result["initial"]["total_loss"]
)

assert result["evaluations"] == 21

print("\nTodas las verificaciones pasaron.")
