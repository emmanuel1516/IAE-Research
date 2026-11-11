
from IAE.memory_controller import MemoryController


controller = MemoryController()

cases = [
    (0.00, 0.5),
    (0.01, 0.5),
    (0.02, 0.7),
    (0.05, 0.7),
    (0.10, 0.9),
    (0.176, 0.9),
    (-0.03, 0.5)
]

print("=== TEST DEL CONTROLADOR DE MEMORIA ===")

for loss_increase, expected in cases:
    weight = controller.calculate_weight(
        loss_increase
    )

    print(
        f"Aumento de error: {loss_increase:+.3f} "
        f"-> Peso de memoria: {weight:.2f}"
    )

    assert weight == expected, (
        f"Esperado {expected}, obtenido {weight}"
    )

print("\nTodas las verificaciones pasaron.")
