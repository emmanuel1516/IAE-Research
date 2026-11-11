
from IAE.memory_monitor import MemoryMonitor


monitor = MemoryMonitor(tolerance=0.005)

before = {
    "loss": 0.18,
    "accuracy": 94.0
}

after_forgetting = {
    "loss": 0.23,
    "accuracy": 88.0
}

after_improvement = {
    "loss": 0.16,
    "accuracy": 95.0
}

result_1 = monitor.detect_forgetting(
    before, after_forgetting
)

result_2 = monitor.detect_forgetting(
    before, after_improvement
)

print("=== CASO 1: AUMENTO DEL ERROR ===")
print(result_1)

print("\n=== CASO 2: MEJORA ===")
print(result_2)

assert result_1["forgetting_detected"] is True
assert result_2["forgetting_detected"] is False

print("\nTodas las verificaciones pasaron.")
