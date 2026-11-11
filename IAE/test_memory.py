
import numpy as np
from memory import ExperienceMemory

memory = ExperienceMemory(capacity=5)

X = np.array([
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0]
])

y = np.array([
    [0],
    [1],
    [0]
])

memory.add(X, y)

print("Experiencias guardadas:", len(memory))

X_old, y_old = memory.sample(n=2)

print("Puntos recuperados:")
print(X_old)

print("Categorías:")
print(y_old)
