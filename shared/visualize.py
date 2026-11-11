
import numpy as np
import matplotlib.pyplot as plt

from dataset import generate_dataset


# Generamos los mismos 1000 ejemplos
X, y = generate_dataset(
    n_samples=1000,
    seed=42
)

# Separamos las categorías
category_0 = y[:, 0] == 0
category_1 = y[:, 0] == 1

# Dibujamos los puntos
plt.scatter(
    X[category_0, 0],
    X[category_0, 1],
    label="Categoría 0",
    alpha=0.6
)

plt.scatter(
    X[category_1, 0],
    X[category_1, 1],
    label="Categoría 1",
    alpha=0.6
)

# Dibujamos la frontera real
x = np.linspace(-2, 2, 300)
boundary = np.sin(2 * x)

plt.plot(
    x,
    boundary,
    color="black",
    linewidth=2,
    label="Frontera real"
)

plt.title("IAE Research - Dataset inicial")
plt.xlabel("Coordenada X1")
plt.ylabel("Coordenada X2")
plt.legend()
plt.grid(alpha=0.2)

plt.show()
