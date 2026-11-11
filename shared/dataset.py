
import numpy as np


def generate_dataset(n_samples=1000, seed=42, phase=0.0):
    rng = np.random.default_rng(seed)

    # Generamos puntos con dos coordenadas
    X = rng.uniform(-2, 2, size=(n_samples, 2))

    # Extraemos las coordenadas
    x1 = X[:, 0]
    x2 = X[:, 1]

    # Regla que define las categorías
    boundary = np.sin(2 * x1 + phase)

    # Asignamos la categoría correspondiente
    y = (x2 > boundary).astype(int)

    # La salida tendrá forma (n_samples, 1)
    y = y.reshape(-1, 1)

    return X, y


if __name__ == "__main__":
    X, y = generate_dataset()

    print("Primeros 5 puntos:")
    print(X[:5])

    print("\nSus categorías:")
    print(y[:5])

    print("\nCantidad de ejemplos:", len(X))
