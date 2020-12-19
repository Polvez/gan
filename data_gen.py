import numpy as np


def generate_sin_data(n: int = 10000) -> np.array:
    """Generate data samples from sinus function."""
    x = np.linspace(-np.pi, np.pi, n)
    y = np.sin(x)
    return np.array([[i, j] for i, j in zip(x, y)])


def get_y(x):
    return 10 + x*x


def sample_data(n=10000, scale=100):
    data = []

    x = scale*(np.random.random_sample((n,))-0.5)

    for i in range(n):
        yi = get_y(x[i])
        data.append([x[i], yi])

    return np.array(data)
