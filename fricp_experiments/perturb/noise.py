import numpy as np

def add_gaussian_noise(vertices, normals, sigma=0.01):
    """
    Adds Gaussian noise to the vertices.
    """
    noise = np.random.normal(0, sigma, vertices.shape)
    return vertices + noise, normals
