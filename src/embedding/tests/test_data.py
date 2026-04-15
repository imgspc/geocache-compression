import numpy as np
import math


def complex_data(nreps: int = 1000, nsamples: int = 30) -> np.ndarray:
    """
    Generate a matrix of 3d float32 data.
    """
    # We have three movement vectors, and one origin.
    # We create nreps particles going in each direction at variable
    # speed in (0..1) based on their index, for nsamples timesteps.
    base_vectors = np.array([[1, 1, 1], [0.5, 0.5, 1], [-1, 1, -1]])
    vectors = np.tile(base_vectors, (nreps, 1))
    nverts = len(vectors)
    indices = np.arange(nverts)
    speeds = indices / nverts
    velocities = speeds[:, np.newaxis] * vectors

    origin = np.array([10, 0, -10])

    # there's obviously a nicer way to write this but it's escaping me
    return np.array([origin + sample * velocities for sample in range(nsamples)])
