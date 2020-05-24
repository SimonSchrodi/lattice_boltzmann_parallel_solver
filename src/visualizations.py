import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import streamplot
import typing
from typing import Tuple
import numpy as np


def visualize_velocity_field(velocity_field: np.ndarray, lattice_grid_shape: Tuple[int, int]):
    assert velocity_field.shape[-1] == 2
    y, x = np.mgrid[0:lattice_grid_shape[0], 0:lattice_grid_shape[1]]
    streamplot(
        x,
        y,
        velocity_field[..., 0],
        velocity_field[..., 1]
    )
    plt.show()
