import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import streamplot
import typing
from typing import Tuple
import numpy as np

import matplotlib.cm as cm


def visualize_velocity_streamplot(velocity_field: np.ndarray, lattice_grid_shape: Tuple[int, int]):
    assert velocity_field.shape[-1] == 2
    y, x = np.mgrid[0:lattice_grid_shape[0], 0:lattice_grid_shape[1]]
    streamplot(
        x,
        y,
        velocity_field[..., 0],
        velocity_field[..., 1],
        color=np.linalg.norm(velocity_field, axis=-1),
        cmap='seismic'
    )
    plt.xlim(0, lattice_grid_shape[0] - 1)
    plt.ylim(0, lattice_grid_shape[1] - 1)
    plt.colorbar()
    plt.show()


def visualize_velocity_quiver(velocity_field: np.ndarray, lattice_grid_shape: Tuple[int, int]):
    assert velocity_field.shape[-1] == 2
    y, x = np.mgrid[0:lattice_grid_shape[0], 0:lattice_grid_shape[1]]
    plt.quiver(
        x,
        y,
        velocity_field[..., 0],
        velocity_field[..., 1]
    )
    plt.xlim(0, lattice_grid_shape[0] - 1)
    plt.ylim(0, lattice_grid_shape[1] - 1)
    plt.show()


def visualize_density_surface_plot(density: np.ndarray, lattice_grid_shape: Tuple[int, int], cmap='seismic'):
    y, x = np.mgrid[0:lattice_grid_shape[0], 0:lattice_grid_shape[1]]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, density, cmap=cmap)
    m = cm.ScalarMappable(cmap=cmap)
    m.set_array(density)
    plt.title('Surface Plot of Density Function')
    plt.colorbar(m)
    plt.show()


def visualize_density_contour_plot(density: np.ndarray, lattice_grid_shape: Tuple[int, int], cmap='seismic'):
    """
    Visualizies the density as contour plot
    Args:
        density: density function
        lattice_grid_shape: lattice grid shape, e.g. [10,10]
        cmap: color map

    Returns:

    """
    y, x = np.mgrid[0:lattice_grid_shape[0], 0:lattice_grid_shape[1]]
    plt.contourf(x, y, density, cmap=cmap)
    plt.title('Contour Plot of Density Function')
    plt.colorbar()
    plt.show()
