import matplotlib.pyplot as plt
from matplotlib.pyplot import streamplot
from typing import Tuple
import numpy as np
from PIL import Image
import glob
import re
import collections
from pygifsicle import optimize

import matplotlib.cm as cm


def visualize_velocity_streamplot(velocity_field: np.ndarray, lattice_grid_shape: Tuple[int, int]):
    """
    Visualizies the velocity as streamplot

    Args:
        velocity_field: velocity
        lattice_grid_shape: lattice grid shape, e.g. [10,10]

    """
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
    """
    Visualizies the velocity as quiver

    Args:
        velocity_field: velocity
        lattice_grid_shape: lattice grid shape, e.g. [10,10]

    """
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
    """
    Visualizies the density as surface plot

    Args:
        density: density function
        lattice_grid_shape: lattice grid shape, e.g. [10,10]
        cmap: color map

    """
    y, x = np.mgrid[0:lattice_grid_shape[0], 0:lattice_grid_shape[1]]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, density, cmap=cmap)
    m = cm.ScalarMappable(cmap=cmap)
    m.set_array(density)
    plt.title('Surface Plot of Density Function')
    plt.xlabel('x')
    plt.ylabel('y')
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


def pngs_to_gif():
    """
    Creates gif for the von karman vortex street given the saved pngs
    """
    # Create the frames
    frames = []
    imgs = glob.glob(r"./figures/von_karman_vortex_shedding/all_png_parallel/*.png")
    regex = re.compile(r'\d+')
    numbers = [int(x) for img in imgs for x in regex.findall(img)]

    img_dict = {
        img: number for img, number in zip(imgs, numbers)
    }

    ordered_img_dict = collections.OrderedDict(sorted(img_dict.items(), key=lambda item: item[1]))

    for img, _ in ordered_img_dict.items():
        new_frame = Image.open(img)
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save('./figures/von_karman_vortex_shedding/png_to_gif.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=0.5, loop=0)

    optimize(r'./figures/von_karman_vortex_shedding/png_to_gif.gif')
