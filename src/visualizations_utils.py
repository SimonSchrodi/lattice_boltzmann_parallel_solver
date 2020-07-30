import matplotlib.pyplot as plt
from matplotlib.pyplot import streamplot
from typing import Tuple
import numpy as np
from PIL import Image
import glob
import re
import collections
from pygifsicle import optimize
import os

import matplotlib.cm as cm

from lattice_boltzmann_method import strouhal_number


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


def plot_reynolds_strouhal():
    """
    Creates plot for strouhal number as function of reynolds number
    """
    folder = r'./figures/von_karman_vortex_shedding/reynold_strouhal'
    d = 40
    strouhal = []
    reynolds = []
    for file in glob.glob(folder + r"/*.npy"):
        vel_at_p = np.load(file)
        plt.plot(vel_at_p)
        plt.savefig(file + '.svg')
        plt.close()

        reynolds.append(int(file[file.rfind('_') + 1:file.rfind('.npy')]))
        if reynolds[-1] == 40:
            vel_at_p = vel_at_p[150000:]
        elif reynolds[-1] == 70:
            vel_at_p = vel_at_p[125000:]
        else:
            vel_at_p = vel_at_p[70000:]

        vel_at_p -= np.mean(vel_at_p)
        yf = np.fft.fft(vel_at_p)
        freq = np.fft.fftfreq(len(vel_at_p), 1)
        vortex_frequency = np.abs(freq[np.argmax(np.abs(yf))])
        if reynolds[-1] > 130:
            visc = 0.03
            u = reynolds[-1] * visc / 40
            strouhal.append(strouhal_number(vortex_frequency, d, u))
        else:
            u = 0.1
            strouhal.append(strouhal_number(vortex_frequency, d, u))

    strouhal = [x for _, x in sorted(zip(reynolds, strouhal))]
    reynolds = np.sort(reynolds)
    plt.plot(reynolds, strouhal)
    plt.xlabel('Reynolds number')
    plt.ylabel('Strouhal number')
    plt.savefig(os.path.join(folder, "reynold_strouhal.svg"), bbox_inches='tight')
    plt.savefig(os.path.join(folder, "reynold_strouhal.pgf"), bbox_inches='tight')


def plot_nx_strouhal():
    """
    Creates plot for strouhal number as function of lattice grid width nx
    """
    folder = r'./figures/von_karman_vortex_shedding/nx_strouhal'
    d = 40
    u0 = 0.1
    strouhal = []
    lxs = []
    for file in glob.glob(folder + r"/*.npy"):
        vel_at_p = np.load(file)
        plt.plot(vel_at_p)
        plt.savefig(file + '.svg')
        plt.close()

        lxs.append(int(file[file.rfind('_') + 1:file.rfind('.npy')]))

        if lxs[-1] == 260:
            vel_at_p = vel_at_p[125000:]
        elif lxs[-1] == 300 or lxs[-1] == 350 or lxs[-1] == 700:
            vel_at_p = vel_at_p[90000:]
        elif lxs[-1] == 100:
            vel_at_p = vel_at_p[175000:]
        else:
            vel_at_p = vel_at_p[75000:]

        vel_at_p -= np.mean(vel_at_p)
        yf = np.fft.fft(vel_at_p)
        freq = np.fft.fftfreq(len(vel_at_p), 1)
        vortex_frequency = np.abs(freq[np.argmax(np.abs(yf))])
        strouhal.append(strouhal_number(vortex_frequency, d, u0))

    strouhal = [x for _, x in sorted(zip(lxs, strouhal))]
    reynolds = np.sort(lxs)
    plt.plot(reynolds, strouhal)
    plt.xlabel('lx [lu]')
    plt.ylabel('Strouhal number')
    plt.savefig(os.path.join(folder, "nx_strouhal.svg"), bbox_inches='tight')
    plt.savefig(os.path.join(folder, "nx_strouhal.pgf"), bbox_inches='tight')


def plot_blockage_strouhal():
    """
    Creates plot for strouhal number as function of the blockage ratio
    """
    folder = r'./figures/von_karman_vortex_shedding/blockage_strouhal'
    d = 40
    u0 = 0.1
    strouhal = []
    lxs = []
    for file in glob.glob(folder + r"/*.npy"):
        vel_at_p = np.load(file)
        plt.plot(vel_at_p)
        plt.savefig(file + '.svg')
        plt.close()

        vel_at_p = vel_at_p[90000:]

        lxs.append(float(file[file.rfind('_') + 1:file.rfind('.npy')]))

        vel_at_p -= np.mean(vel_at_p)
        yf = np.fft.fft(vel_at_p)
        freq = np.fft.fftfreq(len(vel_at_p), 1)
        vortex_frequency = np.abs(freq[np.argmax(np.abs(yf))])
        strouhal.append(strouhal_number(vortex_frequency, d, u0))

    strouhal = [x for _, x in sorted(zip(lxs, strouhal))]
    reynolds = np.sort(lxs)
    plt.plot(reynolds, strouhal)
    plt.xlabel(r'blockage ratio [\%]')
    plt.ylabel('Strouhal number')
    plt.savefig(os.path.join(folder, "blockage_strouhal.svg"), bbox_inches='tight')
    plt.savefig(os.path.join(folder, "blockage_strouhal.pgf"), bbox_inches='tight')


def plot_scaling_test(lattice_grid_shape: Tuple[int, int] = (420, 180)):
    """
    Creates plot for MLUPS as a function of MPI processes
    """
    lx, ly = lattice_grid_shape
    folder = r'./figures/von_karman_vortex_shedding/scaling_test'
    mlups = []
    n = []
    for file in glob.glob(folder + "/" + str(lx) + "_" + str(ly) + "*.npy"):
        mlups.append(np.load(file) / 10e6)
        n.append(int(file[file.rfind('_') + 1:file.rfind('.npy')]))

    mlups = [x for _, x in sorted(zip(n, mlups))]
    n = np.sort(n)
    plt.loglog(n, mlups)
    plt.xlabel('Number of MPI processes')
    plt.ylabel('Million lattice operations per second')
    plt.savefig(os.path.join(folder, "scaling_test_" + str(lx) + "_" + str(ly) + ".svg"), bbox_inches='tight')
    plt.savefig(os.path.join(folder, "scaling_test_" + str(lx) + "_" + str(ly) + ".pgf"), bbox_inches='tight')
