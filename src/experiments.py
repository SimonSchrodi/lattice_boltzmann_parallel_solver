import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
from typing import Tuple
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.signal import argrelextrema
from collections import OrderedDict
import csv
from tqdm import trange, tqdm
import os
import time

from mpi4py import MPI

from initial_values import (
    sinusoidal_density_x,
    sinusoidal_velocity_x,
    density_1_velocity_0_initial,
    density_1_velocity_x_u0_velocity_y_0_initial
)

from lattice_boltzmann_method import equilibrium_distr_func, lattice_boltzmann_step
from parallelization_utils import (
    communication,
    x_in_process,
    y_in_process,
    get_local_coords,
    global_coord_to_local_coord,
    global_to_local_direction,
    save_mpiio,
    get_xy_size
)

from boundary_utils import (
    couette_flow_boundary_conditions,
    poiseuille_flow_boundary_conditions,
    parallel_von_karman_boundary_conditions
)

matplotlib.use('pgf')
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'pgf.preamble': [r'\usepackage[utf8x]{inputenc}',
                     r'\usepackage{amsmath}']
}
)


def plot_evolution_of_density(lattice_grid_shape: Tuple[int, int] = (50, 50),
                              initial_p0: float = 0.5,
                              epsilon: float = 0.08,
                              omega: float = 1.0,
                              time_steps: int = 2500,
                              number_of_visualizations: int = 20):
    """
    Executes the experiment for shear wave decay given a sinusoidal density and saves the results.

    Args:
        lattice_grid_shape: lattice size
        initial_p0: shift of density
        epsilon: amplitude of sine wave
        omega: relaxation parameter
        time_steps: number of time steps for simulation
        number_of_visualizations: total number of visualization. Has to be divisible by 5.
    """
    assert 0 < omega < 2
    assert time_steps > 0
    assert number_of_visualizations % 5 == 0

    density, velocity = sinusoidal_density_x(lattice_grid_shape, initial_p0, epsilon)
    f = equilibrium_distr_func(density, velocity)

    fig, ax = plt.subplots(int(number_of_visualizations / 5), 5, sharex=True, sharey=True)
    ax[0, 0].plot(np.arange(0, lattice_grid_shape[0]), density[:, int(lattice_grid_shape[0] / 2)])
    ax[0, 0].set_title('initial')
    row_index, col_index = 0, 1
    for i in trange(time_steps):
        f, density, velocity = lattice_boltzmann_step(f, density, velocity, omega)
        if (i + 1) % int(time_steps / number_of_visualizations) == 0:
            ax[row_index, col_index].plot(np.arange(0, lattice_grid_shape[-1]),
                                          density[:, int(lattice_grid_shape[0] / 2)])
            ax[row_index, col_index].set_title('step ' + str(i))
            col_index += 1
            if col_index == 5:
                col_index = 0
                row_index += 1
            if row_index == 4:
                break

    fig.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.75, hspace=0.5)
    plt.savefig(r'./figures/shear_wave_decay/evolution_density_surface.pgf')
    plt.savefig(r'./figures/shear_wave_decay/evolution_density_surface.svg')


def plot_evolution_of_velocity(lattice_grid_shape: Tuple[int, int] = (50, 50),
                               epsilon: float = 0.01,
                               omega: float = 1.0,
                               time_steps: int = 2500,
                               number_of_visualizations: int = 20):
    """
    Executes the experiment for shear wave decay given a sinusoidal velocity and saves the results.

    Args:
        lattice_grid_shape: lattice size
        epsilon: amplitude of sine wave
        omega: relaxation parameter
        time_steps: number of time steps for simulation
        number_of_visualizations: total number of visualization. Has to be divisible by 5
    """
    assert 0 < omega < 2
    assert time_steps > 0
    assert number_of_visualizations % 5 == 0

    density, velocity = sinusoidal_velocity_x(lattice_grid_shape, epsilon)
    f = equilibrium_distr_func(density, velocity)

    fig, ax = plt.subplots(int(number_of_visualizations / 5), 5, sharex=True, sharey=True)
    ax[0, 0].plot(np.arange(0, lattice_grid_shape[-1]), velocity[int(lattice_grid_shape[0] / 2), :, 0])
    ax[0, 0].set_title('initial')
    row_index, col_index = 0, 1
    for i in trange(time_steps):
        f, density, velocity = lattice_boltzmann_step(f, density, velocity, omega)
        if (i + 1) % int(time_steps / number_of_visualizations) == 0:
            ax[row_index, col_index].plot(np.arange(0, lattice_grid_shape[-1]),
                                          velocity[int(lattice_grid_shape[0] / 2), :, 0])
            ax[row_index, col_index].set_title('step ' + str(i))
            col_index += 1
            if col_index == 5:
                col_index = 0
                row_index += 1
            if row_index == 4:
                break

    fig.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.75, hspace=0.5)
    plt.savefig(r'./figures/shear_wave_decay/evolution_velocity_surface.pgf')
    plt.savefig(r'./figures/shear_wave_decay/evolution_velocity_surface.svg')


def plot_measured_viscosity_vs_omega(lattice_grid_shape: Tuple[int, int] = (50, 50),
                                     initial_p0: float = 0.5,
                                     epsilon_p: float = 0.08,
                                     epsilon_v: float = 0.08,
                                     time_steps: int = 2500,
                                     omega_discretization: int = 50):
    """
    Executes the experiment to study the relationship between theoretical kinematic viscosity and relaxation parameter
    omega and saves the results.

    Args:
        lattice_grid_shape: lattice size
        initial_p0: shift of density
        epsilon_p: amplitude of density sine wave
        epsilon_v: amplitude of velocity sine wave
        time_steps: number of time steps
        omega_discretization: number of how many omegas should be discretized
    """
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    omega = np.linspace(0.01, 1.99, omega_discretization)

    initial_distr_funcs = [sinusoidal_density_x(lattice_grid_shape, initial_p0, epsilon_p),
                           sinusoidal_velocity_x(lattice_grid_shape, epsilon_v)]

    for i, initial in enumerate(tqdm(initial_distr_funcs)):
        viscosity_sim = []
        viscosity_true = []
        for om in tqdm(omega):
            density, velocity = initial
            f = equilibrium_distr_func(density, velocity)
            vels = []
            dens = []
            for _ in trange(time_steps):
                f, density, velocity = lattice_boltzmann_step(f, density, velocity, om)
                if i == 0:
                    den_min = np.amin(density)
                    den_max = np.amax(density)
                    dens.append(
                        np.abs(den_min) - initial_p0 if np.abs(den_min) > np.abs(den_max) else np.abs(
                            den_max) - initial_p0
                    )
                elif i == 1:
                    vel_min = np.amin(velocity)
                    vel_max = np.amax(velocity)
                    vels.append(
                        np.abs(vel_min) if np.abs(vel_min) > np.abs(vel_max) else np.abs(vel_max)
                    )

            x = np.arange(0, time_steps)
            if i == 0:
                dens = np.array(dens)
                indizes = argrelextrema(dens, np.greater)
                a = dens[indizes]
                viscosity_sim.append(
                    curve_fit(lambda t, v: epsilon_p * np.exp(-v * np.power(2 * np.pi / lattice_grid_shape[0], 2) * t),
                              np.array(indizes).squeeze(), a)[0][0]
                )
            elif i == 1:
                vels = np.array(vels)
                viscosity_sim.append(
                    curve_fit(lambda t, v: epsilon_v * np.exp(-v * np.power(2 * np.pi / lattice_grid_shape[-1], 2) * t),
                              x,
                              vels)[0][0])
            viscosity_true.append((1 / 3) * (1 / om - 0.5))

        viscosity_sim = np.array(viscosity_sim)
        ax[i].plot(omega, viscosity_sim, label='Simul. visc.')
        viscosity_true = np.array(viscosity_true)
        ax[i].plot(omega, viscosity_true, label='Analyt. visc.')
        ax[i].legend()
        ax[i].set_yscale('log')
        ax[i].set_title("Sinusoidal Density" if i == 0 else "Sinusoidal Velocity")
        ax[i].set_xlabel(r'relaxation parameter $\omega$')
        ax[i].set_ylabel(r'viscosity $\nu$ [$\frac{lu²}{s}$]')

    plt.savefig(r'./figures/shear_wave_decay/meas_visc_vs_omega.svg')
    plt.savefig(r'./figures/shear_wave_decay/meas_visc_vs_omega.pgf')


def plot_couette_flow_evolution(lattice_grid_shape: Tuple[int, int] = (20, 20),
                                omega: float = 1.0,
                                U: float = 0.01,
                                time_steps: int = 4000,
                                number_of_visualizations: int = 30):
    """
    Executes the couette flow evolution experiment and saves the results.

    Args:
        lattice_grid_shape:
        omega: relaxation parameter
        U: velocity of moving wall
        time_steps: number of time steps for simulation
        number_of_visualizations: total number of visualizations. Has to be divisible by 5.
    """
    assert number_of_visualizations % 5 == 0
    assert U <= 1 / np.sqrt(3)

    lx, ly = lattice_grid_shape
    fig, ax = plt.subplots(int(number_of_visualizations / 5), 5, sharex=True, sharey=True)
    row_index, col_index = 0, 0

    density, velocity = density_1_velocity_0_initial((lx, ly))
    f = equilibrium_distr_func(density, velocity)
    velocities = [velocity]
    boundary_func = couette_flow_boundary_conditions(lx, ly, U, np.mean(density))
    for _ in trange(time_steps):
        f, density, velocity = lattice_boltzmann_step(f, density, velocity, omega, boundary_func)
        velocities.append(velocity)

    velocities_for_viz = [velocity for i, velocity in enumerate(velocities) if
                          i % int(time_steps / (number_of_visualizations - 1)) == 0]  # -1 for the initial viz
    indizes_for_viz = [i for i, velocity in enumerate(velocities) if
                       i % int(time_steps / (number_of_visualizations - 1)) == 0]  # -1 for the initial viz

    max_vel = np.amax(np.array(velocities_for_viz)[:, int(lx / 2), :, 0]) + np.amax(
        np.array(velocities_for_viz)[:, int(lx / 2), :, 0]) * 0.1
    for i, velocity in zip(indizes_for_viz, velocities_for_viz):
        vx = velocity[..., 0]

        for vec, y_coord in zip(vx[int(lx / 2), :], np.arange(0, ly)):
            origin = [0, y_coord]
            ax[row_index, col_index].quiver(*origin, *[vec, 0.0], color='blue', scale_units='xy', scale=1,
                                            headwidth=3, width=0.0025)
        ax[row_index, col_index].plot(vx[int(lx / 2), :], np.arange(0, ly), label='Simul. Sol.', linewidth=1,
                                      c='blue', linestyle=':')
        ax[row_index, col_index].plot(U * (ly - np.arange(0, ly + 1)) / ly, np.arange(0, ly + 1) - 0.5,
                                      label='Analyt. Sol.', c='red',
                                      linestyle='--')
        ax[row_index, col_index].plot(np.linspace(0, max_vel, 100),
                                      np.ones_like(np.linspace(0, max_vel, 100)) * (ly - 1) + 0.5, label='Rigid Wall',
                                      linewidth=1.5, c='orange',
                                      linestyle='-.')
        ax[row_index, col_index].plot(np.linspace(0, max_vel, 100), np.zeros_like(np.linspace(0, max_vel, 100)) - 0.5,
                                      label='Moving Wall',
                                      linewidth=1.5, c='green',
                                      linestyle='-')

        if i == 0:
            ax[row_index, col_index].set_title('initial')
        else:
            ax[row_index, col_index].set_title('step ' + str(i))

        col_index += 1
        if col_index == 5:
            col_index = 0
            row_index += 1

    handles, labels = ax[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', borderaxespad=0.1)
    fig.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.75, hspace=1.5)
    plt.subplots_adjust(right=0.77)

    plt.savefig(r'./figures/couette_flow/vel_vectors_evolution.svg', bbox_inches='tight')
    plt.savefig(r'./figures/couette_flow/vel_vectors_evolution.pgf', bbox_inches='tight')


def plot_couette_flow_vel_vectors(lattice_grid_shape: Tuple[int, int] = (20, 30),
                                  omega: float = 1.0,
                                  U: float = 0.05,
                                  time_steps: int = 5000):
    """
    Executes the couette flow experiment and save results. Results contain comparision with analytical solution,
    absolute error, linear regression of simulation results

    Args:
        lattice_grid_shape: lattice size
        omega: relaxation parameter
        U: velocity of moving wall
        time_steps: number of time steps for simulation
    """
    assert U <= 1 / np.sqrt(3)
    lx, ly = lattice_grid_shape

    density, velocity = density_1_velocity_0_initial((lx, ly))
    f = equilibrium_distr_func(density, velocity)
    boundary_func = couette_flow_boundary_conditions(lx, ly, U, np.mean(density))
    for _ in trange(time_steps):
        f, density, velocity = lattice_boltzmann_step(f, density, velocity, omega, boundary_func)
    vx = velocity[..., 0]

    for vec, y_coord in zip(vx[int(lx / 2), :], np.arange(0, ly)):
        origin = [0, y_coord]
        plt.quiver(*origin, *[vec, 0.0], color='blue', scale_units='xy', scale=1, headwidth=3, width=0.0025)
    plt.plot(vx[int(lx / 2), :], np.arange(0, ly), label='Simul. sol.', linewidth=1, c='blue', linestyle=':')
    plt.plot(U * (ly - np.arange(0, ly + 1)) / ly, np.arange(0, ly + 1) - 0.5, label='Analyt. sol.', c='red',
             linestyle='--')
    plt.plot(np.linspace(0, U, 100), np.ones_like(np.linspace(0, U, 100)) * (ly - 1) + 0.5,
             label='Rigid wall', linewidth=1.5, c='orange',
             linestyle='-.')
    plt.plot(np.linspace(0, U, 100), np.zeros_like(np.linspace(0, U, 100)) - 0.5, label='Moving wall',
             linewidth=1.5, c='green',
             linestyle='-')
    plt.ylabel('y position [lu]')
    plt.xlabel(r'velocity in x-direction $\mathbf{u}_x$ [$\frac{lu}{s}$]')
    plt.legend()

    plt.savefig(r'./figures/couette_flow/vel_vectors.svg', bbox_inches='tight')
    plt.savefig(r'./figures/couette_flow/vel_vectors.pgf', bbox_inches='tight')

    plt.close()

    simulated = vx[int(lx / 2)]
    slope, intercept, rvalue, pvalue, stderr = linregress(np.arange(ly), simulated)
    with open('./figures/couette_flow/linregress.csv', 'w', newline='') as csvfile:
        fieldnames = ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                'slope': slope,
                'intercept': intercept,
                'rvalue': rvalue,
                'pvalue': pvalue,
                'stderr': stderr
            }
        )
    analytical_points = [intercept + slope * (x + 0.5) for x in np.arange(0, ly)]
    simulated_points = vx[int(lx / 2), :]
    abs_error = np.abs(simulated_points - analytical_points)
    abs_error = np.where(abs_error < 10e-4, 0, abs_error)
    plt.plot(np.arange(0, ly), abs_error)

    plt.xlabel(r'y position [lu]')
    plt.ylabel(r'absolute error [\%]')

    plt.savefig(r'./figures/couette_flow/absolute_error.svg', bbox_inches='tight')
    plt.savefig(r'./figures/couette_flow/absolute_error.pgf', bbox_inches='tight')


def plot_poiseuille_flow_vel_vectors(lattice_grid_shape: Tuple[int, int] = (200, 60),
                                     omega: float = 1.5,
                                     delta_p: float = 0.001,
                                     time_steps: int = 40000):
    """
    Executes the poiseuille flow experiment. Results contain qualitative comparison to analytical solution,
    difference of area under curve at the inlet and middle channel, pressure along the centerline and the
    absolute error.

    Args:
        lattice_grid_shape: lattice size
        omega: relaxation parameter
        delta_p: pressure difference
        time_steps: number of time steps of the simulation
    """
    lx, ly = lattice_grid_shape

    rho_0 = 1
    delta_rho = delta_p * 3
    rho_inlet = rho_0 + (delta_rho / 2)
    rho_outlet = rho_0 - (delta_rho / 2)
    p_in = rho_inlet / 3
    p_out = rho_outlet / 3

    boundary_func = poiseuille_flow_boundary_conditions(lx, ly, p_in, p_out)

    density, velocity = density_1_velocity_0_initial((lx, ly))
    f = equilibrium_distr_func(density, velocity)
    for _ in trange(time_steps):
        f, density, velocity = lattice_boltzmann_step(f, density, velocity, omega, boundary_func)

    vx = velocity[..., 0]
    x_coords = [1, lx // 2]
    centerline = ly // 2

    areas = []
    colors = ['cyan', 'blue']
    linestyle = [':', '-.']
    for c, ls, x_coord in zip(colors, linestyle, x_coords):
        for vec, y_coord in zip(vx[x_coord, :], np.arange(0, ly)):
            origin = [0, y_coord]
            plt.quiver(*origin, *[vec, 0.0], color=c, scale_units='xy', scale=1, headwidth=3, width=0.0025)
        plt.plot(vx[x_coord, :], np.arange(0, ly), label='Sim. sol. channel ' + str(x_coord), linewidth=1, c=c,
                 linestyle=ls)
        areas.append(
            np.trapz(vx[x_coord, :], np.arange(0, ly))
        )
        viscosity = (1 / 3) * (1 / omega - 0.5)
        dynamic_viscosity = viscosity * np.mean(density[x_coord, :])
        h = ly
        y = np.arange(0, ly + 1)
        dp_dx = np.divide(p_out - p_in, lx)
        uy = -np.reciprocal(2 * dynamic_viscosity) * dp_dx * y * (h - y)
        plt.plot(uy, y - 0.5, label='Analyt. sol.', c='red',
                 linestyle='--')

        plt.plot(np.linspace(0, np.amax(vx) * 1.05, 100), np.zeros_like(np.linspace(0, np.amax(vx) * 1.05, 100)) - 0.5,
                 label='Rigid wall',
                 linewidth=1.5, c='green',
                 linestyle='-')
        plt.plot(np.linspace(0, np.amax(vx) * 1.05, 100),
                 np.ones_like(np.linspace(0, np.amax(vx) * 1.05, 100)) * (ly - 1) + 0.5,
                 label='Rigid wall',
                 linewidth=1.5, c='green',
                 linestyle='-')

        plt.ylabel('y position [lu]')
        plt.xlabel(r'velocity in x-direction $\mathbf{u}_x$ [$\frac{lu}{s}$]')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.legend(by_label.values(), by_label.keys(), loc='lower right')

        plt.savefig(r'./figures/poiseuille_flow/vel_vectors.svg', bbox_inches='tight')
        plt.savefig(r'./figures/poiseuille_flow/vel_vectors.pgf', bbox_inches='tight')

    plt.close()

    areas.append(areas[0] / areas[1])
    areas = np.array(areas)
    with open('./figures/poiseuille_flow/areas.csv', 'w', newline='') as csvfile:
        fieldnames = ['inlet', 'middle', 'relative_difference']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                'inlet': areas[0],
                'middle': areas[1],
                'relative_difference': areas[2]
            }
        )

    plt.plot(np.arange(0, lx - 2), density[1:-1, centerline] / 3, label='Pressure along centerline')
    plt.plot(np.arange(0, lx - 2), np.ones_like(np.arange(0, lx - 2)) * p_out, label='Outgoing pressure')
    plt.plot(np.arange(0, lx - 2), np.ones_like(np.arange(0, lx - 2)) * p_in,
             label='Ingoing pressure')
    plt.xlabel('x position [lu]')
    plt.ylabel(r'pressure along centerline $p$ [$Pa$]')
    plt.legend()

    plt.savefig(r'./figures/poiseuille_flow/density_along_centerline.svg', bbox_inches='tight')
    plt.savefig(r'./figures/poiseuille_flow/density_along_centerline.pgf', bbox_inches='tight')

    plt.close()

    popt, pcov = curve_fit(lambda y, a, b, c: a * (y ** 2) + b * y + c, np.arange(0, ly), vx[x_coord, :])
    with open('./figures/poiseuille_flow/curve_fit.csv', 'w', newline='') as csvfile:
        fieldnames = ['popt', 'pcov']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                'popt': popt,
                'pcov': pcov
            }
        )
    a, b, c = popt
    analytical_points = [a * ((x + 0.5) ** 2) + b * (x + 0.5) + c for x in np.arange(0, ly)]
    simulated_points = vx[int(lx // 2), :]
    abs_error = np.abs(simulated_points - analytical_points)
    abs_error = np.where(abs_error < 10e-4, 0, abs_error)
    plt.plot(np.arange(0, ly), abs_error)

    plt.xlabel(r'y position [lu]')
    plt.ylabel(r'absolute error [\%]')

    plt.savefig(r'./figures/poiseuille_flow/absolute_error.svg', bbox_inches='tight')
    plt.savefig(r'./figures/poiseuille_flow/absolute_error.pgf', bbox_inches='tight')


def plot_poiseuille_flow_evolution(lattice_grid_shape: Tuple[int, int] = (200, 30),
                                   omega: float = 1.5,
                                   delta_p: float = 0.001,
                                   time_steps: int = 10000,
                                   number_of_visualizations: int = 30):
    """
    Executes the poiseuille flow evolution experiment and saves results.

    Args:
        lattice_grid_shape: lattice size
        omega: relaxation parameter
        delta_p: pressure difference at inlet and outlet
        time_steps: number of time steps for the simulation
        number_of_visualizations: total number of visualization. Has to be divisible by 5.
    """
    assert number_of_visualizations % 5 == 0

    lx, ly = lattice_grid_shape

    fig, ax = plt.subplots(int(number_of_visualizations / 5), 5, sharex=True, sharey=True)
    row_index, col_index = 0, 0

    rho_0 = 1
    delta_rho = delta_p * 3
    rho_inlet = rho_0 + delta_rho
    rho_outlet = rho_0
    p_in = rho_inlet / 3
    p_out = rho_outlet / 3

    boundary_func = poiseuille_flow_boundary_conditions(lx, ly, p_in, p_out)
    density, velocity = density_1_velocity_0_initial((lx, ly))
    f = equilibrium_distr_func(density, velocity)
    velocities = [velocity]
    for _ in trange(time_steps):
        f, density, velocity = lattice_boltzmann_step(f, density, velocity, omega, boundary_func)
        velocities.append(velocity)

    x_coord = lx // 2
    viscosity = (1 / 3) * (1 / omega - 0.5)
    dynamic_viscosity = viscosity * np.mean(density[x_coord, :])
    h = ly
    y = np.arange(0, ly + 1)
    dp_dx = np.divide(p_out - p_in, lx)
    uy = -np.reciprocal(2 * dynamic_viscosity) * dp_dx * y * (h - y)

    for i, velocity in enumerate(velocities):
        if i % int(time_steps / (number_of_visualizations - 1)) == 0:
            vx = velocity[..., 0]

            ax[row_index, col_index].plot(uy, y - 0.5, label='Analyt. sol.', c='red', linewidth=0.5)
            for vec, y_coord in zip(vx[x_coord, :], np.arange(0, ly)):
                origin = [0, y_coord]
                ax[row_index, col_index].quiver(*origin, *[vec, 0.0], color='blue', scale_units='xy', scale=1,
                                                headwidth=3, width=0.0025)
            ax[row_index, col_index].plot(vx[x_coord, :], np.arange(0, ly), label='Sim. sol.', linewidth=1, c='blue',
                                          linestyle=':')

            if i == 0:
                ax[row_index, col_index].set_title('initial')
            else:
                ax[row_index, col_index].set_title('step ' + str(i))

            col_index += 1
            if col_index == 5:
                col_index = 0
                row_index += 1

    handles, labels = ax[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', borderaxespad=0.1)
    fig.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.75, hspace=1.5)
    plt.subplots_adjust(right=0.77)

    plt.savefig(r'./figures/poiseuille_flow/vel_vectors_evolution.svg', bbox_inches='tight')
    plt.savefig(r'./figures/poiseuille_flow/vel_vectors_evolution.pgf', bbox_inches='tight')


def plot_parallel_von_karman_vortex_street(lattice_grid_shape: Tuple[int, int] = (420, 180),
                                           plate_size: int = 40,
                                           inlet_density: float = 1.0,
                                           inlet_velocity: float = 0.1,
                                           kinematic_viscosity: float = 0.04,
                                           time_steps: int = 100000):
    """
    Executes the parallel version of the code of the von Karman vortex street and saves each 100 time steps the
    current velocity magnitude field.

    Args:
        lattice_grid_shape: lattice size
        plate_size: size of the plate
        inlet_density: density into the domain
        inlet_velocity: velocity into the domain
        kinematic_viscosity: kinematic viscosity
        time_steps: number of time steps for simulation
    """
    # setup
    lx, ly = lattice_grid_shape
    omega = np.reciprocal(3 * kinematic_viscosity + 0.5)

    p_coords = [3 * lx // 4, ly // 2]

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    comm = MPI.COMM_WORLD
    x_size, y_size = get_xy_size(size)

    cartesian2d = comm.Create_cart(dims=[x_size, y_size], periods=[True, True], reorder=False)
    coords2d = cartesian2d.Get_coords(rank)

    n_local_x, n_local_y = get_local_coords(coords2d, lx, ly, x_size, y_size)

    density, velocity = density_1_velocity_x_u0_velocity_y_0_initial((n_local_x + 2, n_local_y + 2), inlet_velocity)
    f = equilibrium_distr_func(density, velocity)
    process_coord, px, py = global_coord_to_local_coord(coords2d, p_coords[0], p_coords[1], lx, ly, x_size, y_size)
    if process_coord is not None:
        vel_at_p = [np.linalg.norm(velocity[px, py, ...])]

    bound_func = parallel_von_karman_boundary_conditions(coords2d, n_local_x, n_local_y, lx, ly, x_size, y_size,
                                                         inlet_density, inlet_velocity, plate_size)
    communication_func = communication(cartesian2d)

    # main loop
    if rank == 0:
        pbar = tqdm(total=time_steps)
    for i in range(time_steps):
        if rank == 0:
            pbar.update(1)
        f, density, velocity = lattice_boltzmann_step(f, density, velocity, omega, bound_func, communication_func)
        if process_coord is not None:
            vel_at_p.append(np.linalg.norm(velocity[px, py, ...]))

        if i % 100 == 0:
            abs_vel = np.linalg.norm(velocity[1:-1, 1:-1, :], axis=-1)
            save_mpiio(cartesian2d, r'./figures/von_karman_vortex_shedding/all_png_parallel/vel_norm.npy', abs_vel)

            if rank == 0:
                abs_vel = np.load(r'./figures/von_karman_vortex_shedding/all_png_parallel/vel_norm.npy')
                normalized_vel = abs_vel / np.amax(abs_vel)
                img = Image.fromarray(np.uint8(cm.viridis(normalized_vel.T) * 255))
                img.save(r'./figures/von_karman_vortex_shedding/all_png_parallel/' + str(i) + '.png')
                os.remove(r'./figures/von_karman_vortex_shedding/all_png_parallel/vel_norm.npy')


def x_strouhal(folder_name: str,
               lattice_grid_shape: Tuple[int, int] = (420, 180),
               plate_size: int = 40,
               inlet_density: float = 1.0,
               inlet_velocity: float = 0.1,
               kinematic_viscosity: float = 0.04,
               time_steps: int = 200000):
    """
    General functions to execute experiments to study the relationship of the strouhal numbers to a given x
    (e.g. reynolds number, nx, blockage ratio)

    Args:
        folder_name: folder to save the files to
        lattice_grid_shape: lattice size
        plate_size: size of the plate
        inlet_density: density into the domain
        inlet_velocity: velocity into the domain
        kinematic_viscosity: kinematic viscosity
        time_steps: number of time steps for the simulation
    """
    # setup
    lx, ly = lattice_grid_shape
    omega = np.reciprocal(3 * kinematic_viscosity + 0.5)

    p_coords = [3 * lx // 4, ly // 2]

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    comm = MPI.COMM_WORLD
    x_size, y_size = get_xy_size(size)

    cartesian2d = comm.Create_cart(dims=[x_size, y_size], periods=[True, True], reorder=False)
    coords2d = cartesian2d.Get_coords(rank)

    n_local_x, n_local_y = get_local_coords(coords2d, lx, ly, x_size, y_size)

    density, velocity = density_1_velocity_x_u0_velocity_y_0_initial((n_local_x + 2, n_local_y + 2), inlet_velocity)
    f = equilibrium_distr_func(density, velocity)
    process_coord, px, py = global_coord_to_local_coord(coords2d, p_coords[0], p_coords[1], lx, ly, x_size, y_size)
    if process_coord is not None:
        vel_at_p = [np.linalg.norm(velocity[px, py, ...])]

    bound_func = parallel_von_karman_boundary_conditions(coords2d, n_local_x, n_local_y, lx, ly, x_size, y_size,
                                                         inlet_density, inlet_velocity, plate_size)
    communication_func = communication(cartesian2d)

    # main loop
    if rank == 0:
        pbar = tqdm(total=time_steps)
    for i in range(time_steps):
        if rank == 0:
            pbar.update(1)
        f, density, velocity = lattice_boltzmann_step(f, density, velocity, omega, bound_func, communication_func)
        if process_coord is not None:
            vel_at_p.append(np.linalg.norm(velocity[px, py, ...]))

    if process_coord is not None:
        if 'reynold' in folder_name:
            reynolds_number = plate_size * inlet_velocity / kinematic_viscosity
            np.save(r'./figures/von_karman_vortex_shedding/' + folder_name + '/vel_at_p_' + str(
                round(reynolds_number)) + '.npy', vel_at_p)
        elif 'nx' in folder_name:
            np.save(r'./figures/von_karman_vortex_shedding/' + folder_name + '/vel_at_p_' + str(int(lx)) + '.npy',
                    vel_at_p)
        elif 'blockage' in folder_name:
            blockage_ratio = plate_size / ly
            np.save(r'./figures/von_karman_vortex_shedding/' + folder_name + '/vel_at_p_' + str(
                blockage_ratio) + '.npy',
                    vel_at_p)
        else:
            raise Exception('Unknown experiment')


def scaling_test(folder_name: str,
                 lattice_grid_shape: Tuple[int, int] = (420, 180),
                 plate_size: int = 40,
                 inlet_density: float = 1.0,
                 inlet_velocity: float = 0.1,
                 kinematic_viscosity: float = 0.04,
                 time_steps: int = 20000):
    """
    Executes the scaling test. Measures the time for simulation and saves results.

    Args:
        folder_name: folder where to save results
        lattice_grid_shape: lattice size
        plate_size: size of plate
        inlet_density: density into the domain
        inlet_velocity: velocity into the domain
        kinematic_viscosity: kinematic viscosity
        time_steps: number of time steps for simulation
    """
    # setup
    lx, ly = lattice_grid_shape
    omega = np.reciprocal(3 * kinematic_viscosity + 0.5)

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    comm = MPI.COMM_WORLD
    x_size, y_size = get_xy_size(size)

    cartesian2d = comm.Create_cart(dims=[x_size, y_size], periods=[True, True], reorder=False)
    coords2d = cartesian2d.Get_coords(rank)

    n_local_x, n_local_y = get_local_coords(coords2d, lx, ly, x_size, y_size)

    density, velocity = density_1_velocity_x_u0_velocity_y_0_initial((n_local_x + 2, n_local_y + 2), inlet_velocity)
    f = equilibrium_distr_func(density, velocity)

    bound_func = parallel_von_karman_boundary_conditions(coords2d, n_local_x, n_local_y, lx, ly, x_size, y_size,
                                                         inlet_density, inlet_velocity, plate_size)
    communication_func = communication(cartesian2d)

    # main loop
    if rank == 0:
        start = time.time_ns()
    for i in range(time_steps):
        f, density, velocity = lattice_boltzmann_step(f, density, velocity, omega, bound_func, communication_func)
    if rank == 0:
        end = time.time_ns()
        runtime_ns = end - start
        runtime = runtime_ns / 10e9
        mlups = lx * ly * time_steps / runtime
        np.save(r'./figures/von_karman_vortex_shedding/' + folder_name + '/' + str(
            int(lx)) + '_' + str(int(ly)) + '_' + str(int(size)) + '.npy', np.array([mlups]))
