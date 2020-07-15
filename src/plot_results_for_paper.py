import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.signal import argrelextrema
from collections import OrderedDict
import csv

from src.initial_values import sinusoidal_density_x, sinusoidal_velocity_x, density_1_velocity_0_initial
from src.lattice_boltzman_equation import equilibrium_distr_func, lattice_boltzman_step, lattice_boltzman_solver
from src.boundary_conditions import rigid_wall, moving_wall, periodic_with_pressure_variations


def plot_evolution_of_density(lattice_grid_shape: Tuple[int, int] = (50, 50),
                              initial_p0: float = 0.5,
                              epsilon: float = 0.08,
                              omega: float = 1.0,
                              time_steps: int = 2000,
                              number_of_visualizations: int = 20):
    assert 0 < omega < 2
    assert time_steps > 0
    assert number_of_visualizations % 5 == 0

    density, velocity = sinusoidal_density_x(lattice_grid_shape, initial_p0, epsilon)
    f = equilibrium_distr_func(density, velocity)

    fig, ax = plt.subplots(int(number_of_visualizations / 5), 5, sharex=True, sharey=True)
    ax[0, 0].plot(np.arange(0, lattice_grid_shape[0]), density[:, int(lattice_grid_shape[0] / 2)])
    ax[0, 0].set_title('initial')
    row_index, col_index = 0, 1
    for i in range(time_steps):
        f, density, velocity = lattice_boltzman_step(f, density, velocity, omega)
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

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.75, hspace=0.5)
    plt.savefig(r'../figures/shear_wave_decay/evolution_density_surface.svg')


def plot_evolution_of_velocity(lattice_grid_shape: Tuple[int, int] = (50, 50),
                               epsilon: float = 0.01,
                               omega: float = 1.0,
                               time_steps: int = 2000,
                               number_of_visualizations: int = 20):
    assert 0 < omega < 2
    assert time_steps > 0
    assert number_of_visualizations % 5 == 0

    density, velocity = sinusoidal_velocity_x(lattice_grid_shape, epsilon)
    f = equilibrium_distr_func(density, velocity)

    fig, ax = plt.subplots(int(number_of_visualizations / 5), 5, sharex=True, sharey=True)
    ax[0, 0].plot(np.arange(0, lattice_grid_shape[-1]), velocity[int(lattice_grid_shape[0] / 2), :, 0])
    ax[0, 0].set_title('initial')
    row_index, col_index = 0, 1
    for i in range(time_steps):
        f, density, velocity = lattice_boltzman_step(f, density, velocity, omega)
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

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.75, hspace=0.5)
    plt.savefig(r'../figures/shear_wave_decay/evolution_velocity_surface.svg')


def plot_measured_viscosity_vs_omega(lattice_grid_shape: Tuple[int, int] = (50, 50),
                                     initial_p0: float = 0.5,
                                     epsilon_p: float = 0.08,
                                     epsilon_v: float = 0.08,
                                     time_steps: int = 2000,
                                     omega_discretization: int = 50):
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    omega = np.linspace(0.01, 1.99, omega_discretization)

    initial_distr_funcs = [sinusoidal_density_x(lattice_grid_shape, initial_p0, epsilon_p),
                           sinusoidal_velocity_x(lattice_grid_shape, epsilon_v)]

    for i, initial in enumerate(initial_distr_funcs):
        viscosity_sim = []
        viscosity_true = []
        for om in omega:
            density, velocity = initial
            f = equilibrium_distr_func(density, velocity)
            vels = []
            dens = []
            for _ in range(time_steps):
                f, density, velocity = lattice_boltzman_step(f, density, velocity, om)
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
        ax[i].set_xlabel('omega')
        ax[i].set_ylabel(r'viscosity $\nu$ [$\frac{m²}{s}$]')

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.savefig(r'../figures/shear_wave_decay/meas_visc_vs_omega.svg')


def plot_couette_flow_evolution(lattice_grid_shape: Tuple[int, int] = (20, 20),
                                omega: float = 1.0,
                                U: float = 0.01,
                                time_steps: int = 4000,
                                number_of_visualizations: int = 30):
    assert number_of_visualizations % 5 == 0
    assert U <= 1 / np.sqrt(3)

    lx, ly = lattice_grid_shape
    fig, ax = plt.subplots(int(number_of_visualizations / 5), 5, sharex=True, sharey=True)
    row_index, col_index = 0, 0

    def boundary(f_pre_streaming, f_post_streaming, density, velocity, f=None):
        boundary_rigid_wall = np.zeros((lx, ly))
        boundary_rigid_wall[:, -1] = np.ones(ly)
        f_post_streaming = rigid_wall(boundary_rigid_wall.astype(np.bool))(f_pre_streaming, f_post_streaming)
        boundary_moving_wall = np.zeros((lx, ly))
        boundary_moving_wall[:, 0] = np.ones(ly)
        u_w = np.array(
            [U, 0]
        )
        f_post_streaming = moving_wall(boundary_moving_wall.astype(np.bool), u_w, density)(f_pre_streaming,
                                                                                           f_post_streaming)
        return f_post_streaming

    density, velocity = density_1_velocity_0_initial((lx, ly))
    f = equilibrium_distr_func(density, velocity)
    velocities = [velocity]
    for i in range(time_steps):
        f, density, velocity = lattice_boltzman_step(f, density, velocity, omega, boundary)
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

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.savefig(r'../figures/couette_flow/vel_vectors_evolution.svg', bbox_inches='tight')


def plot_couette_flow_vel_vectors(lattice_grid_shape: Tuple[int, int] = (20, 30),
                                  omega: float = 1.0,
                                  U: float = 0.05,
                                  time_steps: int = 4000):
    assert U <= 1 / np.sqrt(3)
    lx, ly = lattice_grid_shape

    def boundary(f_pre_streaming, f_post_streaming, density, velocity=None, f=None):
        boundary_rigid_wall = np.zeros(lattice_grid_shape)
        boundary_rigid_wall[:, -1] = np.ones(lx)
        f_post_streaming = rigid_wall(boundary_rigid_wall.astype(np.bool))(f_pre_streaming, f_post_streaming)
        boundary_moving_wall = np.zeros(lattice_grid_shape)
        boundary_moving_wall[:, 0] = np.ones(lx)
        u_w = np.array(
            [U, 0]
        )
        f_post_streaming = moving_wall(boundary_moving_wall.astype(np.bool), u_w, density)(f_pre_streaming,
                                                                                           f_post_streaming)
        return f_post_streaming

    density, velocity = density_1_velocity_0_initial((lx, ly))
    f = equilibrium_distr_func(density, velocity)
    for i in range(time_steps):
        f, density, velocity = lattice_boltzman_step(f, density, velocity, omega, boundary)
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
    plt.xlabel(r'velocity in y-direction $u$ [$\frac{lu}{s}$]')
    plt.legend()

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.savefig(r'../figures/couette_flow/vel_vectors.svg', bbox_inches='tight')

    plt.close()

    simulated = U * (ly - np.arange(0, ly + 1)) / ly
    slope, intercept, rvalue, pvalue, stderr = linregress(np.arange(ly + 1), simulated)
    with open('../figures/couette_flow/linregress.csv', 'w', newline='') as csvfile:
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
    relative_error = np.abs(analytical_points-simulated_points)/analytical_points
    plt.plot(np.arange(0, ly), relative_error*100)

    plt.xlabel(r'y position [lu]')
    plt.ylabel(r'relative error [\%]')

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.savefig(r'../figures/couette_flow/relative_error.svg', bbox_inches='tight')


def plot_poiseuille_flow_vel_vectors(lattice_grid_shape: Tuple[int, int] = (200, 60),
                                     omega: float = 1.5,
                                     delta_p: float = 0.001,
                                     time_steps: int = 20000):
    lx, ly = lattice_grid_shape

    rho_0 = 1
    delta_rho = delta_p * 3
    rho_inlet = rho_0 + (delta_rho/2)
    rho_outlet = rho_0 - (delta_rho/2)
    p_in = rho_inlet / 3
    p_out = rho_outlet / 3

    def boundary(f_pre_streaming, f_post_streaming, density, velocity, f=None):
        boundary = np.zeros((lx, ly))
        boundary[0, :] = np.ones(ly)
        boundary[-1, :] = np.ones(ly)
        f_post_streaming = periodic_with_pressure_variations(boundary.astype(np.bool), p_in, p_out)(
            f_pre_streaming,
            f_post_streaming,
            density, velocity)

        boundary_rigid_wall = np.zeros((lx, ly))
        boundary_rigid_wall[:, 0] = np.ones(lx)
        f_post_streaming = rigid_wall(boundary_rigid_wall.astype(np.bool))(f_pre_streaming, f_post_streaming)
        boundary_rigid_wall = np.zeros((lx, ly))
        boundary_rigid_wall[:, -1] = np.ones(lx)
        f_post_streaming = rigid_wall(boundary_rigid_wall.astype(np.bool))(f_pre_streaming, f_post_streaming)

        return f_post_streaming

    density, velocity = density_1_velocity_0_initial((lx, ly))
    f = equilibrium_distr_func(density, velocity)
    for i in range(time_steps):
        f, density, velocity = lattice_boltzman_step(f, density, velocity, omega, boundary)

    vx = velocity[..., 0]
    x_coords = [1, lx // 2]
    centerline = ly // 2

    areas = []
    colors = ['cyan', 'blue']
    linestyle = [':', '-.']
    for c, ls, x_coord in zip(colors, linestyle, x_coords):
        # for vec, y_coord in zip(vx[x_coord, :], np.arange(0, ly)):
        #     origin = [0, y_coord]
        #     plt.quiver(*origin, *[vec, 0.0], color='blue', scale_units='xy', scale=1, headwidth=3, width=0.0025)
        plt.plot(vx[x_coord, :], np.arange(0, ly), label='Sim. sol. channel '+str(x_coord), linewidth=1, c=c, linestyle=ls)
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
        plt.xlabel(r'velocity in y-direction $u$ [$\frac{lu}{s}$]')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.legend(by_label.values(), by_label.keys(), loc='lower right')

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        plt.savefig(r'../figures/poiseuille_flow/vel_vectors.svg', bbox_inches='tight')

    plt.close()

    areas.append(areas[0]/areas[1])
    areas = np.array(areas)
    with open('../figures/poiseuille_flow/areas.csv', 'w', newline='') as csvfile:
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

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.savefig(r'../figures/poiseuille_flow/density_along_centerline.svg', bbox_inches='tight')

    plt.close()

    popt, pcov = curve_fit(lambda y, a, b, c: a*(y**2)+b*y+c, np.array(y).squeeze(), uy)
    with open('../figures/poiseuille_flow/curve_fit.csv', 'w', newline='') as csvfile:
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
    analytical_points = [a*((x + 0.5)**2)+b*(x+0.5)+c for x in np.arange(0, ly)]
    simulated_points = vx[int(lx / 2), :]
    relative_error = np.abs(analytical_points - simulated_points) / analytical_points
    plt.plot(np.arange(0, ly), relative_error * 100)

    plt.xlabel(r'y position [lu]')
    plt.ylabel(r'relative error [\%]')

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.savefig(r'../figures/poiseuille_flow/relative_error.svg', bbox_inches='tight')


def plot_poiseuille_flow_evolution(lattice_grid_shape: Tuple[int, int] = (200, 30),
                                   omega: float = 1.5,
                                   delta_p: float = 0.001111,
                                   time_steps: int = 5000,
                                   number_of_visualizations: int = 30):
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

    def boundary(f_pre_streaming, f_post_streaming, density, velocity):
        boundary = np.zeros((lx, ly))
        boundary[0, :] = np.ones(ly)
        boundary[-1, :] = np.ones(ly)
        f_post_streaming = periodic_with_pressure_variations(boundary.astype(np.bool), p_in, p_out)(
            f_pre_streaming,
            f_post_streaming,
            density, velocity)

        boundary_rigid_wall = np.zeros((lx, ly))
        boundary_rigid_wall[:, 0] = np.ones(lx)
        f_post_streaming = rigid_wall(boundary_rigid_wall.astype(np.bool))(f_pre_streaming, f_post_streaming)
        boundary_rigid_wall = np.zeros((lx, ly))
        boundary_rigid_wall[:, -1] = np.ones(lx)
        f_post_streaming = rigid_wall(boundary_rigid_wall.astype(np.bool))(f_pre_streaming, f_post_streaming)

        return f_post_streaming

    density, velocity = density_1_velocity_0_initial((lx, ly))
    f = equilibrium_distr_func(density, velocity)
    velocities = [velocity]
    for i in range(time_steps):
        f, density, velocity = lattice_boltzman_step(f, density, velocity, omega, boundary)
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

            for vec, y_coord in zip(vx[x_coord, :], np.arange(0, ly)):
                origin = [0, y_coord]
                ax[row_index, col_index].quiver(*origin, *[vec, 0.0], color='blue', scale_units='xy', scale=1,
                                                headwidth=3, width=0.0025)
            ax[row_index, col_index].plot(vx[x_coord, :], np.arange(0, ly), label='Sim. sol.', linewidth=1, c='blue',
                                          linestyle=':')
            ax[row_index, col_index].plot(uy, y - 0.5, label='Analyt. sol.', c='red', linestyle='--')

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

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.savefig(r'../figures/poiseuille_flow/vel_vectors_evolution.svg', bbox_inches='tight')
