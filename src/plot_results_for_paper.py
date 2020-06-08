import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np
from scipy.optimize import curve_fit

from src.initial_values import sinusoidal_density_x, sinusoidal_velocity_x, density_1_velocity_0_initial
from src.lattice_boltzman_equation import equilibrium_distr_func, lattice_boltzman_step, lattice_boltzman_solver
from src.boundary_conditions import rigid_wall, moving_wall


def plot_evolution_of_density():
    raise NotImplementedError


def plot_evolution_of_velocity(lattice_grid_shape: Tuple[int, int] = (50, 50),
                               epsilon: float = 0.08,
                               omega: float = 1.5,
                               time_steps: int = 1000):
    assert 0 < omega < 2
    assert time_steps > 0

    density, velocity = sinusoidal_velocity_x(lattice_grid_shape, epsilon)

    f = equilibrium_distr_func(density, velocity)

    vels = []
    for i in range(time_steps):
        f, density, velocity = lattice_boltzman_step(f, density, velocity, omega)
        vel_min = np.amin(velocity)
        vel_max = np.amax(velocity)
        vels.append(
            np.abs(vel_min) if np.abs(vel_min) > np.abs(vel_max) else np.abs(vel_max)
        )

    x = np.arange(0, time_steps)
    vels = np.array(vels)
    viscosity_sim = curve_fit(
        lambda t, v: epsilon * np.exp(-v * np.power(2 * np.pi / lattice_grid_shape[0], 2) * t), x, vels
    )[0][0]
    plt.plot(np.arange(0, time_steps), np.array(vels), label='Simulated (v=' + str(round(viscosity_sim, 3)) + ")")
    viscosity = (1 / 3) * (1 / omega - 0.5)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.plot(x, epsilon * np.exp(-viscosity * np.power(2 * np.pi / lattice_grid_shape[0], 2) * x),
             label='Analytical (v=' + str(round(viscosity, 3)) + ")")
    plt.legend()
    plt.xlabel('Time t')
    plt.ylabel('Amplitude a(t)')
    plt.savefig(r'../figures/shear_wave_decay/evolution_velocity.svg')


def plot_measured_viscosity_vs_omega(lattice_grid_shape: Tuple[int, int] = (50, 50),
                                     initial_p0: float = 0.5,
                                     epsilon: float = 0.08,
                                     time_steps: int = 5000,
                                     omega_discretization: int = 10):
    fig, ax = plt.subplots(1, 2, sharex=True)
    omega = np.linspace(0.01, 1.99, omega_discretization)

    initial_distr_funcs = [sinusoidal_density_x(lattice_grid_shape, initial_p0, epsilon),
                           sinusoidal_velocity_x(lattice_grid_shape, epsilon)]

    for i, initial in enumerate(initial_distr_funcs):
        viscosity_sim = []
        viscosity_true = []
        for om in omega:
            density, velocity = initial
            f = equilibrium_distr_func(density, velocity)
            vels = []
            for _ in range(time_steps):
                f, density, velocity = lattice_boltzman_step(f, density, velocity, om)
                vel_min = np.amin(velocity)
                vel_max = np.amax(velocity)
                vels.append(
                    np.abs(vel_min) if np.abs(vel_min) > np.abs(vel_max) else np.abs(vel_max)
                )
            x = np.arange(0, time_steps)
            vels = np.array(vels)
            viscosity_sim.append(
                curve_fit(lambda t, v: epsilon * np.exp(-v * np.power(2 * np.pi / lattice_grid_shape[0], 2) * t), x,
                          vels)[0][
                    0])
            viscosity_true.append((1 / 3) * (1 / om - 0.5))

        viscosity_sim = np.array(viscosity_sim)
        ax[i].plot(omega, viscosity_sim, label='Simulated')
        viscosity_true = np.array(viscosity_true)
        ax[i].plot(omega, viscosity_true, label='Analytical')
        ax[i].legend()
        ax[i].set_xlabel('Time t')
        ax[i].set_ylabel('Amplitude a(t)')
    plt.show()

    plt.savefig(r'../figures/shear_wave_decay/meas_visc_vs_omega.svg')


def plot_couette_flow_evolution(lattice_grid_shape: Tuple[int, int] = (50, 50),
                                omega: float = 0.5,
                                U: float = 0.5,
                                time_steps: int = 5000,
                                number_of_visualizations: int = 30):
    assert number_of_visualizations % 5 == 0
    assert U <= 1 / np.sqrt(3)

    lx, ly = lattice_grid_shape
    fig, ax = plt.subplots(int(number_of_visualizations / 5), 5)
    row_index, col_index = 0, 0

    def boundary(f_pre_streaming, f_post_streaming, density):
        boundary_rigid_wall = np.zeros((lx, ly))
        boundary_rigid_wall[:, -1] = np.ones(ly)
        f_post_streaming = rigid_wall(boundary_rigid_wall.astype(np.bool))(f_pre_streaming, f_post_streaming)
        boundary_moving_wall = np.zeros((lx, ly))
        boundary_moving_wall[:, 0] = np.ones(ly)
        u_w = np.array(
            [U, 0]
        )
        f_post_streaming = moving_wall(boundary_moving_wall.astype(np.bool), u_w)(f_pre_streaming, f_post_streaming,
                                                                                  density)
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
        ax[row_index, col_index].plot(U * (ly - 1 - np.arange(0, ly)) / ly, np.arange(0, ly),
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
        ax[row_index, col_index].set_axis_off()
        ax[row_index, col_index].set_axis_off()
        ax[row_index, col_index].set_title('Step ' + str(i))

        col_index += 1
        if col_index == 5:
            col_index = 0
            row_index += 1

    handles, labels = ax[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', borderaxespad=0.1)
    fig.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.75, hspace=0.5)
    plt.subplots_adjust(right=0.77)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.savefig(r'../figures/couette_flow/vel_vectors_evolution.svg', bbox_inches='tight')

    # plt.show()


def plot_couette_flow_vel_vectors(lattice_grid_shape: Tuple[int, int] = (50, 50),
                                  omega: float = 0.5,
                                  U: float = 0.5,
                                  time_steps: int = 10000):
    assert U <= 1 / np.sqrt(3)
    lx, ly = lattice_grid_shape

    def boundary(f_pre_streaming, f_post_streaming, density):
        boundary_rigid_wall = np.zeros((lx, ly))
        boundary_rigid_wall[:, -1] = np.ones(ly)
        f_post_streaming = rigid_wall(boundary_rigid_wall.astype(np.bool))(f_pre_streaming, f_post_streaming)
        boundary_moving_wall = np.zeros((lx, ly))
        boundary_moving_wall[:, 0] = np.ones(ly)
        u_w = np.array(
            [U, 0]
        )
        f_post_streaming = moving_wall(boundary_moving_wall.astype(np.bool), u_w)(f_pre_streaming, f_post_streaming,
                                                                                  density)
        return f_post_streaming

    density, velocity = density_1_velocity_0_initial((lx, ly))
    f = equilibrium_distr_func(density, velocity)
    for i in range(time_steps):
        f, density, velocity = lattice_boltzman_step(f, density, velocity, omega, boundary)
    vx = velocity[..., 0]

    for vec, y_coord in zip(vx[int(lx / 2), :], np.arange(0, ly)):
        origin = [0, y_coord]
        plt.quiver(*origin, *[vec, 0.0], color='blue', scale_units='xy', scale=1, headwidth=3, width=0.0025)
    plt.plot(vx[int(lx / 2), :], np.arange(0, ly), label='Simul. Sol.', linewidth=1, c='blue', linestyle=':')
    plt.plot(U * (ly - 1 - np.arange(0, ly)) / ly, np.arange(0, ly), label='Analyt. Sol.', c='red',
             linestyle='--')
    max_vel = np.amax(vx[int(lx / 2), :]) + np.amax(vx[int(lx / 2), :]) * 0.1
    plt.plot(np.linspace(0, max_vel, 100), np.ones_like(np.linspace(0, max_vel, 100)) * (ly - 1) + 0.5,
             label='Rigid Wall', linewidth=1.5, c='orange',
             linestyle='-.')
    plt.plot(np.linspace(0, max_vel, 100), np.zeros_like(np.linspace(0, max_vel, 100)) - 0.5, label='Moving Wall',
             linewidth=1.5, c='green',
             linestyle='-')
    plt.ylabel('y coordinate')
    plt.xlabel('velocity in y-direction')
    plt.legend()

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.savefig(r'../figures/couette_flow/vel_vectors.svg', bbox_inches='tight')

    # plt.show()
