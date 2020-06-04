import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np
from scipy.optimize import curve_fit

from src.initial_values import sinusoidal_density_x, sinusoidal_velocity_x
from src.lattice_boltzman_equation import equilibrium_distr_func, lattice_boltzman_step, lattice_boltzman_solver


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
                                     time_steps: int = 2000,
                                     omega_discretization: int = 10):
    fig, ax = plt.subplots(1, 2, sharex=True)
    omega = np.linspace(0.01, 1.99, omega_discretization)

    initial_distr_funcs = [sinusoidal_density_x(lattice_grid_shape, initial_p0, epsilon),
                           sinusoidal_velocity_x(lattice_grid_shape, epsilon)]

    for i, initial in enumerate(initial_distr_funcs):
        density, velocity = initial
        for om in omega:
            dens = []
            vels = []
            for i in range(time_steps):
                f, density, velocity = lattice_boltzman_step(f, density, velocity, omega)

                if i == 0:
                    pass
                elif i == 1:
                    pass

    plt.savefig(r'../figures/shear_wave_decay/meas_visc_vs_omega.svg')
