import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from src.lattice_boltzman_equation import compute_density, compute_velocity_field, streaming, equilibrium_distr_func, \
    lattice_boltzman_step
from src.visualizations import visualize_velocity_field, visualize_density_contour_plot, visualize_density_surface_plot

from src.initial_values import milestone_2_test_1_initial_val, milestone_2_test_2_initial_val, shear_wave_decay_1, \
    shear_wave_decay_2


def milestone_1():
    lx, ly = 50, 50
    time_steps = 20

    prob_density_func = np.zeros((lx, ly, 9))
    prob_density_func[0:int(lx / 2), 0:int(ly / 2), 5] = np.ones((int(lx / 2), int(ly / 2)))
    for i in range(time_steps):
        density = compute_density(prob_density_func)
        velocity = compute_velocity_field(density, prob_density_func)
        prob_density_func = streaming(prob_density_func)
        visualize_velocity_field(velocity, (lx, ly))


def milestone_2_test_1():
    lx, ly = 50, 50
    time_steps = 70
    omega = 0.5
    density, velocity = milestone_2_test_1_initial_val((lx, ly))
    f = equilibrium_distr_func(density, velocity, 9)
    for i in range(time_steps):
        f, density, velocity = lattice_boltzman_step(f, density, velocity, omega)
    visualize_density_surface_plot(density, (lx, ly))


def milestone_2_test_2():
    lx, ly = 50, 50
    omega = 0.5
    time_steps = 10000

    density, velocity = milestone_2_test_2_initial_val((lx, ly))
    f = equilibrium_distr_func(density, velocity, 9)
    for i in range(time_steps):
        f, density, velocity = lattice_boltzman_step(f, density, velocity, omega)
    visualize_density_surface_plot(density, (lx, ly))


def milestone_3_test_1():
    lx, ly = 50, 50
    initial_p0 = 0.5
    epsilon = 0.1
    omega = 0.5
    time_steps = 1000

    density, velocity = shear_wave_decay_1((lx, ly), initial_p0, epsilon)
    f = equilibrium_distr_func(density, velocity, 9)
    visualize_density_surface_plot(density, (lx, ly))
    vels = []
    for i in range(time_steps):
        f, density, velocity = lattice_boltzman_step(f, density, velocity, omega)
        vel_min = np.amin(velocity)
        vel_max = np.amax(velocity)
        vels.append(
            np.abs(vel_min) if np.abs(vel_min) > np.abs(vel_max) else np.abs(vel_max)
        )

    x = t = np.arange(0, time_steps)
    vels = np.array(vels)
    viscosity_sim = curve_fit(lambda t, v: epsilon * np.exp(-v * np.power(2 * np.pi / lx, 2) * t), x, vels)[0][0]
    plt.plot(np.arange(0, time_steps), np.array(vels), label='Simulated (v=' + str(round(viscosity_sim, 3)) + ")")
    viscosity = (1 / 3) * (1 / omega - 0.5)
    plt.plot(t, epsilon * np.exp(-viscosity * np.power(2 * np.pi / lx, 2) * t),
             label='Analytical (v=' + str(viscosity) + ")")
    plt.legend()
    plt.xlabel('Time t')
    plt.ylabel('Amplitude a(t)')
    plt.show()


def milestone_3_test_2():
    lx, ly = 50, 50
    epsilon = 0.05
    omega = 1.5
    time_steps = 1000

    density, velocity = shear_wave_decay_2((lx, ly), epsilon)
    f = equilibrium_distr_func(density, velocity, 9)
    # visualize_density_surface_plot(density, (50, 50))
    vels = []
    for i in range(time_steps):
        f, density, velocity = lattice_boltzman_step(f, density, velocity, omega)
        # if i % 100 == 0:
        # visualize_density_surface_plot(density, (50, 50))
        # visualize_velocity_field(velocity, (50, 50))
        vel_min = np.amin(velocity)
        vel_max = np.amax(velocity)
        vels.append(
            np.abs(vel_min) if np.abs(vel_min) > np.abs(vel_max) else np.abs(vel_max)
        )

    x = t = np.arange(0, time_steps)
    vels = np.array(vels)
    viscosity_sim = curve_fit(lambda t, v: epsilon * np.exp(-v * np.power(2 * np.pi / lx, 2) * t), x, vels)[0][0]
    plt.plot(np.arange(0, time_steps), np.array(vels), label='Simulated (v=' + str(round(viscosity_sim, 3)) + ")")
    viscosity = (1 / 3) * (1 / omega - 0.5)
    plt.plot(t, epsilon * np.exp(-viscosity * np.power(2 * np.pi / lx, 2) * t),
             label='Analytical (v=' + str(viscosity) + ")")
    plt.legend()
    plt.xlabel('Time t')
    plt.ylabel('Amplitude a(t)')
    plt.show()


def milestone_4():
    lx, ly = 50, 50
    omega = 0.5
    time_steps = 1000

    density, velocity = shear_wave_decay_2((lx, ly), 0.08)
    f = equilibrium_distr_func(density, velocity, 9)
    visualize_density_surface_plot(density, (lx, ly))
    for i in range(time_steps):
        f, density, velocity = lattice_boltzman_step(f, density, velocity, omega)
        if i % 10 == 0:
            # visualize_density_surface_plot(density, (50, 50))
            visualize_velocity_field(velocity, (lx, ly))
