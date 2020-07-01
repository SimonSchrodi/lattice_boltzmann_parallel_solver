import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from src.lattice_boltzman_equation import compute_density, compute_velocity_field, streaming, equilibrium_distr_func, \
    lattice_boltzman_step
from src.visualizations import visualize_velocity_quiver, visualize_velocity_streamplot, \
    visualize_density_contour_plot, visualize_density_surface_plot

from src.initial_values import milestone_2_test_1_initial_val, milestone_2_test_2_initial_val, sinusoidal_density_x, \
    sinusoidal_velocity_x, density_1_velocity_0_initial

from src.boundary_conditions import rigid_wall, moving_wall, periodic_with_pressure_variations


def milestone_1():
    lx, ly = 50, 50
    time_steps = 20

    prob_density_func = np.zeros((lx, ly, 9))
    prob_density_func[0:int(lx / 2), 0:int(ly / 2), 5] = np.ones((int(lx / 2), int(ly / 2)))
    for i in range(time_steps):
        density = compute_density(prob_density_func)
        velocity = compute_velocity_field(density, prob_density_func)
        prob_density_func = streaming(prob_density_func)
        visualize_velocity_streamplot(velocity, (lx, ly))


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
    f = equilibrium_distr_func(density, velocity)
    for i in range(time_steps):
        f, density, velocity = lattice_boltzman_step(f, density, velocity, omega)
    visualize_density_surface_plot(density, (lx, ly))


def milestone_3_test_1():
    lx, ly = 50, 50
    initial_p0 = 0.5
    epsilon = 0.01
    omega = 1.95
    time_steps = 2000

    density, velocity = sinusoidal_density_x((lx, ly), initial_p0, epsilon)
    f = equilibrium_distr_func(density, velocity)
    # visualize_density_surface_plot(density, (lx, ly))
    dens = []
    for i in range(time_steps):
        f, density, velocity = lattice_boltzman_step(f, density, velocity, omega)
        den_min = np.amin(density)
        den_max = np.amax(density)
        dens.append(
            np.abs(den_min) - initial_p0 if np.abs(den_min) > np.abs(den_max) else np.abs(den_max) - initial_p0
        )
        # if i % 1000 == 0:
        #    visualize_density_surface_plot(density, (lx, ly))

    x = np.arange(0, time_steps)
    dens = np.array(dens)
    from scipy.signal import argrelextrema
    indizes = argrelextrema(np.array(dens), np.greater)
    a = dens[indizes]
    viscosity_sim = curve_fit(lambda t, v: epsilon * np.exp(-v * np.power(2 * np.pi / lx, 2) * t),
                              np.array(indizes).squeeze(), a)[0][0]
    plt.plot(np.array(indizes).squeeze(), a, label='Simulated (v=' + str(round(viscosity_sim, 3)) + ")")
    plt.plot(np.arange(0, time_steps), np.array(dens), label='Simulated True (v=' + str(round(viscosity_sim, 3)) + ")")
    viscosity = (1 / 3) * (1 / omega - 0.5)
    plt.plot(x, epsilon * np.exp(-viscosity * np.power(2 * np.pi / lx, 2) * x),
             label='Analytical (v=' + str(viscosity) + ")")
    plt.legend()
    plt.xlabel('Time t')
    plt.ylabel('Amplitude a(t)')
    plt.show()

    a = epsilon * np.exp(-viscosity_sim * np.power(2 * np.pi / lx, 2)) * x
    b = epsilon * np.exp(-viscosity * np.power(2 * np.pi / lx, 2)) * x

    print(viscosity * 0.8 <= viscosity_sim <= viscosity * 1.2)
    print(np.divide(
        viscosity - viscosity_sim,
        viscosity
    ))


def milestone_3_test_2():
    lx, ly = 50, 200
    epsilon = 0.08
    omega = 1.5
    time_steps = 2000

    density, velocity = sinusoidal_velocity_x((lx, ly), epsilon)
    f = equilibrium_distr_func(density, velocity)
    # visualize_density_surface_plot(velocity[..., 1], (lx, ly))
    vels = []
    for i in range(time_steps):
        f, density, velocity = lattice_boltzman_step(f, density, velocity, omega)
        # if i % 100 == 0:
        #    visualize_density_surface_plot(velocity[..., 0], (lx, ly))
        # visualize_velocity_field(velocity, (50, 50))
        vel_min = np.amin(velocity)
        vel_max = np.amax(velocity)
        vels.append(
            np.abs(vel_min) if np.abs(vel_min) > np.abs(vel_max) else np.abs(vel_max)
        )

    x = t = np.arange(0, time_steps)
    vels = np.array(vels)
    viscosity_sim = curve_fit(lambda t, v: epsilon * np.exp(-v * np.power(2 * np.pi / ly, 2) * t), x, vels)[0][0]
    plt.plot(np.arange(0, time_steps), np.array(vels), label='Simulated (v=' + str(round(viscosity_sim, 3)) + ")")
    viscosity = (1 / 3) * (1 / omega - 0.5)
    plt.plot(t, epsilon * np.exp(-viscosity * np.power(2 * np.pi / ly, 2) * t),
             label='Analytical (v=' + str(round(viscosity, 3)) + ")")
    plt.legend()
    plt.xlabel('Time t')
    plt.ylabel('Amplitude a(t)')
    plt.show()


def milestone_4():
    lx, ly = 50, 50
    omega = 0.5
    time_steps = 5000
    U = 50

    def boundary(f_pre_streaming, f_post_streaming, density, velocity):
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
    for i in range(time_steps):
        f, density, velocity = lattice_boltzman_step(f, density, velocity, omega, boundary)
    vx = velocity[..., 0]

    for vec, y_coord in zip(vx[25, :], np.arange(0, ly)):
        origin = [0, y_coord]
        plt.quiver(*origin, *[vec, 0.0], color='blue', scale_units='xy', scale=1, headwidth=3, width=0.0025)
    plt.plot(vx[25, :], np.arange(0, ly), label='Simulated Solution', linewidth=1, c='blue', linestyle=':')
    plt.plot(U * (ly - np.arange(0, ly + 1)) / ly, np.arange(0, ly + 1) - 0.5, label='Analyt. Sol.', c='red',
             linestyle='--')
    max_vel = np.ceil(np.amax(vx[25, :])).astype(np.int) + 1
    plt.plot(np.arange(0, max_vel), np.ones(max_vel) * (ly - 1) + 0.5, label='Rigid Wall', linewidth=1.5,
             c='orange', linestyle='-.')
    plt.plot(np.arange(0, max_vel), np.zeros(max_vel) - 0.5, label='Moving Wall', linewidth=1.5, c='green',
             linestyle='-')
    plt.ylabel('y coordinate')
    plt.xlabel('velocity in y-direction')
    plt.legend()
    plt.show()


def milestone_5():
    lx, ly = 200, 30
    omega = 1.5
    time_steps = 5000
    delta_p = 0.001111
    rho_0 = 1
    delta_rho = delta_p*3
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
    vels = []
    for i in range(time_steps):
        f, density, velocity = lattice_boltzman_step(f, density, velocity, omega, boundary)
        print(i, np.sum(density))
        vel_min = np.amin(velocity)
        vel_max = np.amax(velocity)
        vels.append(
            np.abs(vel_min) if np.abs(vel_min) > np.abs(vel_max) else np.abs(vel_max)
        )

    vx = velocity[..., 0]
    print(np.amax(velocity))
    x_coord = lx // 2
    centerline = ly // 2

    for vec, y_coord in zip(vx[x_coord, :], np.arange(0, ly)):
        origin = [0, y_coord]
        plt.quiver(*origin, *[vec, 0.0], color='blue', scale_units='xy', scale=1, headwidth=3, width=0.0025)
    plt.plot(vx[x_coord, :], np.arange(0, ly), label='Simulated Solution', linewidth=1, c='blue', linestyle=':')

    viscosity = (1 / 3) * (1 / omega - 0.5)
    dynamic_viscosity = viscosity * np.mean(density[x_coord, :])
    h = ly
    y = np.arange(0, ly + 1)
    dp_dx = np.divide(p_out - p_in, lx)
    uy = -np.reciprocal(2 * dynamic_viscosity) * dp_dx * y * (h - y)
    print(np.amax(uy))
    plt.plot(uy, y - 0.5, label='Analytical Solution', c='red',
             linestyle='--')
    plt.ylabel('y coordinate')
    plt.xlabel('velocity in y-direction')
    plt.legend()
    plt.show()

    plt.plot(np.arange(0, lx - 2), density[1:-1, centerline] / 3, label='Pressure along centerline')
    plt.plot(np.arange(0, lx - 2), np.ones_like(np.arange(0, lx - 2)) * p_out, label='Outgoing Pressure')
    plt.plot(np.arange(0, lx - 2), np.ones_like(np.arange(0, lx - 2)) * p_in,
             label='Ingoing Pressure')
    plt.xlabel('x coordinate')
    plt.ylabel('density along centerline')
    plt.legend()
    plt.show()

    print(np.amax(density[1:-1, centerline])/3, p_in)
    print(np.amin(density[1:-1, centerline])/3, p_out)
