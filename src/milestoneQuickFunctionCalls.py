import numpy as np
from src.lattice_boltzman_equation import compute_density, compute_velocity_field, streaming, equilibrium_distr_func
from src.visualizations import visualize_velocity_field, visualize_density_contour_plot, visualize_density_surface_plot


def milestone_1():
    prob_density_func = np.zeros((10, 10, 9))
    prob_density_func[0:5, 0:5, 5] = np.ones((5, 5))
    for i in range(20):
        density = compute_density(prob_density_func)
        velocity = compute_velocity_field(density, prob_density_func)
        prob_density_func = streaming(prob_density_func)
        visualize_velocity_field(velocity, (10, 10))


def milestone_2_test_1():
    density = np.ones((50, 50))*0.5
    density[24, 24] = 0.6

    # visualize_density_surface_plot(density, (50, 50))

    velocity = np.ones((50, 50, 2))*0.0
    omega = 0.5

    f = equilibrium_distr_func(density, velocity, 9)

    for i in range(70):
        f_eq = equilibrium_distr_func(density, velocity, 9)
        f = streaming(f)

        # Collision / relaxation step
        f = f + (f_eq - f) * omega

        density = compute_density(f)
        velocity = compute_velocity_field(density, f)

        # visualize_density_contour_plot(density, (50, 50))
    visualize_density_surface_plot(density, (50, 50))


def milestone_2_test_2():
    density = np.random.uniform(0, 1, (50, 50))
    print(np.sum(density))
    velocity = np.random.uniform(-0.1, 0.1, (50, 50, 2))*0
    omega = 0.5

    f = equilibrium_distr_func(density, velocity, 9)

    for i in range(10000):
        f_eq = equilibrium_distr_func(density, velocity, 9)
        f = streaming(f)

        # Collision / relaxation step
        f = f + (f_eq - f) * omega

        density = compute_density(f)

        velocity = compute_velocity_field(density, f)

        # visualize_density_contour_plot(density, (50, 50))
        # if i&20 == 0:
        #    visualize_density_surface_plot(density, (50, 50))

    print(np.sum(density))
    visualize_density_surface_plot(density, (50, 50))
