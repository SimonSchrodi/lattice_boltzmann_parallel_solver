import numpy as np
from src.lattice_boltzman_equation import compute_density, compute_velocity_field
from src.visualizations import visualize_velocity_field

if __name__ == "__main__":
    for i in range(20):
        prob_density_func = np.zeros((10, 10, 9))
        prob_density_func[..., 5] = np.ones((10, 10))
        density = compute_density(prob_density_func)
        velocity = compute_velocity_field(density, prob_density_func)
        visualize_velocity_field(velocity, (10,10))


