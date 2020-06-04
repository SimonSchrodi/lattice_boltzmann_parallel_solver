import unittest
import logging
import numpy as np

from src.lattice_boltzman_equation import compute_density, streaming, lattice_boltzman_step, equilibrium_distr_func


class TestDensityComputation(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)

    def test_density_computation(self):
        """
        Test density computation
        """
        prob_density_func = np.ones((10, 10, 9))/9
        density = compute_density(prob_density_func)
        prob_density_func_new = streaming(prob_density_func)
        density_new = compute_density(prob_density_func_new)
        self.assertAlmostEqual(np.sum(density), np.sum(density_new), places=1)

    def test_lattice_boltzman_comp_long_(self):
        """
        Test density computation
        """
        density = np.ones((50, 50)) * 0.5
        density[24, 24] = 0.6
        velocity = np.ones((50, 50, 2)) * 0.0
        f = equilibrium_distr_func(density, velocity)
        omega = 0.5
        for _ in range(10000):
            f_new, density_new, velocity_new = lattice_boltzman_step(f, density, velocity, omega)
            self.assertAlmostEqual(np.sum(density), np.sum(density_new), places=1)
            f, density, velocity = f_new, density_new, velocity_new


if __name__ == "__main__":
    unittest.main()