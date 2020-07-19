import unittest
import logging
import numpy as np

from src.lattice_boltzman_equation import compute_density, equilibrium_distr_func, compute_velocity_field, get_velocity_sets


class TestNavieStockEquations(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)

    def test_mass_conservation_1(self):
        """
        Test 1st mass conservation equation
        """
        prob_density_func = np.ones((10, 10, 9))/9
        density = compute_density(prob_density_func)
        velocity_field = compute_velocity_field(density, prob_density_func)
        f_eq = equilibrium_distr_func(density, velocity_field)
        self.assertAlmostEqual(np.unique(np.sum(f_eq, axis=-1)-density), [0], places=1)

    def test_mass_conservation_2(self):
        """
        Test 2nd mass conservation equation
        """
        f = np.random.random((10, 10, 9))/9
        density = compute_density(f)
        velocity_field = compute_velocity_field(density, f)
        f_eq = equilibrium_distr_func(density, velocity_field)
        tau = 1
        self.assertAlmostEqual(np.sum(-(1/tau)*(f-f_eq)), 0, places=1)

    def test_impulse_conservation_1(self):
        """
        Test 1st impulse conservation equation
        """
        prob_density_func = np.ones((10, 10, 9))/9
        density = compute_density(prob_density_func)
        velocity_field = compute_velocity_field(density, prob_density_func)
        f_eq = equilibrium_distr_func(density, velocity_field)
        c = get_velocity_sets()

        self.assertAlmostEqual(np.unique((f_eq @ c)-(np.expand_dims(density,axis=-1)*velocity_field)), [0], places=1)

    def test_impulse_conservation_2(self):
        """
        Test 2nd impulse conservation equation
        """

        c_i = get_velocity_sets()
        self.assertAlmostEqual(np.sum(c_i), 0, places=1)

if __name__ == "__main__":
    unittest.main()