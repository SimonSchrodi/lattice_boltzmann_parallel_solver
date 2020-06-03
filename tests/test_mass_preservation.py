import unittest
import logging
import numpy as np

from src.lattice_boltzman_equation import compute_density, streaming


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


if __name__ == "__main__":
    unittest.main()