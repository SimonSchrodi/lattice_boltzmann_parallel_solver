import unittest
import logging
import numpy as np

from src.lattice_boltzman_equation import compute_density


class TestDensityComputation(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)

    def test_density_computation(self):
        """
        Test density computation
        """
        prob_density_func = np.ones((10, 10, 9))/9
        density = compute_density(prob_density_func)
        self.assertAlmostEqual(10*10, np.sum(density), places=1)


if __name__ == "__main__":
    unittest.main()