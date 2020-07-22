import unittest
import logging
import numpy as np

from src.lattice_boltzmann_method import compute_velocity_field


class TestVelocityComputation(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)

    def test_velocity_computation(self):
        """
        Test velocity computation
        """
        density_func = np.ones((10, 10))
        prob_density_func = np.ones((10, 10, 9))/9
        velocity = compute_velocity_field(density_func, prob_density_func)
        self.assertEqual((10, 10, 2), velocity.shape)
        self.assertAlmostEqual(0, np.sum(velocity), places=1)


if __name__ == "__main__":
    unittest.main()