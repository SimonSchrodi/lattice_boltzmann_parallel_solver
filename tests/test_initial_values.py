import unittest
import logging
import numpy as np

from src.initial_values import sinusoidal_density_x, sinusoidal_velocity_x


class TestDensityComputation(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)

    def test_shear_wave_decay_1(self):
        """
        Test density computation
        """
        lattice_grid_shape = (10, 10)
        initial_p0 = 0.5
        epsilon = 0.3

        rho_true = np.array(
            [
                0.5, 0.676, 0.785, 0.785, 0.676, 0.5, 0.324, 0.215, 0.215, 0.324
            ]
        )
        u_true = 0

        rho, u = sinusoidal_density_x(lattice_grid_shape=lattice_grid_shape, initial_p0=initial_p0, epsilon=epsilon)
        self.assertEqual(lattice_grid_shape, rho.shape)
        self.assertEqual(lattice_grid_shape + (2,), u.shape)
        for i in range(lattice_grid_shape[0]):
            for j in range(lattice_grid_shape[1]):
                self.assertAlmostEqual(rho[i, 0], rho[i, j], places=3)
                self.assertAlmostEqual(u_true, u[i, j, 0], places=3)
                self.assertAlmostEqual(u_true, u[i, j, 1], places=3)

        for i in range(lattice_grid_shape[0]):
            self.assertAlmostEqual(rho_true[i], rho[i, 0], places=3)

    def test_shear_wave_decay_2(self):
        """
        Test density computation
        """
        lattice_grid_shape = (10, 10)
        epsilon = 0.05

        rho_true = 1.0
        u_true = np.array(
            [
                0.0, 0.029, 0.048, 0.048, 0.029, 0.0, -0.029, -0.048, -0.048, -0.029
            ]
        )

        rho, u = sinusoidal_velocity_x(lattice_grid_shape=lattice_grid_shape, epsilon=epsilon)
        self.assertEqual(lattice_grid_shape, rho.shape)
        self.assertEqual(lattice_grid_shape + (2,), u.shape)

        for i in range(lattice_grid_shape[0]):
            for j in range(lattice_grid_shape[1]):
                self.assertAlmostEqual(u[0, i, 0], u[j, i, 0], places=3)
                self.assertAlmostEqual(rho_true, rho[i, j], places=3)

        for i in range(lattice_grid_shape[0]):
            self.assertAlmostEqual(u_true[i], u[0, i, 0], places=3)


if __name__ == "__main__":
    unittest.main()
