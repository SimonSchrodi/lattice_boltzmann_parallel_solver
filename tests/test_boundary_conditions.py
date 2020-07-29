import unittest
import logging
import numpy as np

from src.boundary_conditions import inlet, outlet, moving_wall, rigid_wall, periodic_with_pressure_variations
from src.lattice_boltzmann_method import equilibrium_distr_func

class TestMassPreservation(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)

    def test_inlet(self):
        """
        Test inlet boundary conditon
        """
        lattice_grid_shape = (10, 10)
        density_in = 1
        velocity_in = 0.1
        velocity = np.zeros(lattice_grid_shape + (2,))
        velocity[..., 0] = velocity_in
        f_eq = equilibrium_distr_func(
            np.ones(lattice_grid_shape) * density_in,
            velocity
        )
        f = np.zeros(lattice_grid_shape + (9,))
        f_new = inlet(lattice_grid_shape, density_in, velocity_in)(f.copy())
        self.assertTrue(np.allclose(f_new[0, ...], f_eq[0, ...]))
        self.assertTrue(np.allclose(f_new[1:, ...], f[1:, ...]))

    def test_outlet(self):
        """
        Test outlet boundary condition
        """
        lattice_grid_shape = (10, 10)
        f = np.zeros(lattice_grid_shape + (9,))
        f_prev = np.ones(lattice_grid_shape + (9,))
        f_new = outlet()(f_prev, f)
        for i in range(9):
            if i == 3 or i == 6 or i == 7:
                self.assertTrue(np.allclose(f_new[-1, ..., i], np.ones(10)))
            else:
                self.assertTrue(np.allclose(f_new[-1, ..., i], np.zeros(10)))

    def test_rigid_wall(self):
        """
        Test rigid wall boundary condition
        """
        lattice_grid_shape = (10, 10)
        f_after_streaming = np.zeros(lattice_grid_shape + (9,))
        f_pre_streaming = np.ones(lattice_grid_shape + (9,))
        boundary_rigid_wall = np.zeros(lattice_grid_shape)
        boundary_rigid_wall[:, -1] = np.ones(lattice_grid_shape[0])
        boundary_rigid_wall = boundary_rigid_wall.astype(np.bool)
        f = rigid_wall(boundary_rigid_wall)(f_pre_streaming, f_after_streaming)
        for i in range(9):
            if i == 4 or i == 7 or i == 8:
                self.assertTrue(np.allclose(f[boundary_rigid_wall, i], np.ones(10)))
            else:
                self.assertTrue(np.allclose(f[boundary_rigid_wall, i], np.zeros(10)))


    def test_moving_wall(self):
        """
        Test rigid wall boundary condition
        """
        lattice_grid_shape = (10, 10)
        f_after_streaming = np.zeros(lattice_grid_shape + (9,))
        f_pre_streaming = np.ones(lattice_grid_shape + (9,))
        boundary_moving_wall = np.zeros(lattice_grid_shape)
        boundary_moving_wall[:, -1] = np.ones(lattice_grid_shape[0])
        boundary_moving_wall = boundary_moving_wall.astype(np.bool)
        u_w = np.array(
            [2, 0]
        )
        avg_density = 1
        f = moving_wall(boundary_moving_wall, u_w, avg_density)(f_pre_streaming, f_after_streaming)
        for i in range(9):
            if i == 4:
                self.assertTrue(np.allclose(f[boundary_moving_wall, 4], np.ones(10)))
            elif i == 7:
                self.assertTrue(np.allclose(f[boundary_moving_wall, 7], np.ones(10)-1/3))
            elif i == 8:
                self.assertTrue(np.allclose(f[boundary_moving_wall, 8], np.ones(10))+1/3)
            else:
                self.assertTrue(np.allclose(f[boundary_moving_wall, i], np.zeros(10)))

    def test_periodic_with_pressure_variations(self):
        """
        Test rigid wall boundary condition
        """
        lattice_grid_shape = (10, 10)
        f_pre_streaming = np.ones(lattice_grid_shape + (9,))
        density = np.ones(lattice_grid_shape)
        velocity_in = 0.1
        velocity = np.zeros(lattice_grid_shape + (2,))
        velocity[..., 0] = velocity_in
        boundary = np.zeros(lattice_grid_shape)
        boundary[0, :] = np.ones(lattice_grid_shape[-1])
        boundary[-1, :] = np.ones(lattice_grid_shape[-1])
        boundary = boundary.astype(np.bool)
        p_in = 1
        p_out = 0.1
        f = periodic_with_pressure_variations(boundary, p_in, p_out)(f_pre_streaming, density, velocity)
        for i in range(9):
            if i == 1:
                self.assertTrue(np.allclose(f[0, ..., 1], np.ones(10)*1.29555))
                self.assertTrue(np.allclose(f[1:, ..., 1], np.ones((9, 10))))
            elif i == 5 or i == 8:
                self.assertTrue(np.allclose(f[0, ..., i], np.ones(10)*1.073888))
                self.assertTrue(np.allclose(f[1:, ..., i], np.ones((9, 10))))
            elif i == 8:
                self.assertTrue(np.allclose(f[0, ..., 8], np.ones(10) * 1.073888))
                self.assertTrue(np.allclose(f[1:, ..., 8], np.ones((9, 10))))
            elif i == 3:
                self.assertTrue(np.allclose(f[-1, ..., 3], np.ones(10) * 0.943222))
                self.assertTrue(np.allclose(f[:-1, ..., 3], np.ones((9, 10))))
            elif i == 6 or i == 7:
                self.assertTrue(np.allclose(f[-1, ..., i], np.ones(10) * 0.9858))
                self.assertTrue(np.allclose(f[:-1, ..., i], np.ones((9, 10))))
            else:
                self.assertTrue(np.allclose(f[..., i], np.ones((10, 10))))



if __name__ == "__main__":
    unittest.main()