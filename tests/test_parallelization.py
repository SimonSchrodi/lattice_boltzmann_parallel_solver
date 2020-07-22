import numpy as np
from mpi4py import MPI
import os
import shutil

from tqdm import tqdm

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, './src')

from parallelization_utils import x_in_process, y_in_process, save_mpiio, global_to_local_direction, get_local_coords, global_coord_to_local_coord, communication, get_xy_size

from boundary_conditions import inlet, outlet
from boundary_utils import parallel_von_karman_boundary_conditions
from lattice_boltzmann_method import equilibrium_distr_func, lattice_boltzmann_step
from initial_values import density_1_velocity_x_u0_velocity_y_0_initial

def run_test():
    lx, ly = 420, 180
    d = 40
    u0 = 0.1
    density_in = 1.0
    kinematic_viscosity = 0.04
    omega = np.reciprocal(3 * kinematic_viscosity + 0.5)
    time_steps = 11

    p_coords = [3 * lx // 4, ly // 2]

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    comm = MPI.COMM_WORLD
    x_size, y_size = get_xy_size(size)

    cartesian2d = comm.Create_cart(dims=[x_size, y_size], periods=[True, True], reorder=False)
    coords2d = cartesian2d.Get_coords(rank)

    n_local_x, n_local_y = get_local_coords(coords2d, lx, ly, x_size, y_size)

    density, velocity = density_1_velocity_x_u0_velocity_y_0_initial((n_local_x + 2, n_local_y + 2), u0)
    f = equilibrium_distr_func(density, velocity)
    process_coord, px, py = global_coord_to_local_coord(coords2d, p_coords[0], p_coords[1], lx, ly, x_size, y_size)
    if process_coord is not None:
        vel_at_p = [np.linalg.norm(velocity[px, py, ...])]

    bound_func = parallel_von_karman_boundary_conditions(coords2d, n_local_x, n_local_y, lx, ly, x_size, y_size, density_in, u0, d)
    communication_func = communication(cartesian2d)

    if rank == 0:
        pbar = tqdm(total=time_steps)
    for i in range(time_steps):
        if rank == 0:
            pbar.update(1)
        f, density, velocity = lattice_boltzmann_step(f, density, velocity, omega, bound_func, communication_func)
        if process_coord is not None:
            vel_at_p.append(np.linalg.norm(velocity[px, py, ...]))
            vel_at_p_test = np.load(r'./tests/von_karman_vortex_shedding/vel_at_p.npy')
            assert vel_at_p[-1] == vel_at_p_test[i + 1]

        for j in range(9):
            save_mpiio(cartesian2d, r'./tests/tmp/f_' + str(j) + '.npy', f[1:-1, 1:-1, j])
        if rank == 0:
            f_gather = [np.load(r'./tests/tmp/f_' + str(j) + '.npy') for j in range(9)]
            f_gather = np.stack(f_gather, axis=-1)
            f_test = np.load(r'./tests/von_karman_vortex_shedding/f_' + str(i) + '.npy')
            assert f_gather.shape == f_test.shape
            assert np.allclose(f_gather, f_test)

if __name__ == "__main__":
    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        if not os.path.exists(r'./tests/tmp'):
            os.makedirs(r'./tests/tmp')
    run_test()
    if rank == 0:
        shutil.rmtree(r'./tests/tmp')